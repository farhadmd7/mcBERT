import torch.nn as nn
from transformers import AutoModel, RobertaConfig


class McBERT_Encoder(nn.Module):
    def __init__(self, cfg, out_as_dict=True, **kwargs):
        super(McBERT_Encoder, self).__init__()
        self.cfg = cfg
        self.dense_pre_process = nn.Sequential(
            nn.Linear(cfg.model.vocab_size, cfg.model.embed_dim),
            nn.LayerNorm(cfg.model.embed_dim, eps=1e-5, elementwise_affine=True),
            nn.Dropout(0.1, inplace=False),
        )

        configuration = RobertaConfig(
            hidden_size=cfg.model.hidden_size,
            num_attention_heads=cfg.model.num_attention_heads,
            num_hidden_layers=cfg.model.num_hidden_layers,
        )

        self.encoder = AutoModel.from_config(configuration)

        # Omit the embedding and only use the Transformer encoder layers
        self.pooler = self.encoder.pooler
        self.encoder = self.encoder.encoder
        self.out_as_dict = out_as_dict
        self.__dict__.update(kwargs)

    def forward(self, inputs, mask=None, **kwargs):
        """
        Forward inputs through the encoder and extract transformer/attention layers outputs

        Args:
            inputs: source tokens
            mask: bool masked indices
            kwargs: keyword args specific to the encoder's forward method

        Returns:
            A dictionary of the encoder outputs including transformer layers outputs and attentions outputs

        """
        # Note: inputs are already masked for MLM so mask is not used
        pre_processed = self.dense_pre_process(inputs)
        outputs = self.encoder(
            pre_processed, output_hidden_states=True, output_attentions=True, **kwargs
        )
        outputs["pooler_output"] = self.pooler(outputs["hidden_states"][-1])

        encoder_states = outputs["hidden_states"][
            :-1
        ]  # encoder layers outputs separately
        encoder_out = outputs["hidden_states"][-1]  # last encoder output (accumulated)
        attentions = outputs["attentions"]
        if self.out_as_dict:
            return {
                "encoder_states": encoder_states,
                "encoder_out": encoder_out,
                "attentions": attentions,
            }
        else:
            # return encoder_out[:, 0] # use [CLS]
            return encoder_out.mean(dim=1)  # use GAP mean pooling
