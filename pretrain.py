import omegaconf
from data2vec.trainer import mcRNA_Trainer

if __name__ == "__main__":
    import argparse
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to yaml config file")
    args = parser.parse_args()

    cfg_path = args.config
    cfg = omegaconf.OmegaConf.load(cfg_path)
    trainer = mcRNA_Trainer(cfg)
    trainer.train()
