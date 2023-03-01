from argparse import ArgumentParser

from omegaconf import OmegaConf
from utils.config import instantiate_from_config

from pytorch_lightning import Trainer

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('config', type=str, help='Path to config file')
    
    return parser.parse_args()

def main(args):
    config = OmegaConf.load(args.config)
    model = instantiate_from_config(config.model)
    datamodule = instantiate_from_config(config.datamodule)

    trainer = Trainer(**config.trainer)
    trainer.fit(model, datamodule)

if __name__ == '__main__':
    args = parse_args()
    main(args)
