#!python

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    # load environment variables from `.env` file if it exists
    # recursively searches for `.env` in all folders starting from work dir
    dotenv=True,
)


import hydra
from omegaconf import DictConfig


@hydra.main(
    version_base="1.1", config_path=f"{root}/configs", config_name="config.yaml"
)
def main(config: DictConfig):
    from byprot import utils
    from byprot.training_pipeline import train

    # Applies optional utilities
    config = utils.extras(config)

    # Train model
    return train(config)


if __name__ == "__main__":
    main()
