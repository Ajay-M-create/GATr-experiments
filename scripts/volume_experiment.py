#!/usr/bin/env python3
# Added by Ajay

import hydra

from gatr.experiments.volume import VolumeExperiment


@hydra.main(config_path="../config", config_name="volume", version_base=None)
def main(cfg):
    """Entry point for volume experiment."""
    exp = VolumeExperiment(cfg)
    exp()


if __name__ == "__main__":
    main()