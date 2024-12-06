#!/usr/bin/env python3
# Added by Ajay

import hydra

from gatr.experiments.symmetry import SymmetryExperiment


@hydra.main(config_path="../config", config_name="symmetry", version_base=None)
def main(cfg):
    """Entry point for symmetry experiment."""
    exp = SymmetryExperiment(cfg)
    exp()


if __name__ == "__main__":
    main()