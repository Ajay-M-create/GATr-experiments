#!/usr/bin/env python3
# Added by Ajay

import hydra

from gatr.experiments.surface_area import SurfaceAreaExperiment


@hydra.main(config_path="../config", config_name="surface_area", version_base=None)
def main(cfg):
    """Entry point for surface area experiment."""
    exp = SurfaceAreaExperiment(cfg)
    exp()


if __name__ == "__main__":
    main() 