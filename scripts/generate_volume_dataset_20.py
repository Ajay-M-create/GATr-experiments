#!/usr/bin/env python3
# Added by Ajay

from pathlib import Path
import hydra
import numpy as np
import torch
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

def generate_volume_dataset(filename, num_samples, num_points=20):
    """Generates a dataset of random 20-point convex hulls and their volumes.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to save the generated dataset (npz format).
    num_samples : int
        Number of samples to generate.
    num_points : int, optional
        Number of points per sample, set to 20.
    """
    assert not Path(filename).exists(), f"File {filename} already exists!"

    volumes = []
    points = []

    for _ in range(num_samples):
        # Generate random points in [0, 1]^3
        pts = np.random.uniform(0, 1, size=(num_points, 3))
        points.append(pts)

        # Compute convex hull volume
        try:
            hull = ConvexHull(pts)
            volumes.append(hull.volume)
        except Exception:
            # If points are coplanar or degenerate, volume is zero
            volumes.append(0.0)

    points = np.array(points)  # Shape: (num_samples, num_points, 3)
    volumes = np.array(volumes)  # Shape: (num_samples,)

    # Plot histogram of volumes
    plt.figure(figsize=(10, 6))
    plt.hist(volumes, bins=50, density=True)
    plt.xlabel('Volume')
    plt.ylabel('Density')
    plt.title(f'Distribution of Volumes (n={num_samples})')
    plt.savefig(str(Path(filename).parent / f'volume_dist_{Path(filename).stem}.png'))
    plt.close()

    np.savez(filename, points=points, volumes=volumes)


def generate_datasets(path):
    """Generates train, validation, test, and generalization datasets with 20 points.

    Parameters
    ----------
    path : str or pathlib.Path
        Directory to save the datasets.
    """
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=True)

    print(f"Creating volume datasets in {str(path)}")

    # Training dataset
    generate_volume_dataset(path / "train.npz", num_samples=100000, num_points=20)

    # Validation dataset
    generate_volume_dataset(path / "val.npz", num_samples=5000, num_points=20)

    # Test dataset
    generate_volume_dataset(path / "test.npz", num_samples=5000, num_points=20)

    # Generalization dataset (e.g., with more points)
    generate_volume_dataset(path / "generalization.npz", num_samples=5000, num_points=20)

    print("Dataset generation complete!")


@hydra.main(config_path="../config", config_name="volume", version_base=None)
def main(cfg):
    """Entry point for volume dataset generation with 20 points."""
    data_dir = cfg.data.data_dir_20
    np.random.seed(cfg.seed)
    generate_datasets(data_dir)


if __name__ == "__main__":
    main()