#!/usr/bin/env python3

from pathlib import Path
import hydra
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

def compute_symmetry_score(points):
    """Compute symmetry score for a set of convex hull points."""
    centroid = np.mean(points, axis=0)
    reflected_points = 2 * centroid - points
    distances = cdist(points, reflected_points)
    min_distances = np.min(distances, axis=1)
    bbox_size = np.max(points, axis=0) - np.min(points, axis=0)
    diagonal = np.sqrt(np.sum(bbox_size**2))
    symmetry_score = np.exp(-np.mean(min_distances) / (diagonal * 0.1))
    return symmetry_score

def generate_symmetry_dataset(filename, num_samples, num_points=5):
    """Generates a dataset of random convex hulls and their symmetry scores."""
    assert not Path(filename).exists(), f"File {filename} already exists!"
    symmetries = []
    points = []

    for _ in range(num_samples):
        # Generate random points in [0, 1]^3
        pts = np.random.uniform(0, 1, size=(num_points, 3))
        points.append(pts)

        # Compute symmetry score
        try:
            hull = ConvexHull(pts)
            convex_hull_points = pts[hull.vertices]
            score = compute_symmetry_score(convex_hull_points)
            symmetries.append(score)
        except Exception:
            # If points are degenerate, symmetry is zero
            symmetries.append(0.0)

    points = np.array(points)  # Shape: (num_samples, num_points, 3)
    symmetries = np.array(symmetries)  # Shape: (num_samples,)

    # Plot histogram of symmetry scores
    plt.figure(figsize=(10, 6))
    plt.hist(symmetries, bins=50, density=True)
    plt.xlabel('Symmetry Score')
    plt.ylabel('Density')
    plt.title(f'Distribution of Symmetry Scores (n={num_samples})')
    plt.savefig(str(Path(filename).parent / f'symmetry_dist_{Path(filename).stem}.png'))
    plt.close()

    np.savez(filename, points=points, symmetries=symmetries)

def generate_datasets(path):
    """Generates train, validation, test, and generalization datasets.

    Parameters
    ----------
    path : str or pathlib.Path
        Directory to save the datasets.
    """
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=True)

    print(f"Creating symmetry datasets in {str(path)}")

    # Training dataset
    generate_symmetry_dataset(path / "train.npz", num_samples=100000, num_points=5)

    # Validation dataset
    generate_symmetry_dataset(path / "val.npz", num_samples=5000, num_points=5)

    # Test dataset
    generate_symmetry_dataset(path / "test.npz", num_samples=5000, num_points=5)

    # Generalization dataset (e.g., with more points)
    generate_symmetry_dataset(path / "generalization.npz", num_samples=5000, num_points=10)

    print("Dataset generation complete!")

@hydra.main(config_path="../config", config_name="symmetry", version_base=None)
def main(cfg):
    """Entry point for symmetry dataset generation."""
    data_dir = cfg.data.data_dir
    np.random.seed(cfg.seed)
    generate_datasets(data_dir)

if __name__ == "__main__":
    main()
