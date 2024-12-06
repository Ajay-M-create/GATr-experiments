import numpy as np
import torch


class SurfaceAreaDataset(torch.utils.data.Dataset):
    """Surface Area prediction dataset.

    Loads data generated with generate_surface_area_dataset.py from disk.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the npz file with the dataset to be loaded.
    subsample : None or float
        If not None, defines the fraction of the dataset to be used. For instance, `subsample=0.1`
        uses just 10% of the samples in the dataset.
    keep_extra : bool
        Whether to keep additional data like the point coordinates for visualization.
    """

    def __init__(self, filename, subsample=None, keep_extra=False):
        super().__init__()
        self.x, self.y, self.extra = self._load_data(
            filename, subsample, keep_extra=keep_extra
        )

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.x)

    def __getitem__(self, idx):
        """Returns the `idx`-th sample from the dataset."""
        if self.extra is not None:
            return self.x[idx], self.y[idx], self.extra[idx]
        return self.x[idx], self.y[idx]

    @staticmethod
    def _load_data(filename, subsample=None, keep_extra=False):
        """Loads data from file and converts to input and output tensors."""
        # Load data from file
        npz = np.load(filename, "r")
        points, surface_areas = npz["points"], npz["surface_areas"]

        # Convert to tensors
        points = torch.from_numpy(points).to(torch.float32)  # (num_samples, num_points, 3)
        surface_areas = torch.from_numpy(surface_areas).to(torch.float32).unsqueeze(1)  # (num_samples, 1)

        # Optionally, keep extra data (points)
        if keep_extra:
            extra = points.clone()
        else:
            extra = None

        # Subsample
        if subsample is not None and subsample < 1.0:
            n_original = len(points)
            n_keep = int(round(subsample * n_original))
            assert 0 < n_keep <= n_original
            points = points[:n_keep]
            surface_areas = surface_areas[:n_keep]
            if extra is not None:
                extra = extra[:n_keep]

        return points, surface_areas, extra 