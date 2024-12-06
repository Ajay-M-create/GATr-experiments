# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
import torch

from gatr.experiments.base_wrapper import BaseWrapper
from gatr.interface import embed_point, extract_scalar
from tests_regression.regression_datasets.constants import DATASET_SIZE, DEVICE


class PointsDistanceDataset(torch.utils.data.Dataset):
    """Toy dataset that maps two points X, Y to the Euclidean distance |X - Y|."""

    def __init__(self):
        super().__init__()
        self._inputs = torch.randn((DATASET_SIZE, 1, 2, 3))
        self._target = torch.linalg.norm(
            (self._inputs[..., 1, :] - self._inputs[..., 0, :]), dim=-1, keepdim=True
        )

        # If there's space on the GPU, let's keep the data on the GPU
        try:
            self._inputs.to(DEVICE)
            self._target.to(DEVICE)
        except RuntimeError:
            pass

    def __len__(self):
        """Return number of samples."""
        return len(self._inputs)

    def __getitem__(self, idx):
        """Return datapoint."""
        return self._inputs[idx], self._target[idx]


class PointsDistanceWrapper(BaseWrapper):
    """Wrapper around GATr networks for PointsDistanceDataset."""

    mv_in_channels = 2
    mv_out_channels = 1
    s_in_channels = 1
    s_out_channels = 1
    raw_in_channels = 3
    raw_out_channels = 1

    def __init__(self, net):
        super().__init__(net, scalars=True, return_other=False)

    def embed_into_ga(self, inputs):
        """Embeds raw inputs into the geometric algebra (+ scalar) representation.

        To be implemented by subclasses.

        Parameters
        ----------
        inputs : torch.Tensor
            Raw inputs, as given by dataset.

        Returns
        -------
        mv_inputs : torch.Tensor
            Multivector inputs, as expected by geometric network.
        scalar_inputs : torch.Tensor or None
            Scalar inputs, as expected by geometric network.
        """

        num_objects, num_features = inputs.shape[-2:]
        assert num_objects == 2
        assert num_features == 3

        multivector = embed_point(inputs)  # (batchsize, 1, 2, 16)

        # Note that we're encoding the two points as two different *channels*, not two different
        # objects. The reason is that an object-permutation-invariant function cannot learn the
        # translation vector between two objects (but we have no invariance across channel
        # dimensions).

        scalars = torch.zeros((*inputs.shape[:-2], 1), device=inputs.device)  # (batchsize, 1, 1)

        return multivector, scalars

    def extract_from_ga(self, multivector, scalars):
        """Embeds raw inputs into the geometric algebra (+ scalar) representation.

        To be implemented by subclasses.

        Parameters
        ----------
        multivector : torch.Tensor
            Multivector outputs from geometric network.
        scalars : torch.Tensor or None
            Scalar outputs from geometric network.

        Returns
        -------
        outputs : torch.Tensor
            Raw outputs, as expected in dataset.
        other : torch.Tensor
            Additional output data, e.g. required for regularization.
        """

        num_channels, num_ga_components = multivector.shape[-2:]
        assert num_channels == 1
        assert num_ga_components == 16

        norm = extract_scalar(multivector[..., 0, :])  # (batchsize, 1, 3)

        return norm, None
