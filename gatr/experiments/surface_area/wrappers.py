# Added by Ajay

# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
import torch
from torch import nn

from gatr.baselines.transformer import BaselineTransformer
from gatr.experiments.base_wrapper import BaseWrapper
from gatr.interface import embed_point, extract_scalar


class SurfaceAreaGATrWrapper(BaseWrapper):
    """Wraps around GATr for the surface area prediction experiment.

    Parameters
    ----------
    net : torch.nn.Module
        GATr model that accepts inputs with multivector channels and returns multivector outputs.
    """

    def __init__(self, net):
        super().__init__(net, scalars=False, return_other=True)
        self.supports_variable_items = False  # Fixed number of points

    def embed_into_ga(self, inputs):
        """Embeds raw inputs into the geometric algebra representation.

        Parameters
        ----------
        inputs : torch.Tensor with shape (batchsize, num_points, 3)
            Input points.

        Returns
        -------
        mv_inputs : torch.Tensor
            Multivector representation.
        scalar_inputs : None
        """
        batchsize, num_points, _ = inputs.shape

        # Embed points into multivectors
        multivector = embed_point(inputs)  # (batchsize, num_points, 16)
        multivector = multivector.unsqueeze(2)  # (batchsize, num_points, 1, 16)

        return multivector, None

    def extract_from_ga(self, multivector, scalars):
        """Extracts predicted surface areas from the GATr multivector outputs.

        Parameters
        ----------
        multivector : torch.Tensor
            Multivector outputs from GATr.
        scalars : None
            Not used.

        Returns
        -------
        outputs : torch.Tensor
            Predicted surface areas, shape (batchsize, 1).
        other : torch.Tensor
            Regularization terms.
        """

        # Assume the first grade is related to surface area
        scalar_output = extract_scalar(multivector[:, :, 0, :])  # (batchsize, num_points)

        # Aggregate the scalar outputs across all points to get a single surface area prediction per sample
        # Here, we use mean aggregation. You can choose sum, max, or another aggregation method if appropriate.
        scalar_output = scalar_output.mean(dim=1, keepdim=True)  # (batchsize, 1)

        # For regularization, sum of squares of multivector components
        reg = torch.sum(multivector ** 2, dim=[1, 2, 3])  # (batchsize,)

        return scalar_output, reg


class SurfaceAreaTransformerWrapper(nn.Module):
    """Wraps around a baseline Transformer for the surface area prediction experiment.

    Parameters
    ----------
    net : torch.nn.Module
        Transformer model that accepts inputs with 5x3 coordinates and returns outputs with 1 scalar.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.supports_variable_items = False  # Fixed number of points

    def forward(self, inputs):
        """Wrapped forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            Input points, shape (batchsize, num_points, 3).

        Returns
        -------
        outputs : torch.Tensor
            Predicted surface areas, shape (batchsize, 1).
        other : torch.Tensor
            Dummy term, since the baseline does not require regularization.
        """
        batchsize = inputs.shape[0]
        predictions = self.net(inputs)  # (batchsize, num_points, 1)
        predictions = predictions.mean(dim=1, keepdim=True)  # (batchsize, 1)
        predictions = predictions.squeeze(-1)  # (batchsize,)
        return predictions, torch.zeros(batchsize, device=inputs.device) 