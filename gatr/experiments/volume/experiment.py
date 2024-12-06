# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
from pathlib import Path
import os
from collections import defaultdict
from typing import Dict
from tqdm import tqdm
import pandas as pd
import numpy as np

import torch
import matplotlib.pyplot as plt

from gatr.experiments.base_experiment import BaseExperiment
from gatr.experiments.volume.dataset import VolumeDataset
from gatr.utils.misc import get_batchsize
from gatr.utils.logger import logger
from gatr.utils.mlflow import log_mlflow


class VolumeExperiment(BaseExperiment):
    """Experiment manager for volume prediction.

    Parameters
    ----------
    cfg : OmegaConf
        Experiment configuration. See the config folder in the repository for examples.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self._mse_criterion = torch.nn.MSELoss()
        self._mae_criterion = torch.nn.L1Loss(reduction="mean")
        self._predictions = {}
        self._actuals = {}
        self._current_eval_tag = None

    def _load_dataset(self, tag):
        """Loads dataset.

        Parameters
        ----------
        tag : str
            Dataset tag, like "train", "val", or one of self._eval_tags.

        Returns
        -------
        dataset : torch.utils.data.Dataset
            Dataset.
        """
        if tag == "train":
            subsample_fraction = self.cfg.data.subsample
        else:
            subsample_fraction = None

        filename = Path(self.cfg.data.data_dir) / f"{tag}.npz"
        keep_extra = tag == "val"
        return VolumeDataset(
            filename, subsample=subsample_fraction, keep_extra=keep_extra
        )

    def _forward(self, *data):
        """Model forward pass.

        Parameters
        ----------
        data : tuple of torch.Tensor
            Data batch.

        Returns
        -------
        loss : torch.Tensor
            Loss
        metrics : dict with str keys and float values
            Additional metrics for logging
        """

        # Forward pass
        assert self.model is not None

        x, y = data[0], data[1]
        y_pred, reg = self.model(x)

        # Reshape y_pred and y to remove any extra dimensions
        y_pred = y_pred.squeeze()
        y = y.squeeze()

        # Store predictions and actuals for plotting
        self._store_predictions(y_pred.detach().cpu(), y.detach().cpu())

        # Compute loss
        mse = self._mse_criterion(y_pred, y)
        output_reg = torch.mean(reg)
        loss = mse + self.cfg.training.output_regularization * output_reg

        # Additional metrics
        mae = self._mae_criterion(y_pred, y)
        metrics = dict(
            mse=mse.item(),
            rmse=(mse + self.cfg.training.output_regularization * output_reg).item() ** 0.5,
            output_reg=output_reg.item(),
            mae=mae.item()
        )

        return loss, metrics

    def _compute_metrics(self, dataloader):
        """Given a dataloader, computes all relevant metrics. Can be adapted by subclasses.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            Dataloader.

        Returns
        -------
        metrics : dict with str keys and float values
            Metrics computed from dataset.
        """
        # Move to eval mode and eval device
        assert self.model is not None
        self.model.eval()
        eval_device = torch.device(self.cfg.training.eval_device)
        self.model = self.model.to(eval_device)

        aggregate_metrics: Dict[str, float] = defaultdict(float)

        # Reset predictions and actuals for this evaluation
        current_tag = self._current_eval_tag
        self._predictions[current_tag] = []
        self._actuals[current_tag] = []

        # Loop over dataset and compute error
        with torch.no_grad():  # Add no_grad context for evaluation
            for data in tqdm(dataloader, disable=False, desc="Evaluating"):
                data = self._prep_data(data, device=eval_device)

                # Forward pass
                loss, metrics = self._forward(*data)

                # Weight for this batch (last batches may be smaller)
                batchsize = get_batchsize(data[0])
                weight = batchsize / len(dataloader.dataset)

                # Book-keeping
                aggregate_metrics["loss"] += loss.item() * weight
                for key, val in metrics.items():
                    aggregate_metrics[key] += val * weight

        # Move model back to training mode and training device
        self.model.train()
        self.model = self.model.to(self.device)

        # After evaluation, plot predictions vs actuals if we have data
        if self._predictions[current_tag]:
            try:
                self._plot_predictions(current_tag)
            except Exception as e:
                logger.error(f"Error plotting predictions for tag '{current_tag}': {str(e)}")

        # Return metrics
        return aggregate_metrics

    def _store_predictions(self, y_pred, y_true):
        """Stores predictions and actuals for the current evaluation tag."""
        if self._current_eval_tag is None:
            # If _current_eval_tag is not set, skip storing predictions
            return

        if self._current_eval_tag in self._predictions:
            self._predictions[self._current_eval_tag].append(y_pred)
            self._actuals[self._current_eval_tag].append(y_true)
        else:
            self._predictions[self._current_eval_tag] = [y_pred]
            self._actuals[self._current_eval_tag] = [y_true]

    def _plot_predictions(self, tag, step=None):
        """Plots predictions vs actual volumes and saves the plot as a .png file.

        Parameters
        ----------
        tag : str
            Dataset tag for which the plot is being created.
        step : int, optional
            Current training step, used for filename
        """
        # Check if we have predictions to plot
        if not self._predictions[tag] or not self._actuals[tag]:
            logger.warning(f"No predictions available for tag '{tag}'. Skipping plot generation.")
            return

        # Concatenate all predictions and actuals
        preds = torch.cat(self._predictions[tag], dim=0).numpy()
        actuals = torch.cat(self._actuals[tag], dim=0).numpy()
        
        # Get model type from the experiment config class path
        if hasattr(self.cfg.model, '_target_'):
            model_type = self.cfg.model._target_.split('.')[-1]  # Gets the class name from the full path
        else:
            model_type = "Unknown Model"
        
        # Get task type from the experiment class name
        task_type = self.__class__.__name__.replace('Experiment', '')
        
        plt.figure(figsize=(8, 6))
        plt.scatter(actuals, preds, alpha=0.5)
        plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        
        # Calculate and add metric annotations
        mse = np.mean((preds - actuals) ** 2)
        mae = np.mean(np.abs(preds - actuals))
        annotation_text = f"MSE: {mse:.2e}\nMAE: {mae:.2e}"
        plt.text(0.95, 0.05, annotation_text, horizontalalignment='right', 
                verticalalignment='bottom', transform=plt.gca().transAxes, 
                fontsize=10, bbox=dict(facecolor='white', alpha=0.8, 
                                     boxstyle='round,pad=0.5'))
        
        # Create title
        main_title = f'{model_type} - {task_type}'
        subtitle = f'Predictions vs Actuals ({tag.capitalize()})'
        if step is not None:
            subtitle += f' - Step {step}'
            
        plt.title(f'{main_title}\n{subtitle}', pad=15)
        plt.grid(True)

        # Ensure the metrics directory exists
        metrics_dir = Path(self.cfg.exp_dir) / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        # Include step in filename if available
        filename = f'predictions_vs_actuals_{tag}'
        if step is not None:
            filename += f'_step_{step}'
        filename += '.png'
        
        plot_path = metrics_dir / filename
        plt.savefig(plot_path)
        plt.close()

        logger.info(f"Saved predictions vs actuals plot for '{tag}' at {plot_path}")

    def evaluate(self):
        """Evaluates self.model on all eval datasets and logs the results."""

        # Should we evaluate with EMA in addition to without?
        ema_values = [False]
        if self.ema is not None:
            ema_values.append(True)

        # Loop over evaluation datasets
        dfs = {}
        for tag in self._eval_dataset_tags:
            self._current_eval_tag = tag  # Set current tag for storing predictions
            dataset = self._load_dataset(tag)
            eval_batchsize = self.cfg.training.get("eval_batchsize", self.cfg.training.batchsize)
            dataloader = self._make_data_loader(dataset, batch_size=eval_batchsize, shuffle=False)

            # Initialize lists to store predictions and actuals
            self._predictions[tag] = []
            self._actuals[tag] = []

            # Loop over EMA on / off
            for ema in ema_values:
                # Effective tag name
                full_tag = (tag + "_ema") if ema else tag

                # Run evaluation
                if ema:
                    with self.ema.average_parameters():
                        metrics = self._compute_metrics(dataloader)
                else:
                    metrics = self._compute_metrics(dataloader)

                # Log results
                self.metrics[full_tag] = metrics
                logger.info(f"Ran evaluation on dataset {full_tag}:")
                for key, val in metrics.items():
                    logger.info(f"    {key} = {val}")
                    log_mlflow(f"eval.{full_tag}.{key}", val)

                # Store results in csv file
                # Pandas does not like scalar values, have to be iterables
                test_metrics_ = {key: [val] for key, val in metrics.items()}
                df = pd.DataFrame.from_dict(test_metrics_)
                df.to_csv(Path(self.cfg.exp_dir) / "metrics" / f"eval_{full_tag}.csv", index=False)
                dfs[full_tag] = df
        return dfs

    @property
    def _eval_dataset_tags(self):
        """Eval dataset tags.

        Returns
        -------
        tags : iterable of str
            Eval dataset tags
        """
        return {"test", "generalization"}

    def validate(self, dataloader, step):
        """Validates the model on the validation set.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            Validation dataloader
        step : int
            Current training step
        """
        # Set current eval tag to 'val' for validation
        self._current_eval_tag = 'val'
        
        # Initialize storage for this validation step
        self._predictions['val'] = []
        self._actuals['val'] = []
        
        # Compute metrics
        metrics = self._compute_metrics(dataloader)
        
        # Plot with step information
        if self._predictions['val']:
            try:
                self._plot_predictions('val', step)
            except Exception as e:
                logger.error(f"Error plotting validation predictions at step {step}: {str(e)}")
        
        # Log validation metrics
        for key, val in metrics.items():
            log_mlflow(f"val.{key}", val, step=step)
        
        return metrics
