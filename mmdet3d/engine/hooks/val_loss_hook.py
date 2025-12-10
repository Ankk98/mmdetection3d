# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import Dict, Sequence

import torch
from mmengine.hooks import Hook
from mmengine.logging import print_log
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner

from mmdet3d.registry import HOOKS
from mmdet3d.structures import Det3DDataSample


@HOOKS.register_module()
class ValLossHook(Hook):
    """Hook to compute and log validation loss during validation.

    This hook computes validation loss by running the model in eval mode
    and calling the loss function on validation batches.

    Args:
        interval (int): The interval of validation iterations to log loss.
            Defaults to 1 (log every iteration).
    """

    def __init__(self, interval: int = 1):
        self.interval = interval
        self.val_losses = []

    def before_val_epoch(self, runner: Runner) -> None:
        """Called before validation epoch starts."""
        self.val_losses = []

    def after_val_iter(self, runner: Runner, batch_idx: int,
                       data_batch: dict,
                       outputs: Sequence[Det3DDataSample]) -> None:
        """Compute validation loss after each validation iteration.

        Args:
            runner: The runner of the validation process.
            batch_idx: The index of the current batch in the val loop.
            data_batch: Data from dataloader (should contain ground truth).
            outputs: A batch of data samples that contain predictions.
        """
        if batch_idx % self.interval != 0:
            return

        # Get the model and unwrap if it's wrapped (e.g., MMDistributedDataParallel)
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        # Check if data_batch has ground truth annotations
        # In MMDetection3D, data_batch typically has 'inputs' and 'data_samples'
        if 'data_samples' not in data_batch:
            return

        # Prepare inputs for loss computation
        # Extract inputs from data_batch - MMDetection3D uses 'inputs' key
        if 'inputs' not in data_batch:
            return

        inputs = data_batch['inputs']
        data_samples = data_batch['data_samples']

        # Check if data_samples have ground truth (gt_instances_3d or gt_bboxes_3d)
        has_gt = False
        for sample in data_samples:
            if hasattr(sample, 'gt_instances_3d') or hasattr(sample, 'gt_bboxes_3d'):
                has_gt = True
                break

        if not has_gt:
            # No ground truth available, skip loss computation
            return

        # Check if required input keys are present (e.g., 'voxels' for VoxelNet/PointPillars)
        # Some models require 'voxels' key which may be missing for empty point clouds
        # or failed voxelization. Skip loss computation if required keys are missing.
        # VoxelNet-based models (like PointPillars) require 'voxels' in inputs
        if hasattr(model, 'voxel_encoder'):
            # This is a VoxelNet model that requires voxels
            if 'voxels' not in inputs:
                # Voxels are missing - this can happen for empty point clouds or
                # voxelization failures. Skip this batch to avoid KeyError.
                return

        # Compute loss
        try:
            # Keep model in eval mode for validation
            # Enable gradients temporarily for loss computation (some losses need it)
            # but we won't backpropagate
            with torch.enable_grad():
                # Call model's loss method
                losses = model.loss(inputs, data_samples)

                # Store losses (detach to avoid keeping computation graph)
                if isinstance(losses, dict):
                    # Convert tensors to values for logging
                    loss_dict = {}
                    for key, value in losses.items():
                        if isinstance(value, torch.Tensor):
                            loss_dict[key] = value.detach().cpu().item()
                        else:
                            loss_dict[key] = value
                    self.val_losses.append(loss_dict)
                elif isinstance(losses, list) and len(losses) > 0:
                    # If losses is a list, aggregate any mix of tensors and scalars
                    total_loss = 0.0
                    valid_items = 0
                    for item in losses:
                        if isinstance(item, torch.Tensor):
                            total_loss += item.detach().cpu().item()
                            valid_items += 1
                        elif isinstance(item, (float, int)):
                            total_loss += float(item)
                            valid_items += 1
                    if valid_items > 0:
                        self.val_losses.append({'loss': total_loss})
        except Exception as e:
            # If loss computation fails, skip this batch
            print_log(
                f'Warning: Failed to compute validation loss at batch {batch_idx}: {e}',
                logger='current',
                level=logging.WARNING)
            return

    def after_val_epoch(self, runner: Runner, metrics: Dict = None) -> None:
        """Log average validation loss after validation epoch.

        Args:
            runner: The runner of the validation process.
            metrics: Dictionary of evaluation metrics.
        """
        if len(self.val_losses) == 0:
            return

        # Aggregate losses across all batches
        aggregated_losses = {}
        for loss_dict in self.val_losses:
            for key, value in loss_dict.items():
                # Values should already be scalars from after_val_iter
                if key not in aggregated_losses:
                    aggregated_losses[key] = []
                aggregated_losses[key].append(value)

        # Compute averages
        avg_losses = {}
        for key, values in aggregated_losses.items():
            avg_losses[f'val/{key}'] = sum(values) / len(values)

        # Log the losses
        runner.logger.info('Validation Losses:')
        for key, value in avg_losses.items():
            runner.logger.info(f'  {key}: {value:.6f}')
            # Also add to message_hub for visualization
            runner.message_hub.update_scalar(key, value, runner.epoch)

        # Clear losses for next epoch
        self.val_losses = []

