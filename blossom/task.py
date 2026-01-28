# Standard library imports
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Type

# Torch imports
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

# Local imports
from blossom.models import ENCODER_PREFIX, FUSION_PREFIX, HEAD_PREFIX


class Task(ABC):
    """Abstract base class for defining a learning task."""

    def __init__(
        self,
        optimizer_class: Type[Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = {"lr": 0.001},
        scheduler_class: Optional[Type[LRScheduler]] = None,
        scheduler_kwargs: Optional[Dict[str, Any]] = {},
        gradient_accumulation_steps: int = 1,
        use_mixed_precision: bool = False,
        gradient_clip_threshold: Optional[float] = None,
    ):
        """
        Initialize task with training configuration.

        Args:
            optimizer_class: Optimizer class (e.g., torch.optim.Adam)
            optimizer_kwargs: Keyword arguments for optimizer initialization
            scheduler_class: Learning rate scheduler class (e.g., torch.optim.lr_scheduler.StepLR)
            scheduler_kwargs: Keyword arguments for scheduler initialization
            gradient_accumulation_steps: Number of steps to accumulate gradients
            use_mixed_precision: Whether to use automatic mixed precision training
            gradient_clip_threshold: Maximum norm for gradient clipping (None to disable)
        """
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_class = scheduler_class
        self.scheduler_kwargs = scheduler_kwargs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_mixed_precision = use_mixed_precision
        self.gradient_clip_threshold = gradient_clip_threshold

    @abstractmethod
    def prepare_batch(
        self,
        batch: Dict[str, Any],
        modalities: Tuple[str, ...],
        all_modalities: List[str],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare batch data for model input.

        The keys in the returned dictionary should correspond to the model's expected input keys.

        Args:
            batch: Raw batch data from dataloader
            modalities: Modalities available to this client
            all_modalities: All possible modalities in the dataset
            device: Device to move data to

        Returns:
            Processed batch ready for model input
        """
        pass

    @abstractmethod
    def compute_loss(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute the loss for a given batch.

        Args:
            model: The model to evaluate
            batch: Dictionary containing prepared batch data
            device: Device to run computation on

        Returns:
            Loss tensor
        """
        pass

    @abstractmethod
    def compute_batch_metrics(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the loss and evaluation metrics for a given batch.

        Args:
            model: The model to evaluate
            batch: Dictionary containing prepared batch data
            device: Device to run computation on

        Returns:
            A tuple of (loss tensor, dictionary of metric names to values)
        """
        pass

    @abstractmethod
    def compute_aggregated_metrics(
        self,
        accumulated_metrics: Dict[str, float],
        num_samples: int,
        total_loss: float,
    ) -> Dict[str, float]:
        """
        Compute aggregated metrics over the entire dataset.

        Args:
            accumulated_metrics: Dictionary of accumulated metric values
            num_samples: Total number of samples evaluated
            total_loss: Total loss accumulated over all samples

        Returns:
            Dictionary of aggregated metric names to values
        """
        pass


def should_select_parameter(
    param_name: str,
    modalities: Tuple[str, ...]
) -> bool:
    """Determine if a parameter should be selected based on available modalities."""
    # Always train fusion and head modules
    if param_name.startswith(FUSION_PREFIX) or param_name.startswith(HEAD_PREFIX):
        return True

    # Train encoder if it matches any available modality
    for modality_name in modalities:
        if param_name.startswith(ENCODER_PREFIX + modality_name):
            return True

    return False


def mask_modalities(
    batch: Dict[str, Any],
    modalities: Tuple[str, ...],
    all_modalities: List[str],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Prepare batch with modality masking for missing modalities."""
    # Prepare batch based on available modalities
    processed_batch = {}
    for key, val in batch.items():
        if key in all_modalities:
            if key in modalities:
                # Use real data for available modalities
                processed_batch[key] = val.to(device)
            else:
                # Zero out missing modalities
                processed_batch[key] = torch.zeros_like(val).to(device)
        else:
            # Non-modality data (e.g., labels) remains unchanged
            processed_batch[key] = val.to(device)

    return processed_batch


def train(
    model: nn.Module,
    task: Task,
    modalities: Tuple[str, ...],
    all_modalities: List[str],
    trainloader: DataLoader,
    epochs: int,
    device: torch.device,
    batch_size_threshold: int = 2,
    scheduler_state: Optional[Dict] = None,
) -> Tuple[float, int, Dict[str, Any]]:
    """Train the model, freezing unused encoders for missing modalities."""
    model.to(device)

    # Freeze encoders not in the client's modality tuple
    for name, param in model.named_parameters():
        param.requires_grad = should_select_parameter(name, modalities)

    # Filter parameters for the optimizer after freezing
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = task.optimizer_class(trainable_params, **task.optimizer_kwargs)

    # Create learning rate scheduler if specified
    scheduler = None
    if task.scheduler_class is not None:
        scheduler = task.scheduler_class(optimizer, **task.scheduler_kwargs)
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
    
    # Create gradient scaler for mixed precision training
    scaler = torch.amp.GradScaler(device.type) if task.use_mixed_precision else None

    model.train()

    total_loss = 0.0
    total_samples = 0
    
    for epoch in range(epochs):
        for batch_idx, batch in enumerate(trainloader):
            batch_size = next(iter(batch.values())).size(0)
            if batch_size < batch_size_threshold:
                continue

            # Prepare batch using task-specific logic and modality masking
            prepared_batch = mask_modalities(
                task.prepare_batch(batch, modalities, all_modalities, device),
                modalities,
                all_modalities,
                device,
            )

            # Forward pass with mixed precision
            with torch.amp.autocast(device_type=device.type, enabled=task.use_mixed_precision):
                loss = task.compute_loss(model, prepared_batch, device)
            
            # Track unscaled loss for reporting (per-sample average)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Scale loss for gradient accumulation
            scaled_loss = loss / task.gradient_accumulation_steps

            # Backward pass with optional mixed precision
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            
            # Check if we should update weights
            should_step = (batch_idx + 1) % task.gradient_accumulation_steps == 0
            is_last_batch = (batch_idx + 1) == len(trainloader)
            
            if should_step or is_last_batch:
                # Gradient clipping if specified
                if task.gradient_clip_threshold is not None:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), task.gradient_clip_threshold
                    )

                # Optimizer step
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

    # Step the learning rate scheduler at the end of training
    if scheduler is not None:
        scheduler.step()

    # Prepare scheduler state to return
    updated_scheduler_state = scheduler.state_dict() if scheduler is not None else None

    return total_loss, total_samples, updated_scheduler_state


def test(
    model: nn.Module,
    task: Task,
    modalities: Tuple[str, ...],
    all_modalities: List[str],
    valloader: DataLoader,
    device: torch.device
) -> Tuple[float, float]:
    """Evaluate the model, simulating missing modalities by providing zero inputs."""
    model.to(device)
    model.eval()

    accumulated_metrics = {}
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in valloader:
            batch_size = next(iter(batch.values())).size(0)

            # Prepare batch using task-specific logic and modality masking
            prepared_batch = mask_modalities(
                task.prepare_batch(batch, modalities, all_modalities, device),
                modalities,
                all_modalities,
                device,
            )

            # Compute metrics
            loss, batch_metrics = task.compute_batch_metrics(model, prepared_batch, device)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Accumulate metrics
            for key, value in batch_metrics.items():
                if key not in accumulated_metrics:
                    accumulated_metrics[key] = 0.0
                accumulated_metrics[key] += value

    return total_loss, total_samples, accumulated_metrics


def get_parameters(
    model: torch.nn.Module,
    modalities: Tuple[str, ...]
) -> Tuple[List[np.ndarray], List[str]]:
    """Get model parameters and keys relevant to the specified modality for sending updates."""
    state_dict = model.state_dict()
    relevant_parameters = []
    relevant_keys = []

    for name, param in state_dict.items():
        if should_select_parameter(name, modalities):
            relevant_parameters.append(param.detach().cpu().numpy())
            relevant_keys.append(name)

    return relevant_parameters, relevant_keys


def set_parameters(
    model: torch.nn.Module, params_with_keys: Tuple[List[np.ndarray], List[str]]
) -> None:
    """Set model parameters using provided parameters and their keys."""
    parameters, keys = params_with_keys
    model_device = next(model.parameters()).device

    # Create updates dictionary, ensuring tensors are on the correct device
    updates = {
        key: torch.tensor(param, device=model_device)
        for key, param in zip(keys, parameters)
    }

    current_state_dict = model.state_dict()
    updated_state_dict = OrderedDict()

    # Iterate through the model's state dict to ensure order and handle missing updates
    for name, current_param in current_state_dict.items():
        if name in updates:
            updated_state_dict[name] = updates[name].to(current_param.dtype)
        else:
            # Keep the existing parameter if no update was provided for it
            updated_state_dict[name] = current_param

    model.load_state_dict(updated_state_dict, strict=True)