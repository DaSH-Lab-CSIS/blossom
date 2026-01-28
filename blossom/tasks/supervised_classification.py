# Standard library imports
from typing import Any, Dict, List, Tuple, Optional

# Torch imports
import torch
from torch import nn

# Local imports
from blossom.task import Task


class SupervisedClassificationTask(Task):
    """Supervised classification task implementation."""

    def __init__(
        self,
        criterion: nn.Module,
        input_modalities: List[str],
        output_key: str,
        optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = {"lr": 0.001},
        scheduler_class: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        scheduler_kwargs: Optional[Dict[str, Any]] = {},
        gradient_accumulation_steps: int = 1,
        use_mixed_precision: bool = False,
        gradient_clip_threshold: Optional[float] = None,
    ) -> None:
        """Initialize the classification task with a loss criterion."""
        super().__init__(
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_class=scheduler_class,
            scheduler_kwargs=scheduler_kwargs,
            gradient_accumulation_steps=gradient_accumulation_steps,
            use_mixed_precision=use_mixed_precision,
            gradient_clip_threshold=gradient_clip_threshold,
        )
        self.criterion = criterion
        self.input_modalities = input_modalities
        self.output_key = output_key

    def prepare_batch(
        self,
        batch: Dict[str, Any],
        modalities: Tuple[str, ...],
        all_modalities: List[str],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Prepare batch with modality masking for missing modalities."""
        return batch

    def compute_loss(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """Compute classification loss."""
        model_input = {mod: batch[mod] for mod in self.input_modalities}
        outputs = model(model_input)
        loss = self.criterion(outputs, batch[self.output_key])
        return loss

    def compute_batch_metrics(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> Dict[str, float]:
        """Compute accuracy metrics."""
        model_input = {mod: batch[mod] for mod in self.input_modalities}
        outputs = model(model_input)
        loss = self.criterion(outputs, batch[self.output_key])
        _, predicted = torch.max(outputs.data, 1)
        labels = batch[self.output_key]
        correct = (predicted == labels).sum().item()
        metrics = {"correct": correct}
        return loss, metrics

    def compute_aggregated_metrics(
        self,
        accumulated_metrics: Dict[str, float],
        total_samples: int,
        total_loss: float,
    ) -> Dict[str, float]:
        """Aggregate metrics over all batches."""
        accuracy = (
            accumulated_metrics["correct"] / total_samples if total_samples > 0 else 0.0
        )
        avg_loss = total_loss / total_samples if total_samples > 0 else float("nan")
        return {
            "val_accuracy": accuracy * 100.0,  # Return accuracy as a percentage
            "val_loss": avg_loss,
        }


class SupervisedClassificationTaskWithF1(Task):
    """Supervised classification task with F1 score metrics."""

    def __init__(
        self,
        criterion: nn.Module,
        input_modalities: List[str],
        output_key: str,
        num_classes: int,
        optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = {"lr": 0.001},
        scheduler_class: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        scheduler_kwargs: Optional[Dict[str, Any]] = {},
        gradient_accumulation_steps: int = 1,
        use_mixed_precision: bool = False,
        gradient_clip_threshold: Optional[float] = None,
    ) -> None:
        """Initialize the classification task with F1 score metrics."""
        super().__init__(
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_class=scheduler_class,
            scheduler_kwargs=scheduler_kwargs,
            gradient_accumulation_steps=gradient_accumulation_steps,
            use_mixed_precision=use_mixed_precision,
            gradient_clip_threshold=gradient_clip_threshold,
        )
        self.criterion = criterion
        self.input_modalities = input_modalities
        self.output_key = output_key
        self.num_classes = num_classes

    def prepare_batch(
        self,
        batch: Dict[str, Any],
        modalities: Tuple[str, ...],
        all_modalities: List[str],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Prepare batch with modality masking for missing modalities."""
        return batch

    def compute_loss(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """Compute classification loss."""
        model_input = {mod: batch[mod] for mod in self.input_modalities}
        outputs = model(model_input)
        loss = self.criterion(outputs, batch[self.output_key])
        return loss

    def compute_batch_metrics(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> Dict[str, float]:
        """Compute per-batch confusion matrix components for F1 calculation."""
        model_input = {mod: batch[mod] for mod in self.input_modalities}
        outputs = model(model_input)
        loss = self.criterion(outputs, batch[self.output_key])
        _, predicted = torch.max(outputs.data, 1)
        labels = batch[self.output_key]

        # Initialize per-class confusion matrix components
        metrics = {}
        for c in range(self.num_classes):
            # True positives: predicted c and label is c
            tp = ((predicted == c) & (labels == c)).sum().item()
            # False positives: predicted c but label is not c
            fp = ((predicted == c) & (labels != c)).sum().item()
            # False negatives: predicted not c but label is c
            fn = ((predicted != c) & (labels == c)).sum().item()

            metrics[f"tp_class_{c}"] = tp
            metrics[f"fp_class_{c}"] = fp
            metrics[f"fn_class_{c}"] = fn

        # Also track overall correct predictions for accuracy
        metrics["correct"] = (predicted == labels).sum().item()

        return loss, metrics

    def compute_aggregated_metrics(
        self,
        accumulated_metrics: Dict[str, float],
        total_samples: int,
        total_loss: float,
    ) -> Dict[str, float]:
        """Aggregate metrics and compute F1 scores."""
        avg_loss = total_loss / total_samples if total_samples > 0 else float("nan")

        # Compute per-class precision, recall, and F1
        f1_scores = []

        for c in range(self.num_classes):
            tp = accumulated_metrics.get(f"tp_class_{c}", 0)
            fp = accumulated_metrics.get(f"fp_class_{c}", 0)
            fn = accumulated_metrics.get(f"fn_class_{c}", 0)

            # Precision = TP / (TP + FP)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            # Recall = TP / (TP + FN)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            # F1 = 2 * (Precision * Recall) / (Precision + Recall)
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            f1_scores.append(f1)

        # Macro F1: average of per-class F1 scores
        macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

        return {"val_f1": macro_f1 * 100.0, "val_loss": avg_loss}