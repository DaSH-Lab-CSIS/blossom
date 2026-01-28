# Torch imports
import torch
import torch.nn as nn

# Local imports
from blossom.task import Task
from blossom.tasks.supervised_classification import SupervisedClassificationTaskWithF1


# ============================================================================
# Task Configuration
# ============================================================================

CRITERION = nn.CrossEntropyLoss()
INPUT_MODALITIES = ["acc", "gyro"]
OUTPUT_KEY = "label"
NUM_CLASSES = 6
OPTIMIZER_CLASS = torch.optim.SGD
OPTIMIZER_KWARGS = {"lr": 0.05, "momentum": 0.9, "weight_decay": 1e-5}
SCHEDULER_CLASS = torch.optim.lr_scheduler.CosineAnnealingLR
SCHEDULER_KWARGS = {"T_max": 60, "eta_min": 1e-6}


# ============================================================================
# Factory Functions
# ============================================================================

def get_partition_by() -> str:
    """Return the column name to partition data by."""
    return OUTPUT_KEY


def get_task() -> Task:
    """Return the supervised classification task instance."""
    return SupervisedClassificationTaskWithF1(
        criterion=CRITERION,
        input_modalities=INPUT_MODALITIES,
        output_key=OUTPUT_KEY,
        num_classes=NUM_CLASSES,
        optimizer_class=OPTIMIZER_CLASS,
        optimizer_kwargs=OPTIMIZER_KWARGS,
        scheduler_class=SCHEDULER_CLASS,
        scheduler_kwargs=SCHEDULER_KWARGS,
    )
