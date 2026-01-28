# Standard library imports
import os
from pathlib import Path
import shutil
from typing import Dict, List, Optional, Tuple

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.table import Table
from rich.style import Style
from rich.box import ROUNDED

BANNER_PATH = Path(__file__).parent / ".." / "assets" / "banner.ansi"

def print_banner():
    if not BANNER_PATH.exists():
        return

    width = shutil.get_terminal_size((80, 20)).columns
    
    with BANNER_PATH.open() as f:
        lines = [line.rstrip("\n") for line in f]
    
    # Find the max line width (accounting for ANSI codes)
    import re
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    max_width = max(len(ansi_escape.sub('', line)) for line in lines) if lines else 80
    
    left_padding = (width - max_width) // 2
    border = "─" * width
    print(f"\033[38;2;252;165;159m{border}\033[0m")
    
    for line in lines:
        print(' ' * left_padding + line)
    
    print(f"\033[38;2;252;165;159m{border}\033[0m")



def print_config(cfg: DictConfig):
    """Print the Hydra configuration in a styled table."""
    console = Console()
    table = Table(
        box=ROUNDED,
        expand=True,
        border_style="#fca59f",  
        show_lines=False,
    )

    table.title_style = Style(color="#fca59f", bold=True)
    table.row_styles = [Style(color="#fcd2cf"), Style(color="#fcd2cf")]

    # Collect all keys
    keys = set()
    for key, value in cfg.items():
        keys.add(key)
        if isinstance(value, dict):
            for nested_key in value.keys():
                keys.add(nested_key)

    # Sort keys for consistent column ordering
    keys = sorted(keys)

    # Add columns for each key
    for key in keys:
        table.add_column(key, style="#fca59f", header_style=Style(color="#fca59f", bold=True))

    # Populate values in respective columns
    row = {}
    for key, value in cfg.items():
        if isinstance(value, dict): # What we want here is to recursively resolve into tables
            for nested_key, nested_value in value.items():
                row[nested_key] = OmegaConf.to_yaml(nested_value)
        else:
            row[key] = OmegaConf.to_yaml(value)

    # Add a row with values in respective columns
    table.add_row(*[row.get(key, "") for key in keys])

    console.print(table)


def parse_dict_with_tuple_keys(src_dict: Dict[str, int]) -> Dict[Tuple[str, ...], int]:
    """Parse a dictionary with string keys representing tuples into actual tuple keys."""
    result = {}
    for key, value in src_dict.items():
        # Split by underscore and strip whitespace, then convert to tuple
        modalities = tuple(m.strip() for m in key.split('_'))
        result[modalities] = value
    return result


def create_metric_plot(
    lines_data: List[Dict],
    title: str,
    ylabel: str,
    xlabel: str = "SERVER ROUND",
    filename: Optional[str] = None,
    save_dir: Optional[str] = None,
    show_legend_once_per_group: bool = False,
    annotate_lines: bool = False,
    figsize: tuple = (15, 10),
) -> None:
    """Generic plotting function for all metric types."""
    if not lines_data:
        return
    
    plt.figure(figsize=figsize)
    
    # Assign colors to groups
    unique_groups = sorted(set(line["group"] for line in lines_data))
    colors = [plt.cm.viridis(i / len(unique_groups)) for i in range(len(unique_groups))]
    group_to_color = {group: colors[i] for i, group in enumerate(unique_groups)}
    
    # Track which groups have been labeled
    labeled_groups = set()
    min_vals, max_vals = [], []
    
    # Plot each line
    for line in lines_data:
        rounds = np.array(line["rounds"], dtype=float)
        values = np.array(line["values"], dtype=float)
        
        if rounds.size == 0 or values.size == 0:
            continue
        
        mask = np.isfinite(values)
        if not mask.any():
            continue
        
        color = group_to_color[line["group"]]
        
        # Determine label
        if show_legend_once_per_group:
            label = line["label"] if line["group"] not in labeled_groups else None
            labeled_groups.add(line["group"])
        else:
            label = line["label"]
        
        plt.plot(
            rounds[mask],
            values[mask],
            marker="o",
            linewidth=2,
            markersize=8,
            color=color,
            label=label,
        )
        
        min_vals.append(np.nanmin(values))
        max_vals.append(np.nanmax(values))
        
        # Add annotations if requested
        if annotate_lines:
            add_line_annotations(rounds, values, mask, color)
    
    # Set y-axis limits with better scaling
    if min_vals:
        min_metric = float(min(min_vals))
        max_metric = float(max(max_vals))
        
        if max_metric < 20:  # Likely a loss metric
            # Add 10% padding for better visibility
            range_val = max_metric - min_metric
            padding = max(range_val * 0.1, 0.1)
            min_lim = max(min_metric - padding, 0)
            max_lim = max_metric + padding
        else:  # Likely accuracy or percentage metric
            min_lim = 0
            max_lim = max(int(max_metric // 10) + 1, 10) * 10
        
        plt.ylim([min_lim, max_lim])
    
    plt.legend(loc="best")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save if filename and directory provided
    if filename and save_dir:
        # Ensure directory exists before saving
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300)
    
    plt.close()


def add_line_annotations(
    rounds: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
    color: tuple,
    fontsize: int = 7,
) -> None:
    """Add min/max/final annotations for a single line."""
    if not mask.any():
        return

    # Calculate offset relative to data range (5% of range)
    min_val = float(np.nanmin(values))
    max_val = float(np.nanmax(values))
    data_range = max_val - min_val
    offset = max(data_range * 0.05, 0.01)  # At least 0.01 offset

    # Min annotation
    min_metric = min_val
    min_idx = int(np.nanargmin(values))
    min_round = float(rounds[min_idx])
    plt.annotate(
        f"Min: {min_metric:.2f}",
        xy=(min_round, min_metric),
        xytext=(min_round, min_metric + offset),
        arrowprops=dict(facecolor=color, shrink=0.05, width=1, headwidth=8),
        fontsize=fontsize,
        ha="center",
        bbox=dict(
            boxstyle="round,pad=0.2",
            facecolor="white",
            edgecolor=color,
            alpha=0.8
        )
    )

    # Max annotation
    max_metric = max_val
    max_idx = int(np.nanargmax(values))
    max_round = float(rounds[max_idx])
    plt.annotate(
        f"Max: {max_metric:.2f}",
        xy=(max_round, max_metric),
        xytext=(max_round, max_metric + offset),
        arrowprops=dict(facecolor=color, shrink=0.05, width=1, headwidth=8),
        fontsize=fontsize,
        ha="center",
        bbox=dict(
            boxstyle="round,pad=0.2",
            facecolor="white",
            edgecolor=color,
            alpha=0.8
        )
    )

    # Final annotation
    final_metric = float(values[mask][-1]) if mask.sum() > 0 else float(values[-1])
    final_round = float(rounds[mask][-1]) if mask.sum() > 0 else float(rounds[-1])
    plt.annotate(
        f"Final: {final_metric:.2f}",
        xy=(final_round, final_metric),
        xytext=(final_round, final_metric - offset),
        arrowprops=dict(facecolor=color, shrink=0.05, width=1, headwidth=8),
        fontsize=fontsize,
        ha="center",
        bbox=dict(
            boxstyle="round,pad=0.2",
            facecolor="white",
            edgecolor=color,
            alpha=0.8
        )
    )


def extract_modality_metrics_from_client_data(
    client_dataframes: List,
) -> Dict:
    """Extract and aggregate modality-wise metrics from client dataframes."""
    modality_metrics = {}
    
    for df in client_dataframes:
        # Extract modality for each row
        df = df.copy()
        df['modality'] = df['client'].str.extract(r'\((.*?)\)')[0]
        
        # Group by modality and round
        for modality in df['modality'].unique():
            modality_df = df[df['modality'] == modality]
            
            if modality not in modality_metrics:
                modality_metrics[modality] = {}
            
            # Get all metric columns (exclude client, round, modality)
            metric_cols = [col for col in df.columns if col not in ['client', 'round', 'modality']]
            
            for metric in metric_cols:
                if metric not in modality_metrics[modality]:
                    modality_metrics[modality][metric] = {}
                
                # Average across clients in this modality per round
                grouped = modality_df.groupby('round')[metric].mean()
                
                for round_num, value in grouped.items():
                    if round_num not in modality_metrics[modality][metric]:
                        modality_metrics[modality][metric][round_num] = []
                    modality_metrics[modality][metric][round_num].append(value)
    
    return modality_metrics


def prepare_lines_data_from_modality_metrics(
    modality_metrics: Dict,
    metric: str,
) -> List[Dict]:
    """Convert modality metrics to lines_data format for plotting."""
    lines_data = []
    
    for modality in sorted(modality_metrics.keys()):
        if metric not in modality_metrics[modality]:
            continue
        
        # Average across runs for each round
        rounds_data = modality_metrics[modality][metric]
        rounds = sorted(rounds_data.keys())
        avg_values = [np.mean(rounds_data[r]) for r in rounds]
        
        lines_data.append({
            "rounds": rounds,
            "values": avg_values,
            "label": modality.upper(),
            "group": modality.upper()
        })
    
    return lines_data