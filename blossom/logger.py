# Standard library imports
import os
import logging
from typing import Dict, List

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from blossom.utils import (
    create_metric_plot,
    prepare_lines_data_from_modality_metrics,
    extract_modality_metrics_from_client_data
)


class PinkColoredFormatter(logging.Formatter):
    """Custom formatter for pink-themed logging with gradient intensity by level."""
    
    # Custom color scheme
    TEXT_COLOR = '\033[38;2;252;210;207m'     # Text: #fcd2cf
    INFO_COLOR = '\033[38;2;252;165;159m'     # INFO: #fca59f
    WARNING_COLOR = '\033[38;2;255;105;180m'  # WARNING: #ff69b4
    ERROR_COLOR = '\033[38;2;139;0;139m'      # ERROR: #8b008b
    RESET = '\033[0m'
    
    LEVEL_COLORS = {
        'DEBUG': INFO_COLOR,
        'INFO': INFO_COLOR,
        'WARNING': WARNING_COLOR,
        'ERROR': ERROR_COLOR,
        'CRITICAL': ERROR_COLOR,
    }
    
    def format(self, record):
        # Get level-specific color
        level_color = self.LEVEL_COLORS.get(record.levelname, self.INFO_COLOR)
        
        # Format level name in level-specific color
        levelname = f"{level_color}{record.levelname}{self.RESET}"
        
        # Format message in text color for readability
        message = f"{self.TEXT_COLOR}{record.getMessage()}{self.RESET}"
        
        # Combine into log line
        return f"{levelname} : {message}"


class LoggingManager:
    """Manages logging and result aggregation for federated learning experiments."""

    def __init__(self, results_dir: str) -> None:
        """Initialize the logging manager."""
        self.results_dir = results_dir
        self.client_metrics_history: Dict[str, Dict[str, List]] = {}
        self.aggregated_metrics_history: Dict[str, List] = {"round": []}

    def log_client_metrics(
        self,
        client_name: str,
        server_round: int,
        metrics: Dict[str, float]
    )-> None:
        """Add metrics for a specific client and server round."""
        # Initialize history for new client
        if client_name not in self.client_metrics_history:
            self.client_metrics_history[client_name] = {"round": []}

        # Append round
        self.client_metrics_history[client_name]["round"].append(server_round)

        # Append metrics
        for key, value in metrics.items():
            if key in ("client_name", "round"):
                continue
            if key not in self.client_metrics_history[client_name]:
                self.client_metrics_history[client_name][key] = []
            self.client_metrics_history[client_name][key].append(value)
    
    def log_aggregated_metrics(self, server_round: int, metrics: Dict[str, float]) -> None:
        """Add aggregated metrics for a specific server round."""
        # Append round
        self.aggregated_metrics_history["round"].append(server_round)

        # Append metrics
        for key, value in metrics.items():
            if key == "round":
                continue
            if key not in self.aggregated_metrics_history:
                self.aggregated_metrics_history[key] = []
            self.aggregated_metrics_history[key].append(value)

    def save_results(self) -> None:
        """Generate plots and save metrics to CSV files."""
        logging.info(f"Final round: Saving metrics and plots to {self.results_dir}")

        # Plot client-wise metrics
        for metric in self._get_metrics_from_history(self.client_metrics_history):
            self._plot_client_wise_metrics(metric)

        # Plot aggregated metrics
        for metric in self._get_metrics_from_history(self.aggregated_metrics_history):
            self._plot_aggregated_metrics(metric)
        
        # Plot modality-wise metrics
        self._plot_modality_wise_metrics()

        # Save CSVs
        self._export_metrics_to_csv()

    def _get_metrics_from_history(self, history: Dict) -> List[str]:
        """Extract metric names from history dictionary."""
        if isinstance(history, dict) and "round" in history:
            return [m for m in history.keys() if m != "round"]
        else:
            all_metrics = set()
            for client_history in history.values():
                all_metrics.update(client_history.keys())
            all_metrics.discard("round")
            return list(all_metrics)

    def _plot_client_wise_metrics(self, metric: str) -> None:
        """Plot client-wise metrics grouped by modality."""
        lines_data = []
        # Sort clients by their modality
        sorted_clients = sorted(
            self.client_metrics_history.items(),
            key=lambda item: item[0].split("(")[-1].rstrip(")")
        )
        
        for client_name, history in sorted_clients:
            rounds = history.get("round", [])
            values = history.get(metric, [])
            if rounds and values:
                modality = client_name.split("(")[-1].rstrip(")")
                lines_data.append({
                    "rounds": rounds,
                    "values": values,
                    "label": modality,
                    "group": modality
                })
        
        create_metric_plot(
            lines_data=lines_data,
            title=f"CLIENT-WISE {metric.upper()} PER SERVER ROUND",
            ylabel=metric.upper(),
            filename=f"client_wise_{metric}.png",
            save_dir=self.results_dir,
            show_legend_once_per_group=True,
            annotate_lines=False
        )

    def _plot_modality_wise_metrics(self) -> None:
        """Plot modality-wise averaged metrics per run."""
        # Extract and group data by modality
        modality_metrics = {}
        for client_name, history in self.client_metrics_history.items():
            modality = client_name.split("(")[-1].rstrip(")")
            
            if modality not in modality_metrics:
                modality_metrics[modality] = {}
            
            for metric, values in history.items():
                if metric == "round":
                    continue
                    
                if metric not in modality_metrics[modality]:
                    modality_metrics[modality][metric] = {}
                
                rounds = history["round"]
                for i, round_num in enumerate(rounds):
                    if round_num not in modality_metrics[modality][metric]:
                        modality_metrics[modality][metric][round_num] = []
                    modality_metrics[modality][metric][round_num].append(values[i])
        
        # Plot each metric
        all_metrics = set()
        for modality_data in modality_metrics.values():
            all_metrics.update(modality_data.keys())
        
        for metric in all_metrics:
            lines_data = []
            for modality in sorted(modality_metrics.keys()):
                if metric not in modality_metrics[modality]:
                    continue
                
                rounds_data = modality_metrics[modality][metric]
                rounds = sorted(rounds_data.keys())
                avg_values = [np.mean(rounds_data[r]) for r in rounds]
                
                lines_data.append({
                    "rounds": rounds,
                    "values": avg_values,
                    "label": modality,
                    "group": modality
                })
            
            create_metric_plot(
                lines_data=lines_data,
                title=f"MODALITY-WISE {metric.upper()} PER SERVER ROUND",
                ylabel=metric.upper(),
                filename=f"modality_wise_{metric}.png",
                save_dir=self.results_dir,
                show_legend_once_per_group=False,
                annotate_lines=True
            )

    def _plot_aggregated_metrics(self, metric: str) -> None:
        """Plot aggregated metrics."""
        rounds = self.aggregated_metrics_history.get("round", [])
        values = self.aggregated_metrics_history.get(metric, [])
        
        if not rounds or not values:
            return
        
        lines_data = [{
            "rounds": rounds,
            "values": values,
            "label": "AGGREGATED",
            "group": "AGGREGATED"
        }]
        
        create_metric_plot(
            lines_data=lines_data,
            title=f"AGGREGATED {metric.upper()} PER SERVER ROUND",
            ylabel=f"VALIDATION {metric.upper()}",
            filename=f"aggregated_{metric}.png",
            save_dir=self.results_dir,
            show_legend_once_per_group=False,
            annotate_lines=True
        )

    def _export_metrics_to_csv(self) -> None:
        """Save client and aggregated metrics to CSV files."""
        # Save client metrics
        client_data = []
        for client_name, history in self.client_metrics_history.items():
            for i, round_num in enumerate(history["round"]):
                data_entry = {"client": client_name, "round": round_num}
                for key in history:
                    if key != "round":
                        data_entry[key] = history[key][i]
                client_data.append(data_entry)

        if client_data:
            client_df = pd.DataFrame(client_data)
            client_path = os.path.join(self.results_dir, "client_metrics.csv")
            client_df.to_csv(client_path, index=False)
            logging.info(f"Saved client metrics to {client_path}")

        # Save aggregated metrics
        aggregated_data = []
        for i, round_num in enumerate(self.aggregated_metrics_history["round"]):
            data_entry = {"round": round_num}
            for key in self.aggregated_metrics_history:
                if key != "round":
                    data_entry[key] = self.aggregated_metrics_history[key][i]
            aggregated_data.append(data_entry)

        if aggregated_data:
            aggregated_df = pd.DataFrame(aggregated_data)
            aggregated_path = os.path.join(self.results_dir, "aggregated_metrics.csv")
            aggregated_df.to_csv(aggregated_path, index=False)
            logging.info(f"Saved aggregated metrics to {aggregated_path}")


def save_aggregated_results(
    run_results_dirs: List[str],
    results_dir: str,
    dataset_name: str = "Unknown",
    partitioner_name: str = "Unknown",
    client_config: str = "Unknown",
    aggregation_name: str = "Unknown"
) -> None:
    """Generate plots and CSV aggregating results across multiple runs."""
    if not run_results_dirs:
        logging.warning("No run results directories provided for aggregation")
        return

    # Load aggregated metrics CSVs
    aggregated_dataframes = []
    for run_dir in run_results_dirs:
        csv_path = os.path.join(run_dir, "aggregated_metrics.csv")
        df = pd.read_csv(csv_path)
        aggregated_dataframes.append(df)

    # Get all metric names (excluding 'round')
    all_columns = set()
    for df in aggregated_dataframes:
        all_columns.update(df.columns)
    all_columns.discard("round")
    metric_names = sorted(all_columns)

    # Find common rounds across all runs
    all_rounds = [df["round"].values for df in aggregated_dataframes]
    common_rounds = sorted(set(all_rounds[0]).intersection(*all_rounds[1:]))

    # Build averaged metrics dictionary
    averaged_metrics = {"round": common_rounds}

    # For each metric, compute average across runs and plot
    for metric in metric_names:
        averaged_values = []
        for round_num in common_rounds:
            values_at_round = []
            for df in aggregated_dataframes:
                row = df[df["round"] == round_num]
                if not row.empty and metric in df.columns:
                    values_at_round.append(row[metric].values[0])

            if values_at_round:
                averaged_values.append(np.mean(values_at_round))
            else:
                averaged_values.append(np.nan)

        averaged_metrics[metric] = averaged_values

        # Plot aggregated metric
        lines_data = [{
            "rounds": common_rounds,
            "values": averaged_values,
            "label": "AGGREGATED",
            "group": "AGGREGATED"
        }]
        
        try:
            create_metric_plot(
                lines_data=lines_data,
                title=f"{dataset_name}: AGGREGATED {metric.upper()} ACROSS CLIENTS"
                        f" ({partitioner_name}, {client_config}, {aggregation_name})",
                ylabel=f"AVERAGED {metric.upper()}",
                xlabel="ROUND",
                filename=f"averaged_{metric}.png",
                save_dir=results_dir,
                annotate_lines=True
            )
        except Exception as e:
            logging.error(f"Error plotting averaged metric {metric}: {e}")

    # Save averaged metrics to CSV
    averaged_df = pd.DataFrame(averaged_metrics)
    csv_path = os.path.join(results_dir, "aggregated_metrics.csv")
    averaged_df.to_csv(csv_path, index=False)
    
    # Plot modality-wise metrics across runs
    plot_modality_wise_metrics_across_runs(
        run_results_dirs,
        results_dir,
        dataset_name,
        partitioner_name,
        client_config,
        aggregation_name
    )


def plot_modality_wise_metrics_across_runs(
    run_results_dirs: List[str],
    results_dir: str,
    dataset_name: str = "Unknown",
    partitioner_name: str = "Unknown",
    client_config: str = "Unknown",
    aggregation_name: str = "Unknown"
) -> None:
    """Generate modality-wise plots averaged across multiple runs."""
    # Load client metrics from all runs
    client_dataframes = []
    for run_dir in run_results_dirs:
        csv_path = os.path.join(run_dir, "client_metrics.csv")
        df = pd.read_csv(csv_path)
        client_dataframes.append(df)
    
    # Extract modality metrics
    modality_metrics = extract_modality_metrics_from_client_data(client_dataframes)
    
    # Get all metrics
    all_metrics = set()
    for modality_data in modality_metrics.values():
        all_metrics.update(modality_data.keys())
    
    # Plot each metric
    for metric in all_metrics:
        lines_data = prepare_lines_data_from_modality_metrics(modality_metrics, metric)
        try:
            create_metric_plot(
                lines_data=lines_data,
                title=f"{dataset_name}: MODALITY-WISE {metric.upper()} ACROSS CLIENTS"
                        f" ({partitioner_name}, {client_config}, {aggregation_name})",
                ylabel=f"AVERAGED {metric.upper()}",
                xlabel="ROUND",
                filename=f"modality_wise_{metric}.png",
                save_dir=results_dir,
                annotate_lines=True
            )
        except Exception as e:
            logging.error(f"Error plotting modality-wise metric {metric}: {e}")