import re
import os
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = "../results"

def extract_data(dataset_name: str, data_split: str, modality_config: str, aggregation_method: str):
    """Extract metrics data from CSV file for a specific configuration."""
    start_path = f"{RESULTS_DIR}/{dataset_name}/{data_split}/{modality_config}/{aggregation_method}"
    exp = r"aggregated_metrics\.csv$"
    pattern = re.compile(exp)

    for root, dirs, files in os.walk(start_path):
        for filename in files:
            if pattern.search(filename):
                full_path = os.path.join(root, filename)
                df = pd.read_csv(full_path)
                
                rounds = df.iloc[:, 0].tolist()
                metric_values = df.iloc[:, 1].tolist()
                val_loss = df.iloc[:, 2].tolist()
                metric_name = df.columns[1]  # Get the name of the metric column
                
                return rounds, metric_values, val_loss, metric_name
    
    return None, None, None, None


def plot_comparison(dataset_name: str, data_split: str, modality_config: str, metric: str = 'metric'):
    """Plot comparison of all three aggregation methods."""
    aggregation_methods = ['full-model', 'private-head', 'private-head-fusion']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    markers = ['o', '^', '*']  
    labels = ['Full Model', 'Private Head', 'Private Head Fusion']

    plot_ylabel = {
        "val_f1" : "Validation F1 Score",
        "val_accuracy" : "Validation Accuracy"
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metric_name = None
    for i, method in enumerate(aggregation_methods):
        rounds, metric_values, val_loss, metric_name = extract_data(dataset_name, data_split, modality_config, method)
        
        if rounds is None:
            print(f"Warning: No data found for {method}")
            continue
        
        if metric == 'metric':
            ax.plot(rounds, metric_values, label=labels[i], color=colors[i], linewidth=1, marker=markers[i], markersize=4)
        else:
            ax.plot(rounds, val_loss, label=labels[i], color=colors[i], linewidth=1, marker=markers[i], markersize=4)
    
    ax.set_xlabel('Round', fontsize=12)
    if metric == 'metric':
        ylabel = plot_ylabel.get(str(metric_name), 'Validation Metric')
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'{dataset_name} {data_split.upper()} {modality_config} Comparison', fontsize=14)
    else:
        ax.set_ylabel('Validation Loss', fontsize=12)
        ax.set_title(f'{dataset_name} {data_split.upper()} {modality_config} Validation Loss Comparison', fontsize=14)
    
    ax.legend(loc='lower right', fontsize=10)
    plt.ylim(0, 100)
    plt.tight_layout()
    
    save_dir = os.path.join("..", "comparisons", dataset_name, data_split)
    os.makedirs(save_dir, exist_ok=True)
    output_file_name = os.path.join(save_dir, f"{modality_config}_comparison.png")
    plt.savefig(output_file_name, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_file_name}")
    plt.close()


def main():
    dataset_names = ["AV-MNIST", "PTB-XL", "IEMOCAP", "MELD", "UCI-HAR", "KU-HAR"]
    data_splits = ["iid", "niid"]
    modality_configs = ["0-0-10", "3-3-4", "5-5-0"]
    
    for dataset_name in dataset_names:
        for data_split in data_splits:
            for modality_config in modality_configs:
                plot_comparison(dataset_name, data_split, modality_config, metric='metric')


if __name__ == "__main__":
    main()