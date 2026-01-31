# Standard library imports
import os
import logging
from datetime import datetime

# Third-party imports
import hydra
from omegaconf import OmegaConf

# Flower imports
from flwr.client import ClientApp
from flwr.server import ServerApp
from flwr.simulation import run_simulation

# Local imports
from blossom.config import Config
from blossom.logger import save_aggregated_results, PinkColoredFormatter
from blossom.utils import print_banner, print_config, parse_dict_with_tuple_keys


# Initialize Hydra config store
config_store = hydra.core.config_store.ConfigStore.instance()
config_store.store(name="base_config", node=Config)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: Config) -> None:
    """Main function to run federated learning experiments."""
    formatter = PinkColoredFormatter()
    
    for logger_name in [None, 'flwr', 'multifl', 'hydra', 'asyncio']:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        
        logger.handlers.clear()
        
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    print_banner()

    # Load and print configuration
    OmegaConf.resolve(cfg)
    cfg_yaml = OmegaConf.to_yaml(cfg)
    print_config(cfg)

    # Parse modality dictionary
    client_config_dict = parse_dict_with_tuple_keys(cfg.experiment.clients)
    client_config = "-".join([str(v) for k, v in client_config_dict.items()])
    num_clients = sum(client_config_dict.values())

    # Create common directories and paths
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = os.path.join(
        cfg.dataset.name,
        cfg.partitioner.name,
        client_config,
        cfg.aggregation.name,
    )
    results_dir = os.path.join(cfg.logging.results_dir, base_dir, current_date)

    # Create the results directory
    os.makedirs(results_dir, exist_ok=True)

    # Set up logging
    log_file = os.path.join(results_dir, "run.log")
    logging_level = logging.DEBUG if cfg.experiment.verbose_logging else logging.INFO

    logging.basicConfig(
        filename=log_file,
        level=logging_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    logging.root = logging.getLogger("flwr")
    logging.getLogger().setLevel(logging_level)
    logging.info(f"Logging to: {log_file}")

    # Save config to output directories
    with open(os.path.join(results_dir, "config.yaml"), "w") as outfile:
        outfile.write(cfg_yaml)

    # Store run-specific results directories
    run_results_dirs = []

    # Start the federated learning process
    for run_idx in range(cfg.experiment.num_runs):
        # Create run-specific results directory
        run_dir = os.path.join(results_dir, f"run_{run_idx+1}")
        run_results_dirs.append(run_dir)
        os.makedirs(run_dir, exist_ok=True)

        # Create server and client apps
        server_app = ServerApp(server_fn=hydra.utils.call(cfg.server_fn, results_dir=run_dir))
        client_app = ClientApp(client_fn=hydra.utils.call(cfg.client_fn))

        logging.info(f"Starting run {run_idx + 1}/{cfg.experiment.num_runs}")

        run_simulation(
            server_app=server_app,
            client_app=client_app,
            num_supernodes=num_clients,
            backend_config={
                "client_resources": {
                    "num_cpus": cfg.experiment.num_cpus_per_client,
                    "num_gpus": cfg.experiment.num_gpus_per_client,
                },
                "init_args": {
                    "logging_level": logging_level,
                    "log_to_driver": False,
                },
            },
            verbose_logging=cfg.experiment.verbose_logging,
        )

    # Save aggregated results across runs
    save_aggregated_results(
        run_results_dirs,
        results_dir,
        cfg.dataset.name.upper(),
        cfg.partitioner.name.upper(),
        client_config,
        cfg.aggregation.name.upper())
    logging.info("All runs completed and aggregated results saved")


if __name__ == "__main__":
    main()