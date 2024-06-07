from typing import Dict, Optional, Tuple
from collections import OrderedDict
import argparse
import flwr as fl
import torch
from task import load_faceboxes, get_weights, save_faceboxes
import warnings
from typing import List, Union, Optional, Callable
from flwr.common import Parameters, Scalar, Metrics, ndarrays_to_parameters
import numpy as np
import os
from pathlib import Path

warnings.filterwarnings("ignore")


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, model, weights_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.weights_dir = weights_dir

        if not os.path.exists(self.weights_dir):
            Path(self.weights_dir).mkdir(parents=True, exist_ok=True)


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_results = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Save the model
            save_faceboxes(self.model, aggregated_parameters, os.path.join(self.weights_dir, f"model_round_{server_round}.pth"))

        return aggregated_parameters, aggregated_results


    
def on_fit_config(server_round: int) -> Dict[str, str]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "learning_rate": 1e-3,
        "batch_size": 32,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "local_epochs": 1,
        "loc_weight": 2.0,
        "img_dim": 1024,
    }
    return config

def get_on_evaluate_config_fn(num_rounds: int) -> Callable[[int], Dict[str, str]]:
    """Return a function which returns a configuration with static batch size."""
    def on_evaluate_config(server_round: int) -> Dict[str, str]:
        return {
            "batch_size": 32,
            "loc_weight": 2.0,
            "img_dim": 1024,
            "current_round": server_round,
            "num_rounds": num_rounds
        }
    return on_evaluate_config



def fit_weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate metrics using weighted average."""
    train_losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    train_regr_losses = [num_examples * m["train_regression_loss"] for num_examples, m in metrics]
    train_class_losses = [num_examples * m["train_classification_loss"] for num_examples, m in metrics]
    val_losses = [num_examples * m["val_loss"] for num_examples, m in metrics]
    val_regr_losses = [num_examples * m["val_regression_loss"] for num_examples, m in metrics]
    val_class_losses = [num_examples * m["val_classification_loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"train_loss": sum(train_losses) / sum(examples),
            "train_regression_loss": sum(train_regr_losses) / sum(examples),
            "train_classification_loss": sum(train_class_losses) / sum(examples),
            "val_loss": sum(val_losses) / sum(examples),
            "val_regression_loss": sum(val_regr_losses) / sum(examples),
            "val_classification_loss": sum(val_class_losses) / sum(examples)
    }


def evaluate_weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate metrics using weighted average."""
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    losses_l = [num_examples * m["regr_loss"] for num_examples, m in metrics]
    losses_c = [num_examples * m["class_loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"loss": sum(losses) / sum(examples),
            "regr_loss": sum(losses_l) / sum(examples),
            "class_loss": sum(losses_c) / sum(examples)}

# Parse command line argument `partition`
parser = argparse.ArgumentParser(description="Flower")
parser.add_argument("--toy", action="store_true", 
                    help="Set to true to use only 10 datasamples for validation. \
                        Useful for testing purposes. Default: False")
parser.add_argument("--num_rounds", default=2, type=int, help="Number of rounds of federated learning. Default: 4")
parser.add_argument("--server_address", default="0.0.0.0:8080", type=str, help="Server address. Default: 0.0.0.0:8080")
parser.add_argument("--num_clients", default=2, type=int, help="Number of clients. Default: 2")
parser.add_argument("--resume_net", default=None, type=str, help="Path to resume network for retraining. Default: None")
parser.add_argument("--phase", type=str, default="train", choices=["train", "test"], help="Phase of the model. Default: train")
parser.add_argument("--img_dim", type=int, default=1024, help="Image dimension. Default: 1024")
parser.add_argument("--num_classes", type=int, default=2, help="Number of classes. Default: 2")    
parser.add_argument("--weights_dir", type=str, default="./weights", help="Directory to save model weights. Default: ./weights")
args = parser.parse_args()

# Load model
model = load_faceboxes(args.phase, args.img_dim, args.num_classes, args.resume_net)
ndarrays = get_weights(model)
parameters = ndarrays_to_parameters(ndarrays)

# Create strategy
# Create strategy and run server
strategy = SaveModelStrategy(
    model=model,
    weights_dir=args.weights_dir,
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=args.num_clients,
    min_evaluate_clients=args.num_clients,
    min_available_clients=args.num_clients,
    on_fit_config_fn=on_fit_config,
    on_evaluate_config_fn=get_on_evaluate_config_fn(args.num_rounds),
    fit_metrics_aggregation_fn=fit_weighted_average,
    evaluate_metrics_aggregation_fn=evaluate_weighted_average,
    initial_parameters=parameters,
)


# define config
config = fl.server.ServerConfig(num_rounds=args.num_rounds)

# Flower ServerApp
app = fl.server.ServerApp(
    config=config,
    strategy=strategy,
)


# Legacy mode
if __name__ == "__main__":
    from flwr.server import start_server

    start_server(
        server_address=args.server_address,
        config=config,
        strategy=strategy,
    )

