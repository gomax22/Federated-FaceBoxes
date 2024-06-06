from typing import List, Tuple

from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Metrics


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    losses_l = [num_examples * m["regr_loss"] for num_examples, m in metrics]
    losses_c = [num_examples * m["class_loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"loss": sum(losses) / sum(examples),
            "regr_loss": sum(losses_l) / sum(examples),
            "class_loss": sum(losses_c) / sum(examples)}


# Define strategy
strategy = FedAvg(evaluate_metrics_aggregation_fn=weighted_average)


# Define config
config = ServerConfig(num_rounds=2)


# Flower ServerApp
app = ServerApp(
    config=config,
    strategy=strategy,
)


# Legacy mode
if __name__ == "__main__":
    from flwr.server import start_server

    start_server(
        server_address="0.0.0.0:8080",
        config=config,
        strategy=strategy,
    )
