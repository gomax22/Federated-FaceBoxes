from helper import load_faceboxes, train, test, get_model_params, load_partition
from torch.utils.data import DataLoader, random_split
import torch
import flwr as fl
import argparse
from collections import OrderedDict
import warnings
from typing import Tuple
from data import detection_collate

warnings.filterwarnings("ignore")


class CifarClient(fl.client.NumPyClient):
    def __init__(
        self,
        trainset,
        testset,
        device: torch.device,
        mode: str,
        img_dim: int = 1024,
        num_classes: int = 2,
        validation_split: int = 0.1,
    ):
        self.device = device
        self.trainset = trainset
        self.testset = testset
        self.validation_split = validation_split
        self.img_dim = img_dim
        self.num_classes = num_classes
        self.mode = mode
        self.model = load_faceboxes(mode, img_dim, num_classes)

    def set_parameters(self, parameters):
        """Loads a alexnet or efficientnet model and replaces it parameters with the
        ones given."""

        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.set_parameters(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        trainset, valset = random_split(self.trainset, [1 - self.validation_split, self.validation_split])
        
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=detection_collate)
        val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, collate_fn=detection_collate)

        results = train(self.model, train_loader, val_loader, epochs, self.device, **config)

        parameters_prime = get_model_params(self.model)
        num_examples_train = len(trainset)

        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""
        # Update local model parameters
        self.set_parameters(parameters)

        # Get batch size from config
        batch_size: int = config["batch_size"]

        # Evaluate global model parameters on the local test data and return results
        testloader = DataLoader(self.testset, batch_size=batch_size, shuffle=False, collate_fn=detection_collate)

        loss, regr_loss, class_loss = test(self.model, testloader, self.device, **config)
        return float(loss), len(self.testset), {"loss": float(loss), "regr_loss": float(regr_loss), "class_loss": float(class_loss)}


def client_dry_run(
        data_dir: str, mode: str, img_dim: int, rgb_mean: Tuple[int, int, int], 
        num_classes: int, test_split: float, device: torch.device
    ):
    """Weak tests to check whether all client methods are working as expected."""

    model = load_faceboxes(mode, img_dim, num_classes)
    trainset, testset = load_partition(data_dir, img_dim, rgb_mean, test_split)
    trainset = trainset.select(range(10))
    testset = testset.select(range(10))
    client = CifarClient(trainset, testset, device)
    client.fit(
        parameters=get_model_params(model),
        config = {
            "learning_rate": 1e-3,
            "batch_size": 32,
            "momentum": 0.9,
            "weight_decay": 5e-4,
            "local_epochs": 1,
            "loc_weight": 2.0,
            "img_dim": 1024,
        }
    )

    client.evaluate(
        parameters=get_model_params(model), 
        config = {
            "learning_rate": 1e-3,
            "batch_size": 32,
            "momentum": 0.9,
            "weight_decay": 5e-4,
            "local_epochs": 1,
            "loc_weight": 2.0,
            "img_dim": 1024,
        }
    )

    print("Dry Run Successful")


def main() -> None:
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--dry", type=bool, default=False, help="Do a dry-run to check the client")
    parser.add_argument("--toy", action="store_true", 
                        help="Set to true to quicky run the client using only 10 datasamples. \
                            Useful for testing purposes. Default: False")
    parser.add_argument("--use_cuda", type=bool, default=False, help="Set to true to use GPU. Default: False")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], 
                        help="Mode of the model. Default: train")
    parser.add_argument("--server_address", default="0.0.0.0:8080", type=str, 
                        help="Server address. Default: 0.0.0.0:8080")
    parser.add_argument("--data_dir", type=str, default="./data/WIDER_FACE",
                        help="Data directory. Default: ./data/WIDER_FACE")
    parser.add_argument("--img_dim", type=int, default=1024, help="Image dimension. Default: 1024")
    parser.add_argument("--rgb_mean", type=tuple, default=(104, 117, 123), 
                        help="RGB mean. Default: (104, 117, 123)")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes. Default: 2")
    parser.add_argument("--validation_split", type=float, default=0.1, help="Validation split. Default: 0.1")
    parser.add_argument("--test_split", type=float, default=0.1, help="Test split. Default: 0.1")
    args = parser.parse_args()

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu"
    )

    if args.dry:
        client_dry_run(device)
    else:
        # Load a subset of CIFAR-10 to simulate the local data partition
        trainset, testset = load_partition(args.data_dir, args.img_dim, args.rgb_mean, args.test_split, args.toy)

        if args.toy:
            trainset = trainset.select(range(10))
            testset = testset.select(range(10))
        # Start Flower client
        client = CifarClient(
            trainset, 
            testset, 
            device, 
            args.mode,
            args.img_dim,
            args.num_classes,
            args.validation_split
        ).to_client()

        fl.client.start_client(server_address=args.server_address, client=client)


if __name__ == "__main__":
    main()