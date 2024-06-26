from task import load_faceboxes, train, test, DEVICE, get_weights, set_weights, load_partition, save_faceboxes
import flwr as fl
import argparse
import warnings
import torch
import os
from pathlib import Path

warnings.filterwarnings("ignore")

# Parse command line argument `partition`
parser = argparse.ArgumentParser(description="Flower-based Client implementation of Federated Learning \
                                 for Face Detection using FaceBoxes on WIDER FACE dataset")
parser.add_argument("--partition_id", type=int, required=True, help="ID of the dataset partition to be loaded for training")
parser.add_argument("--use_cuda", type=bool, default=False, help="Set to true to use GPU. Default: False")
parser.add_argument("--server_address", default="0.0.0.0:8080", type=str, 
                    help="Server address. Default: 0.0.0.0:8080")
parser.add_argument("--data_dir", type=str, default="./data/WIDER_FACE",
                    help="Data directory. Default: ./data/WIDER_FACE")
parser.add_argument("--img_dim", type=int, default=1024, help="Image dimension. Default: 1024")
parser.add_argument("--num_classes", type=int, default=2, help="Number of classes. Default: 2")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size. Default: 32")
parser.add_argument("--validation_split", type=float, default=0.1, help="Validation split. Default: 0.1")
parser.add_argument("--test_split", type=float, default=0.1, help="Test split. Default: 0.1")
parser.add_argument("--weights_dir", type=str, default="./weights", help="Directory to save model weights. Default: ./weights")
args = parser.parse_args()

# Create weights directory
if not os.path.exists(args.weights_dir):
    Path(args.weights_dir).mkdir(parents=True, exist_ok=True)


# load model and data
model = load_faceboxes(img_dim=args.img_dim, num_classes=args.num_classes).to(DEVICE)
trainloader, testloader = load_partition(data_dir=args.data_dir,
                                          partition_id=args.partition_id, 
                                          img_dim=args.img_dim, 
                                          test_split=args.test_split, 
                                          batch_size=args.batch_size)

class FlowerClient(fl.client.NumPyClient):
    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        set_weights(model, parameters)
    
        # Get hyperparameters for this round
        epochs: int = config["local_epochs"]
        results = train(model, trainloader, testloader, epochs, DEVICE, **config)

        # return model weights, the number of examples used for training, and the results
        return get_weights(model), len(trainloader.dataset), results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""
        # Update local model parameters
        set_weights(model, parameters)

        # evaluate the model with updated weights
        loss, regr_loss, class_loss = test(model, testloader, DEVICE, **config)
        print(f"Client server round: {config['current_round']}")
        
        # Save the model
        out_path = os.path.join(args.weights_dir, f"faceboxes_r{config['current_round']}.pth")
        torch.save(model.state_dict(),  out_path)
        print(f"Model saved to: {out_path}")

        # return the loss, the number of examples used for evaluation, and the results
        return float(loss), len(testloader.dataset), {"loss": float(loss), "regr_loss": float(regr_loss), "class_loss": float(class_loss)}

def client_fn(cid: str):
    """Create and return an instance of Flower `Client`."""
    return FlowerClient().to_client()


# Flower ClientApp
app = fl.client.ClientApp(
    client_fn=client_fn,
)


# Legacy mode
if __name__ == "__main__":
    from flwr.client import start_client

    start_client(
        server_address=args.server_address,
        client=FlowerClient().to_client(),
    )