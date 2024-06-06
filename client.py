from __future__ import print_function
import warnings
from collections import OrderedDict

from flwr.client import NumPyClient, ClientApp
import torch
from torch.utils.data import DataLoader



# train: import section
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
from data import AnnotationTransform, VOCDetection, detection_collate, preproc, cfg
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
from models.faceboxes import FaceBoxes

# hyperparams
IMG_DIM = 1024
RGB_MEAN = (104, 117, 123)
NUM_CLASSES = 2
NUM_WORKERS = 2
BATCH_SIZE = 32
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
INITIAL_LR = 1e-3
GAMMA = 0.1
MAX_EPOCH = 300
TRAINING_DATASET = "./data/WIDER_FACE"
SAVE_FOLDER = "./weights/"



# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

priorbox = PriorBox(cfg, image_size=(IMG_DIM, IMG_DIM))
with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.to(DEVICE)

def train(net, trainloader, epochs):
    """Train the model on the training set."""
    optimizer = optim.SGD(net.parameters(), lr=INITIAL_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    criterion = MultiBoxLoss(NUM_CLASSES, 0.35, True, 0, True, 7, 0.35, False)
    num_batches = len(trainloader)

    # PriorBox moved from here
    net.train()
    train_loss = 0.0
    train_regression_loss = 0.0
    train_classification_loss = 0.0
    for batch_idx, (images, targets) in enumerate(trainloader):
        images = images.to(DEVICE)
        targets = [anno.to(DEVICE) for anno in targets]

        # forward
        out = net(images)

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, priors, targets)
        loss = cfg['loc_weight'] * loss_l + loss_c

        train_loss += loss.item()
        train_regression_loss += loss_l.item()
        train_classification_loss += loss_c.item()

        loss.backward()
        optimizer.step()

        print('BatchIter: {}/{} || L: {:.4f} C: {:.4f}'.format(batch_idx, num_batches, loss_l.item(), loss_c.item()))
    
    train_loss /= len(trainloader)
    train_regression_loss /= len(trainloader)
    train_classification_loss /= len(trainloader)

    print('Avg Loss: {:.4f} || Avg Regression Loss: {:.4f} || Avg Classification Loss: {:.4f}'.format(train_loss, train_regression_loss, train_classification_loss))


def test(net, testloader):
    # Validate the model on the test set.
    criterion = MultiBoxLoss(NUM_CLASSES, 0.35, True, 0, True, 7, 0.35, False)
    net.eval()

    test_loss = 0.0
    test_regression_loss = 0.0
    test_classification_loss = 0.0

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(testloader):
            images = images.to(DEVICE)
            targets = [anno.to(DEVICE) for anno in targets]

            outputs = net(images)

            loss_l, loss_c = criterion(outputs, priors, targets)
            loss = cfg['loc_weight'] * loss_l + loss_c

            test_loss += loss.item()
            test_regression_loss += loss_l.item()
            test_classification_loss += loss_c.item()
    
    test_loss /= len(testloader)
    test_regression_loss /= len(testloader)
    test_classification_loss /= len(testloader)

    return test_loss, test_regression_loss, test_classification_loss

def load_data(partition_id):
    """Load partition WIDER FACE data."""
    dataset = VOCDetection(TRAINING_DATASET, preproc(IMG_DIM, RGB_MEAN), AnnotationTransform())
    train_data, test_data = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=detection_collate)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=detection_collate)
    # loader = DataLoader(dataset, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=detection_collate)
    return train_loader, test_loader

# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Get partition id
"""
parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "--partition-id",
    choices=[0, 1],
    default=0,
    type=int,
    help="Partition of the dataset divided into 2 iid partitions created artificially.",
)
partition_id = parser.parse_known_args()[0].partition_id
"""
# Load model and data (simple CNN, CIFAR-10)
net = FaceBoxes('train', IMG_DIM, NUM_CLASSES).to(DEVICE)
print("Printing net...")
print(net)

# trainloader, testloader = load_data(partition_id=partition_id)
dataset = VOCDetection(TRAINING_DATASET, preproc(IMG_DIM, RGB_MEAN), AnnotationTransform())
train_data, test_data = torch.utils.data.random_split(dataset, [int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))])
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=detection_collate)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=detection_collate)

# Define Flower client
class FlowerClient(NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, train_loader, epochs=1)
        return self.get_parameters(config={}), len(train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, loss_l, loss_c = test(net, test_loader)
        return loss, len(test_loader.dataset), {"loss": loss, "regr_loss": loss_l, "class_loss": loss_c}


def client_fn(cid: str):
    """Create and return an instance of Flower `Client`."""
    return FlowerClient().to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)


# Legacy mode
if __name__ == "__main__":
    from flwr.client import start_client

    start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient().to_client(),
    )
