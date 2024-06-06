# Adapted from https://github.com/adap/flower/blob/main/examples/advanced-pytorch/utils.py

import torch
import warnings
from torch.utils.data import random_split
from models.faceboxes import FaceBoxes
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
from data import AnnotationTransform, VOCDetection, preproc, cfg

warnings.filterwarnings("ignore")

# hyperparams
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_partition(data_dir: str, img_dim: int, rgb_mean: int, test_split: float, toy: bool = False):
    """Load partition WIDER_FACE data."""
    dataset = VOCDetection(data_dir, preproc(img_dim, rgb_mean), AnnotationTransform())
    train_data, test_data = random_split(dataset, [1 - test_split, test_split])
    return train_data, test_data

def train(net, trainloader, valloader, epochs, device: torch.device, **kwargs):
    """Train the network on the training set."""
    print("Starting training...")

    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=kwargs['learning_rate'], momentum=kwargs['momentum'], weight_decay=kwargs['weight_decay'])
    criterion = MultiBoxLoss(net.num_classes, 0.35, True, 0, True, 7, 0.35, False)
    num_batches = len(trainloader)

    priorbox = PriorBox(cfg, image_size=(kwargs['img_dim'], kwargs['img_dim']))
    with torch.no_grad():
        priors = priorbox.forward()
        priors = priors.to(device)

    # PriorBox moved from here
    net.train()

    for epoch_idx in range(epochs):
        
        train_loss = 0.0
        train_regr_loss = 0.0
        train_class_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(trainloader):
            images = images.to(device)
            targets = [anno.to(device) for anno in targets]

            # forward
            out = net(images)

            # backprop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, priors, targets)
            loss = kwargs['loc_weight'] * loss_l + loss_c

            train_loss += loss.item()
            train_regr_loss += loss_l.item()
            train_class_loss += loss_c.item()

            loss.backward()
            optimizer.step()

            print('Epoch: {}/{} || BatchIter: {}/{} || L: {:.4f} C: {:.4f}'.format(epoch_idx, epochs, batch_idx, num_batches, loss_l.item(), loss_c.item()))

        train_loss /= len(trainloader)
        train_regr_loss /= len(trainloader)
        train_class_loss /= len(trainloader)

        print('Avg Loss: {:.4f} || Avg Regression Loss: {:.4f} || Avg Classification Loss: {:.4f}'.format(train_loss, train_regr_loss, train_class_loss))

    net.to("cpu")
    
    train_loss, train_regr_loss, train_class_loss = test(net, trainloader, device, **kwargs)
    val_loss, val_regr_loss, val_class_loss = test(net, valloader, device, **kwargs)

    results = {
        "train_loss": train_loss,
        "train_regression_loss": train_regr_loss,
        "train_classification_loss": train_class_loss,
        "val_loss": val_loss,
        "val_regression_loss": val_regr_loss,
        "val_classification_loss": val_class_loss
    }
    return results

    

def test(net, testloader, device: torch.device, **kwargs):
    """Validate the network on the entire test set."""
    print("Starting evaluation...")
    net.to(device) 
    # Validate the model on the test set.
    criterion = MultiBoxLoss(net.num_classes, 0.35, True, 0, True, 7, 0.35, False)
    net.eval()

    test_loss = 0.0
    test_regression_loss = 0.0
    test_classification_loss = 0.0

    
    priorbox = PriorBox(cfg, image_size=(kwargs['img_dim'], kwargs['img_dim']))
    with torch.no_grad():
        priors = priorbox.forward()
        priors = priors.to(device)

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(testloader):
            images = images.to(device)
            targets = [anno.to(device) for anno in targets]

            outputs = net(images)

            loss_l, loss_c = criterion(outputs, priors, targets)
            loss = kwargs['loc_weight'] * loss_l + loss_c

            test_loss += loss.item()
            test_regression_loss += loss_l.item()
            test_classification_loss += loss_c.item()
    
    net.to("cpu") 
    test_loss /= len(testloader)
    test_regression_loss /= len(testloader)
    test_classification_loss /= len(testloader)

    return test_loss, test_regression_loss, test_classification_loss


def get_model_params(model):
    """Returns a model's parameters."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def load_faceboxes(mode: str = 'train', img_dim: int = 1024, num_classes: int = 2, resume_net: bool = None):
    # Load model and data (simple CNN, CIFAR-10)
    net = FaceBoxes(mode, img_dim, num_classes)
    print("Printing net...")
    print(net)


    if resume_net is not None:
        print('Loading resume network...')
        state_dict = torch.load(resume_net)
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:] # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)

    return net