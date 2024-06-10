
# Federated FaceBoxes: a Flower-based implementation of Federated Learning for Face Detection using FaceBoxes on WIDER FACE dataset
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Taking inspiration from a [PyTorch](https://pytorch.org/) implementation of [FaceBoxes: A CPU Real-time Face Detector with High Accuracy](https://arxiv.org/abs/1708.05234) and [Flower examples](https://github.com/adap/flower/tree/main/examples).

## Installation
Clone this repository.

```Shell
git clone --depth 1 https://github.com/gomax22/Federated-FaceBoxes.git
cd Federated-FaceBoxes
```

_Optional_: Compile the nms (for GPU users)
```Shell
./make.sh
```

## Dataset Preparation

### Download
1. Download [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) dataset (train split)
2. Download [converted annotations](https://drive.google.com/open?id=1-s4QCu_v76yNwR-yXMfGqMGgHQ30WxV2) directly from original repository [FaceBoxes.Pytorch](https://github.com/zisianw/FaceBoxes.PyTorch/edit/master/) 
3. Download the images of [AFW](https://drive.google.com/open?id=1Kl2Cjy8IwrkYDwMbe_9DVuAwTHJ8fjev), [PASCAL Face](https://drive.google.com/open?id=1p7dDQgYh2RBPUZSlOQVU4PgaSKlq64ik) and [FDDB](https://drive.google.com/open?id=17t4WULUDgZgiSy5kpCax4aooyPaz3GQH) for testing purposes

N.B. It's mandatory to download at least one of them in order to test the application. 

### Extraction
```Shell 
unzip path/to/WIDER_train.zip -d data/
unzip path/to/pascal_images.zip -d data/PASCAL          # if downloaded
unzip path/to/afw_images.zip -d data/AFW                # if downloaded
unzip path/to/fddb_images.zip -d data/FDDB              # if downloaded
tar xvf path/to/annotations.tar.gz -C data/WIDER_FACE/  

mv data/WIDER_train/images data/WIDER_FACE              
mv data/PASCAL/pascal_images data/PASCAL/images         # if downloaded
mv data/AFW/afw_images data/AFW/images                  # if downloaded
mv data/FDDB/fddb_images data/FDDB/images               # if downloaded
```

## Install dependencies using pip, conda or Docker
### via pip 

Install `virtualenv` via pip (if not already installed):
```Shell
pip install virtualenv
```

Create a virtual environment called faceboxes, activate it and install requirements:
```Shell
python -m venv faceboxes 
source faceboxes/bin/activate
pip install -r requirements.txt
```

### via Conda
Create a conda environment starting from `environment.yml` file.
```Shell 
conda env create -f environment.yml
```

and activate the environment using
```Shell 
conda activate faceboxes
```

### via Docker
_Prerequisites_: all zipped datasets must be placed into the root directory.

Build the server application using the following commands:
```Shell
export DOCKER_BUILDKIT=1
docker build -f Dockerfile.server -t flwr_client:0.0.3 .
```

Build the client application using the following commands:
```Shell
export DOCKER_BUILDKIT=1
docker build -f Dockerfile.client -t flwr_client:0.0.3 .
```

## Training
Train the model starting the server:
```Shell
python server.py
```

and then, start the clients:
```Shell
python client.py
```

### On Docker (build from scratch)
Run the server container specifying the server address and the number of rounds (epochs):
```Shell
docker run -p 8080:8080 -e NUM_ROUNDS=3 -e NUM_CLIENTS=2 -it flwr_server:0.0.3
```
Launch `docker run -t flwr_server:0.0.2 --help` for further information.

Run the client container specifying the server address:
```Shell
docker run -e SERVER_ADDRESS=<server-address> -e NUM_PARTITIONS=<num_partitions> -e PARTITION_ID=<partition_id> -it flwr_client:0.0.3
```
Launch `docker run -t flwr_client:0.0.3 --help` for further information.

N.B.: Docker images can be pulled directly from [Docker Hub](https://hub.docker.com/)
```Shell
docker pull gomax22/flwr_server:0.0.3
docker pull gomax22/flwr_client:0.0.3  
```

## Testing
Evaluate the trained model using:
```Shell
# dataset choices = ['AFW', 'PASCAL', 'FDDB']
python test.py --trained_model /path/to/trained_model.pth --dataset FDDB
# evaluate using cpu
python test.py --trained_model /path/to/trained_model.pth --cpu
# visualize detection results
python test.py --trained_model /path/to/trained_model.pth -s --vis_thres 0.3
```

### On Docker

```Shell
docker run -t flwr_client:0.0.3 python3 test.py --trained_model /path/to/trained_model.pth --dataset PASCAL --cpu --save_images
```

## Google Cloud Platform (GCP) deployment using Docker
Prerequisites: 
* Set up a cluster on Google Cloud Platform.
* Enabled *Cloud Dataproc API, Cloud Dataproc Control API, Compute Engine API, Cloud Loggin API.
* Enabled Docker on each VM instance.
* Docker deamon is running (`sudo systemctl start docker`).
* There's a Firewall rule already created for ports where there will be incoming traffic, from workers to master (e.g. `tcp/8081` `udp/8081`).

Master node will act as a Flower server, while all other workers will act as Flower clients.

First, pull the docker images from [Docker Hub](https://hub.docker.com/) using the following commands:

```Shell
sudo docker pull gomax22/flwr_server:0.0.3  # on master
sudo docker pull gomax22/flwr_client:0.0.3  # on workers
```

Then, start the containers.
For example
```Shell
sudo docker run -p 8081:8081 -e SERVER_ADDRESS=0.0.0.0:8081 -e NUM_CLIENTS=4 -it gomax22/flwr_server:0.0.3               # on master
sudo docker run -e SERVER_ADDRESS=10.200.0.9:8081 -e NUM_PARTITIONS=4 -e PARTITION_ID=0 -it gomax22/flwr_client:0.0.3    # on workers
sudo docker run -e SERVER_ADDRESS=10.200.0.9:8081 -e NUM_PARTITIONS=4 -e PARTITION_ID=1 -it gomax22/flwr_client:0.0.3    # on workers
sudo docker run -e SERVER_ADDRESS=10.200.0.9:8081 -e NUM_PARTITIONS=4 -e PARTITION_ID=2 -it gomax22/flwr_client:0.0.3    # on workers
sudo docker run -e SERVER_ADDRESS=10.200.0.9:8081 -e NUM_PARTITIONS=4 -e PARTITION_ID=3 -it gomax22/flwr_client:0.0.3    # on workers
```

Hint: add `-d` options to detach containers. This could be strongly useful when SSH connection to the VM instances is lost.
Connect again via SSH and then:
```Shell 
sudo docker ps -a   # get container id
sudo docker attach <container-id>
```

After training, execute the detection test on the workers using:
```Shell
docker run -t gomax22/flwr_client:0.0.3 python3 test.py --trained_model /path/to/trained_model.pth --dataset PASCAL --cpu --save_images
```
