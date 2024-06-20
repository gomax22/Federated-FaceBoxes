
# Federated Learning for Face Detection using FaceBoxes on WIDER FACE dataset
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

**Federated FaceBoxes** is a [Flower-based](https://flower.ai) implementation of "Federated Learning for Face Detection using FaceBoxes on WIDER FACE dataset", taking inspiration from a [PyTorch](https://pytorch.org/) implementation of ["_Zhang et al._, FaceBoxes: A CPU Real-time Face Detector with High Accuracy"](https://arxiv.org/abs/1708.05234) and [Flower examples](https://github.com/adap/flower/tree/main/examples). 

The original code can be found [here](https://github.com/zisianw/FaceBoxes.PyTorch).


## Cloud-based Deployment Strategies of Hierarchical Federated Learning Systems for Face Recognition
This repository also contains the materials for Cloud Computing exam project [@uniparthenope](https://github.com/uniparthenope) a.y. 2023/2024 (paper submission, presentation and project implementation). 

### Authors 
* [R. Esposito](https://github.com/RenatoEsposito1999)
* [V. Mele](https://github.com/MeleVincenzo)
* [A. Mungari](https://github.com/mungowz)
* M. Giordano Orsini (me)
* [M. Roscica](https://github.com/martirrrrr)
* [S. Verrilli](https://github.com/StefanoVerrilli)

All the authors contributed equally to this work.


![frontcover](/paper/frontcover.jpg)

Check out our work (paper and slides) at `paper/`.

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
4. Place the dowloaded datasets into the corresponding folder at `data/<dataset>/`


N.B. It's mandatory to download at least one of the test datasets in order to test the application. 

### Extraction
```Shell 
unzip data/WIDER_FACE/WIDER_train.zip -d data/ && mv data/WIDER_train/images data/WIDER_FACE  && rm -rf data/WIDER_train    
unzip data/AFW/afw_images.zip -d data/ && mv data/afw_images data/AFW && rm -rf data/afw_images                             # if downloaded
unzip data/PASCAL/pascal_images.zip -d data/ && mv data/pascal_images data/PASCAL && rm -rf data/pascal_images              # if downloaded
unzip data/FDDB/fddb_images.zip -d data/ && mv data/fddb_images data/FDDB && rm -rf data/fddb_images                        # if downloaded
tar xvf data/WIDER_FACE/annotations.tar.gz -C data/WIDER_FACE/                                                              
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

**WARNING**: in this environment PyTorch has been installed only for CPUs. For installing CUDA-enabled PyTorch, please visit https://pytorch.org/get-started/locally/.

### via Docker
_Prerequisites_: all zipped datasets must be placed into the corresponding folders at `data/`.

Build the server application using the following commands:
```Shell
export DOCKER_BUILDKIT=1
docker build -f docker/server/Dockerfile.server -t flwr_client:0.0.3 .
```

Build the client application using the following commands:
```Shell
export DOCKER_BUILDKIT=1
docker build -f docker/client/Dockerfile.client -t flwr_client:0.0.3 .
```

## Training
Before starting to train, datasets should be partitioned and distributed among clients in order to simulate a real-world federated scenario. 

To simulate this scenario, launch the following commands:
```Shell
python split.py --img_list data/WIDER_FACE/img_list.txt --partitions 2 --output data/WIDER_FACE
```

`split.py` creates `n` partitions starting from the corresponding `img_list.txt` file, which we'll be used exclusively by each client.

Train the model starting the server:
```Shell
python server.py --num_rounds 200 --num_clients 2
```
Launch `python server.py --help` for further information.


and then, start the clients:
```Shell
python client.py --partition_id 0
python client.py --partition_id 1
```
Launch `python client.py --help` for further information.


### On Docker (build from scratch)
Run the server container specifying the server address and the number of rounds (epochs):
```Shell
docker run -p 8080:8080 -e NUM_ROUNDS=3 -e NUM_CLIENTS=2 -it flwr_server:1.0.0
```
Launch `docker run -t flwr_server:1.0.0 --help` for further information.

Run the client container specifying the server address:
```Shell
docker run -e SERVER_ADDRESS=<server-address> -e NUM_PARTITIONS=<num_partitions> -e PARTITION_ID=<partition_id> -it flwr_client:1.0.0
```
Launch `docker run -t flwr_client:1.0.0 --help` for further information.

N.B.: Docker images can be pulled directly from [Docker Hub](https://hub.docker.com/)
```Shell
docker pull gomax22/flwr_server:1.0.0
docker pull gomax22/flwr_client:1.0.0  
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
docker run -it flwr_client:1.0.0 python3 test.py --trained_model /path/to/trained_model.pth --dataset PASCAL --cpu --save_image
```

## Google Cloud Platform (GCP) deployment using Docker
Prerequisites: 
* Set up a cluster on Google Cloud Platform.
* Enabled Cloud Dataproc API, Cloud Dataproc Control API, Compute Engine API, Cloud Logging API.
* Enabled Docker on each VM instance.
* Docker deamon is running (`sudo systemctl start docker`).
* There's a Firewall rule already created for ports where there will be incoming traffic, from workers to master (e.g. `tcp/8081` `udp/8081`).

Master node will act as a Flower server, while all other workers will act as Flower clients.

First, pull the docker images from [Docker Hub](https://hub.docker.com/) using the following commands:

```Shell
sudo docker pull gomax22/flwr_server:1.0.0  # on master
sudo docker pull gomax22/flwr_client:1.0.0  # on workers
```

Then, start the containers.
For example
```Shell
sudo docker run -p 8081:8081 -e SERVER_ADDRESS=0.0.0.0:8081 -e NUM_CLIENTS=4  -e NUM_ROUNDS=200 -it gomax22/flwr_server:1.0.0   # on master
sudo docker run -e SERVER_ADDRESS=10.200.0.9:8081 -e NUM_PARTITIONS=4 -e PARTITION_ID=0 -it gomax22/flwr_client:1.0.0           # on workers
sudo docker run -e SERVER_ADDRESS=10.200.0.9:8081 -e NUM_PARTITIONS=4 -e PARTITION_ID=1 -it gomax22/flwr_client:1.0.0           # on workers
sudo docker run -e SERVER_ADDRESS=10.200.0.9:8081 -e NUM_PARTITIONS=4 -e PARTITION_ID=2 -it gomax22/flwr_client:1.0.0           # on workers
sudo docker run -e SERVER_ADDRESS=10.200.0.9:8081 -e NUM_PARTITIONS=4 -e PARTITION_ID=3 -it gomax22/flwr_client:1.0.0           # on workers
```

_Hint 1_ : add `-d` options to detach containers. This could be strongly useful when SSH connection to the VM instances is lost.

_Hint 2_: you can check logs using the command: `sudo docker logs <container-id>`

Connect again via SSH and then:
```Shell 
sudo docker ps  # get container id
sudo docker attach <container-id>
```

After training, execute the detection test on the workers using:
```Shell
# dataset choices = ['AFW', 'PASCAL', 'FDDB']
sudo docker run -it gomax22/flwr_client:1.0.0 python3 test.py --trained_model /path/to/trained_model.pth --dataset PASCAL --cpu --save_image

# or
sudo docker ps -a   # get container id
sudo docker exec -it <container-id> python3 test.py --trained_model /path/to/trained_model.pth --dataset PASCAL --cpu --save_image
```
