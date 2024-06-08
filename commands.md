
# Federated FaceBoxes

## Installation
Clone this repository.

```Shell
git clone https://github.com/gomax22/Federated-FaceBoxes.git
cd Federated-FaceBoxes
git checkout dev
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
N.B. It's not mandatory to download all of three testing datasets (but at least one for testing the application). 

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

rm -rf data/WIDER_train
rm path/to/pascal_images.zip                            # if downloaded
rm path/to/afw_images.zip                               # if downloaded
rm path/to/fddb_images.zip                              # if downloaded
```

## Install dependencies using pip, conda or Docker
### pip

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

### Conda
Create a conda environment starting from `environment.yml` file.
```Shell 
conda env create -f environment.yml
```

and activate the environment using
```Shell 
conda activate faceboxes
```

### Docker
_Work in progress..._

## Training
Train the model starting the server:
```Shell
python server.py
```

and then, start the clients:
```Shell
python client.py
```
## Testing
Evaluate the trained model using:
```Shell
# dataset choices = ['AFW', 'PASCAL', 'FDDB']
python3 test.py --trained_model /path/to/trained_model.pth --dataset FDDB
# evaluate using cpu
python3 test.py --trained_model /path/to/trained_model.pth --cpu
# visualize detection results
python3 test.py --trained_model /path/to/trained_model.pth -s --vis_thres 0.3
```
