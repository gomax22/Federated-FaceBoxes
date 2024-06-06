
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
1. Download [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) dataset (train, val, test splits)
2. Download [converted annotations](https://drive.google.com/open?id=1-s4QCu_v76yNwR-yXMfGqMGgHQ30WxV2) directly from original repository [FaceBoxes.Pytorch](https://github.com/zisianw/FaceBoxes.PyTorch/edit/master/) 

### Extraction
```Shell 
unzip path/to/WIDER_train.zip -d data/
unzip path/to/WIDER_val.zip -d data/
unzip path/to/WIDER_test.zip -d data/
tar xvf path/to/annotations.tar.gz -C data/WIDER_FACE/

cp -a data/WIDER_train/images data/WIDER_FACE
cp -a data/WIDER_val/images data/WIDER_FACE
cp -a data/WIDER_test/images data/WIDER_FACE

rm -rf data/WIDER_train
rm -rf data/WIDER_val
rm -rf data/WIDER_test
```

## Install dependencies
Create a conda environment starting from `environment.yml` file.
```Shell 
conda env create -f environment.yml
```

and activate the environment
```Shell 
conda activate faceboxes
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