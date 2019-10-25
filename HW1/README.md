# HW1

In this part we compare differnet depth of CNN and SpikeCNN in MNIST dataset.

## Setup

PLease ensure that you have tensorflow(1.14.0 recommended), nengo, nengo_dl, and pytorch. <br />
Please download MNIST from Kaggle. [link](https://www.kaggle.com/oddrationale/mnist-in-csv) <br />
Then unzip it into ./data directory

### Quick setup for Mac OS / Linux

Temporarily no

### Quick setup for Windows

Temporarily no

## Training part

There were two different model was implemented, Convolutional neural network and SpikeCNN.

### CNN (PyTorch)

```
python3 train_cnn.py ./configs/cnn_example.yaml
```

### SpikeCNN (NengoDL, Tensorflow)

In this part still have some difficulties, thus, API for SpikeCNN is only for training, and the program is very slow.

```
python3 train_snn.py ./configs/snn_example.yaml
```

## Testing part

Please store your data in ./data before inference

### CNN (PyTorch)

```
python3 eval.py CNN torch
```

### SpikeCNN (NengoDL, Tensorflow)

```
python3 eval.py SNN nengo
```

## Visualize part

For visualize training history and other comparsion plot

### CNN (PyTorch)

```
python3 plot_cnn_history.py Model1FolderName Model2FolderName ... all
```

### SpikeCNN (NengoDL, Tensorflow)

Temporarily no

