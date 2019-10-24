# HW1

## Training Commands

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

## Visualize part

For visualize training history and other comparsion plot

### CNN (PyTorch)

```
python3 plot_cnn_history.py Model1FolderName Model2FolderName ... all
```

### SpikeCNN (NengoDL, Tensorflow)

Temporarily no
