# CCN HW2

Compare different learning rules on SNN training.

## Setup

Some needed library

```
pip3 install -r requirements.txt
```

## Training

training command with differnet config inputs.

```
python3 train.py ./configs/NoOp.yaml
```

## Evaluation

evaluation command, please type in the name of directory, results will be saved in results.csv.

```
python3 eval.py NoOp
```

## Visualization

visualize training history of each model.

```
python3 plot_history.py Model1FolderName Model2FolderName ... all
```
