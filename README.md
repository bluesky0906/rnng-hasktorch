# rnng-hasktorch

## Pre-requirements
#### hasktorch
Follow [this page](https://github.com/hasktorch/hasktorch#getting-started) to install hasktorch, a library for tensors and neural networks in Haskell.

#### hasktorch-tools
Clone [hasktorch-tools](https://github.com/DaisukeBekki/hasktorch-tools) to your environment.
This library contains useful functions for building neural network model.

#### Path Setting
Edit the hasktorch and hasktorch-tools paths in `stack.yaml` to match your environment.

## Data Preparation
RNNG use WSJ directory of PennTreeBank as data for English.
- Training data : \$2 - \$21
- Validation data : \$23
- Evaluation data : \$24

Run the following command to preprocess those data for rnng-hasktorch.

```
$ stack run Preprocessing -- --path [/path/to/wsj]
```

## Training and Evaluation
Developing...