# Recurrent Neaural Network Grammars (RNNGs) based on hasktorch

This repository provides an RNNG implemented in haskell.
In addition, this model can handle not only Context Free Grammar (CFG) but also Combinatory Categorial Grammar (CCG) as an internal syntax grammar.

## Pre-requirements
#### hasktorch
Follow [this page](https://github.com/hasktorch/hasktorch#getting-started) to install hasktorch, a library for tensors and neural networks in Haskell.

#### hasktorch-tools
Clone [hasktorch-tools](https://github.com/DaisukeBekki/hasktorch-tools) to your environment.
This library contains useful functions for building neural network model.

#### Path Setting
Edit the hasktorch and hasktorch-tools paths in `stack.yaml` to match your environment.

## Data Preparation
RNNGs use Wall Street Jarnal corpus of Penn Treebank or CCGbank as data for English.
- Training data : \$2 - \$21
- Validation data : \$23
- Evaluation data : \$24

Run the following command to preprocess those data for rnng-hasktorch. This command generates three files (training[grammar], validation[grammar] and evaluation[grammar]) under the `data/` repository.

```
$ stack run Preprocessing -- --path [/path/to/wsj] --grammar [CFG or CCG]
```

## Training and Evaluation
The experiment is configured in `config.dhall`. If you want to change the configuration, rewrite this file.

|  Config name  |    |
| ---- | ---- |
|  modeConfig  |　Choose `Train` or `Eval` mode |
|  **parsingModeConfig | Parsing mode during `Eval` mode <br>　`Point`: Use correct data for prediction results before each time step <br>　`All`　: All actions are predicted by the learned model|
|  trainingDataPathConfig | Training data created in `Preprocessing`  |
|  validationDataPathConfig | Validation data created in `Preprocessing`  |
|  evaliationDataPathConfig | Evaliation data created in `Preprocessing`  |
|  actionEmbedSizeConfig | Size ofAction embedding  |
|  wordEmbedSizeConfig | Size of word embedding  |
|  hiddenSizeConfig | Size of hidden Layer |
|  numOfLayerConfig | Number of LSTM  layer |
|  learningRateConfig | Learning late during `Train` mode |
|  *epochConfig  | Number of epoch during training |
|  *validationStepConfig  | How many steps to validate every during `Train` mode |
|  *modelNameConfig  | Name of model (to be saved during `Train` mode \| to be loaded during `Eval` mode) |

<div style="text-align: right;">
*used only by `Train` mode

**used only by `Eval` mode
</div>
Run the following command to train or evaluate of RNNG.
```
$ stack run RNNG
```