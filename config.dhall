{
  modeConfig = "Train", -- | Train or Eval
  trainingDataPathConfig = "data/trainingCFG",
  validationDataPathConfig = "data/validationCFG",
  evaluationDataPathConfig = "data/evaluationCFG",
  epochConfig = 1, --only for training 
  validationStepConfig = 400, --only for training 
  actionEmbedSizeConfig = 128, --only for training 
  wordEmbedSizeConfig = 128, --only for training 
  hiddenSizeConfig = 128, --only for training 
  numOfLayerConfig = 2, --only for training 
  learningRateConfig = 1e-2, --only for training 
  modelNameConfig = "rnng-hasktorch-cfg"
}