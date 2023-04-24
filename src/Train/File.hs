module Train.File where
import Torch
import Torch.Train (update, saveParams, loadParams) -- hasktorch-tools
import Text.Directory (checkFile, getFileList) --nlp-tools
import Data.List

checkpointPath ::
  String ->
  FilePath
checkpointPath modelName = "models/" ++ modelName ++ "-checkpoints"

checkpointModelPath ::
  String ->
  Int -> 
  FilePath
checkpointModelPath modelName epoch = checkpointPath modelName ++ "/epoch-" ++ show epoch ++ ".model"

checkpointImgPath ::
  String ->
  Int -> 
  FilePath
checkpointImgPath modelName epoch = checkpointPath modelName ++ "/epoch-" ++ show epoch ++ ".png"

modelPath ::
  String ->
  FilePath
modelPath modelName = "models/" ++ modelName ++ ".model"

specPath ::
  String ->
  FilePath
specPath modelName = "models/" ++ modelName ++ ".spec"

-- | 最新のcheckpointを返す
loadCheckPoint ::
  (Parameterized p, Eq spec, Randomizable spec p) =>
  -- | modelName
  String ->
  -- | spec
  spec ->
  -- | (model, epoch)
  IO (p, Int)
loadCheckPoint modelName spec = do
  list <- getFileList "model" $ checkpointPath modelName
  let latestModelFile = head $ sortBy (flip compare) list
  model <- Torch.Train.loadParams spec latestModelFile
  return (model, length list)
