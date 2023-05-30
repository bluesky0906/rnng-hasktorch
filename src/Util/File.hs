module Util.File where
import Torch
import Torch.Train (update, saveParams, loadParams) -- hasktorch-tools
import Text.Directory (checkFile, getFileList) --nlp-tools
import Data.List
import Control.Monad
import System.Directory (createDirectory, doesDirectoryExist, doesFileExist)

import Data.SyntaxTree


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

parserEvalFilePath ::
  -- teacher or not
  Maybe String ->
  -- | grammar
  Grammar ->
  -- | pos mode
  Bool ->
  IO FilePath
parserEvalFilePath modelName grammar posMode = do
  let suffix = show grammar ++ if posMode then "POS" else ""
      dataDir = "data/parser"
      fileName = case modelName of
                  Just m -> m ++ ".pred"
                  Nothing -> suffix ++ ".gold" 
  existGoldDataDir <- doesDirectoryExist dataDir
  unless existGoldDataDir $ createDirectory dataDir
  return $ dataDir ++ "/" ++ fileName

dataFilePath ::
  -- | grammar
  String ->
  -- | pos mode
  Bool ->
  -- | train, eval, valid
  (String, String, String)
dataFilePath grammar posMode = 
  ("data/training" ++ suffix, "data/evaluation" ++ suffix, "data/validation" ++ suffix)
  where
    suffix = grammar ++ if posMode then "POS" else ""


