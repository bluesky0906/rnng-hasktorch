{-# LANGUAGE RecordWildCards #-}

module Evaluation where
import Data.RNNGSentence
import Data.SyntaxTree
import Data.CFG
import Data.CCG
import Util.File
import Util
import Model.RNNG

import Torch
-- | hasktorch-tools
import Torch.Train (update, saveParams, loadParams)

import qualified Data.Text as T          --text
import qualified Data.Text.IO as T       --text
import qualified Data.Binary as B
import Options.Applicative


data EvaluationOpt = EvaluationOpt
  {
    path :: Maybe FilePath,
    model :: String,
    grammar :: Grammar,
    pos :: Bool
  } deriving (Show)

evaluation :: Parser EvaluationOpt
evaluation = EvaluationOpt
  <$> (optional $ strOption ( long "path" <> short 'p' <> metavar "FILE_PATH" <> help "path to WSJ" ))
  <*> strOption ( long "model" <> short 'm' <> metavar "MODEL_NAME" <> help "model name" )
  <*> option auto ( long "grammar" <> short 'g' <> metavar "GRAMMAR" <> help "CFG or CCG" )
  <*> switch ( long "pos" <> help "include POS-tag or not" )

opts :: ParserInfo EvaluationOpt
opts = info (evaluation <**> helper)
  ( fullDesc
    <> progDesc "Evaluate as CFG/CCG parser"
  )

parseTreefile ::
  Grammar ->
  Bool ->
  FilePath ->
  IO [Tree]
parseTreefile CFG _ = parseCFGfile
parseTreefile CCG posOpt = parseCCGfile posOpt

predictActions ::
  Mode ->
  RNNG ->
  IndexData ->
  RNNGSentence ->
  -- | (loss, predicted action sequence)
  [Action]
predictActions mode@Mode{..} rnng indexData@IndexData{..} rnngSentence =
  let output = rnngForward mode rnng indexData rnngSentence
      predictionTensor = Torch.stack (Dim 0) $ fmap (argmax (Dim 0) RemoveDim) output
      prediction = fmap indexActionFor (asValue predictionTensor::[Int])
  in prediction

main :: IO()
main = do
  options <- execParser opts
  let pathOpt = path options
      grammarOpt = grammar options
      posOpt = pos options
      modelName = model options
  goldDataPath <- parserEvalFilePath Nothing grammarOpt posOpt
  -- wsj pathが与えられた時だけgoldDataを生成
  case pathOpt of
    Just wsjDirPath -> do
      -- 評価はwsjの$23を用いる
      filePaths <- treefiles grammarOpt [wsjDirPath ++ "23/"]
      treess <- mapM (parseTreefile grammarOpt posOpt) filePaths
      -- posOptを考慮するため一度RNNGSentencesにする
      let trees = fromRNNGSentences $ toRNNGSentences posOpt $ concat treess
      T.writeFile goldDataPath $ T.intercalate (T.pack "\n") $ fmap (evalFormat posOpt) trees
    Nothing -> return ()
  
  let modelFilePath = modelPath modelName
      modelSpecPath = specPath modelName
      (trainDataPath, evalDataPath, validDataPath) = dataFilePath grammarOpt posOpt
  rnngSpec <- B.decodeFile modelSpecPath::(IO RNNGSpec)
  rnngModel <- Torch.Train.loadParams rnngSpec modelFilePath

  -- TODO: indexDataも保存しておく
  trainingData <- loadActionsFromBinary trainDataPath
  validationData <- loadActionsFromBinary validDataPath
  evaluationData <- loadActionsFromBinary evalDataPath

  let (wordIndexFor, indexWordFor, wordEmbDim) = indexFactory (buildVocab trainingData 0 toWordList) (T.pack "unk") Nothing
      (actionIndexFor, indexActionFor, actionEmbDim) = indexFactory (buildVocab trainingData 0 toActionList) (NT (T.pack "unk")) Nothing
      (ntIndexFor, indexNTFor, ntEmbDim) = indexFactory (buildVocab trainingData 0 toNTList) (T.pack "unk") Nothing
      indexData = IndexData wordIndexFor indexWordFor actionIndexFor indexActionFor ntIndexFor indexNTFor


  let mode = Mode {
                  device = modelDevice rnngSpec,
                  parsingMode = All,
                  dropoutProb = Nothing,
                  posMode = modelPosMode rnngSpec,
                  grammarMode = grammarOpt
                }
  let predictedActions = map (predictActions mode rnngModel indexData) evaluationData
      predictedRNNGSentences = zipWith insertDifferentActions evaluationData predictedActions
      predictedTrees = fromRNNGSentences predictedRNNGSentences
  predFilePath <- parserEvalFilePath (Just modelName) grammarOpt posOpt
  T.writeFile predFilePath $ T.intercalate (T.pack "\n") $ fmap (evalFormat posOpt) predictedTrees
  return ()

