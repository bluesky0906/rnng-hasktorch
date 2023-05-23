{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE BlockArguments #-}


module RNNGGen where
import Model.Generative.RNNG
import Data.RNNGSentence
import Data.SyntaxTree
import Data.CCG
import Data.RNNG
import Util
import Train.File
import Torch hiding (foldLoop, take, repeat, RuntimeMode)
-- | hasktorch-tools
import Torch.Control (mapAccumM, makeBatch)
import Torch.Train (update, saveParams, loadParams)
-- | nlp-tools
import ML.Exp.Chart (drawLearningCurve)
import ML.Exp.Classification (showClassificationReport)
import qualified Data.Text.IO as T --text
import qualified Data.Text as T    --text
import qualified Data.Binary as B
import Data.List
import Data.Functor
import Data.Either
import System.Directory (createDirectory, doesDirectoryExist, doesFileExist)
import System.IO
import Text.Directory (checkFile) --nlp-tools
import Debug.Trace
import Control.Monad


data TrainingConfig = TrainingConfig {
  iter :: [Int], -- epoch数
  batchSize :: Int,
  validationStep :: Int,
  learningRate :: LearningRate,
  modelName :: String
}

training ::
  (Optimizer o) =>
  Mode ->
  TrainingConfig ->
  (RNNG, o) ->
  IndexData ->
  -- | trainingData, validationData
  (([RNNGSentence]), [RNNGSentence]) ->
  IO (RNNG, [[Float]])
training mode@Mode{..} TrainingConfig{..} (rnng, optim) IndexData {..} (trainingData, validationData) = do
  existCheckpointDir <- doesDirectoryExist checkpointDirectory 
  unless existCheckpointDir $ createDirectory checkpointDirectory
  ((trained, _), losses) <- mapAccumM iter (rnng, optim) epochStep
  return (trained, losses)
  where
    checkpointDirectory = "models/" ++ modelName ++ "-checkpoints"
    (sents, actions) = unzipRNNGSentence trainingData
    batches = zip (fmap (indexForBatch wordIndexFor) (makeBatch batchSize sents)) (fmap (indexForBatch actionIndexFor) (makeBatch batchSize actions))
    epochStep :: (Optimizer o) => Int -> (RNNG, o) -> IO ((RNNG, o), [Float])
    epochStep epoch model = do
      trained@((trainedModel, trainedOpts), losses) <- mapAccumM [1..(length batches)] model batchStep
      -- | 1stepごとにcheckpointとしてモデル保存
      -- | TODO::　optimizerも保存
      Torch.Train.saveParams trainedModel $ checkpointModelPath modelName epoch
      drawLearningCurve (checkpointImgPath modelName epoch) "Learning Curve" [("", reverse losses)]
      
      -- validation
      -- (validationLoss, validationPrediction) <- evaluate (Mode device Nothing parsingMode posMode) trainedModel IndexData {..} validationData
      -- putStrLn $ "Validation Loss(To not be any help): " ++ show validationLoss
      -- sampledData <- sampleRandomData 5 (zip validationData validationPrediction)
      -- putStr $ unlines $ map (uncurry showResult) sampledData
      -- putStrLn "======================================"
      return ((trainedModel, trainedOpts), losses) -- ((trainedModel, trainedOpts), validationLoss) 
    batchStep :: (Optimizer o) => Int -> (RNNG, o) -> IO ((RNNG, o), Float)
    batchStep idx (batchRNNG, batchOpts) = do
      let (batchedSents, batchedActions) = batches !! (idx - 1)
      let answer = toDevice device $ asTensor batchedActions
          (loss, aLoss, wLoss) = rnngForward mode batchRNNG IndexData {..} batchedSents batchedActions
          nomalizedLoss = loss / asTensor' (length batchedSents) (withDevice (Torch.device loss) defaultOpts)
      updatedBatchRNNG <- runStep batchRNNG batchOpts nomalizedLoss learningRate
      lossValue <- detach nomalizedLoss
      putStrLn $ "Training Loss: " ++ show nomalizedLoss
          -- trainingLoss = calculateLoss
      -- putStrLn $ "#" ++ show idx 
      -- putStrLn $ "Training Loss: " ++ show trainingLoss
      return (updatedBatchRNNG, (asValue lossValue::Float)) -- (updated, trainingLoss)


main :: IO()
main = do
  -- experiment setting
  config <- configLoad
  putStrLn "Experiment Setting: "
  print config
  putStrLn "======================================"
  let mode = modeConfig config
      posMode = posModeConfig config
      grammarMode = grammarModeConfig config
      parsingMode = case parsingModeConfig config of 
                      "Point" -> Point
                      "All" -> All
      resumeTraining = resumeTrainingConfig config
      batchSize = fromIntegral (batchSizeConfig config)::Int
      modelName = modelNameConfig "gen" config
  let modelFilePath = modelPath modelName
      modelSpecPath = specPath modelName
      (trainDataPath, evalDataPath, validDataPath) = dataFilePath grammarMode posMode
  -- data
  trainingData <- loadActionsFromBinary trainDataPath
  validationData <- loadActionsFromBinary validDataPath
  evaluationData <- loadActionsFromBinary evalDataPath
  let dataForTraining = trainingData
  putStrLn $ "Training Data Size: " ++ show (length trainingData)
  putStrLn $ "Validation Data Size: " ++ show (length validationData)
  putStrLn $ "Evaluation Data Size: " ++ show (length evaluationData)
  putStrLn "======================================"

  -- create index data
  let (wordIndexFor, indexWordFor, numWords) = indexFactory (buildVocab dataForTraining 0 toWordList) (T.pack "unk") (Just (T.pack "<pad>")) 
      (actionIndexFor, indexActionFor, numActions) = indexFactory (buildVocab dataForTraining 0 toActionList) (NT (T.pack "unk")) (Just ERROR)
      (ntIndexFor, indexNTFor, numNts) = indexFactory (buildVocab dataForTraining 0 toNTList) (T.pack "unk") (Just (T.pack "<pad>"))
      indexData = IndexData wordIndexFor indexWordFor actionIndexFor indexActionFor ntIndexFor indexNTFor
  putStrLn $ "WordEmbDim: " ++ show numWords
  putStrLn $ "ActionEmbDim: " ++ show numActions
  putStrLn $ "NTEmbDim: " ++ show numNts
  putStrLn "======================================"

  when (mode == "Train") $ do
    let device = Device CUDA 0
    -- model spec
    (initRNNGModel', startEpoch) <- if resumeTraining 
                                      then do
                                        rnngSpec <- B.decodeFile modelSpecPath::(IO RNNGSpec)
                                        (model, lastEpoch) <- loadCheckPoint modelName rnngSpec
                                        return (model, lastEpoch+1)
                                      else do
                                        let wordDim = fromIntegral (wordEmbedSizeConfig config)::Int
                                            hiddenSize = fromIntegral (hiddenSizeConfig config)::Int
                                            numLayers = fromIntegral (numOfLayerConfig config)::Int
                                            rnngSpec = RNNGSpec device numLayers numWords numActions numNts wordDim hiddenSize
                                        -- | spec保存
                                        B.encodeFile modelSpecPath rnngSpec
                                        initRNNGModel <- toDevice device <$> sample rnngSpec
                                        return (initRNNGModel, 1)
    floatParameters <- mapM (makeIndependent . toDType Float . toDependent) (flattenParameters initRNNGModel')
    let initRNNGModel = replaceParameters initRNNGModel' floatParameters

    -- | training
    let trainingConfig = TrainingConfig {
                           iter = [(startEpoch)..(fromIntegral (epochConfig config)::Int)],
                           batchSize = batchSize,
                           learningRate = toDevice device $ asTensor (learningRateConfig config),
                           validationStep = fromIntegral (validationStepConfig config)::Int,
                           modelName = modelName
                         }
        optim = GD
        mode = Mode {
                      device = device,
                      parsingMode = Point,
                      dropoutProb = Nothing,
                      posMode = posMode
                    }
    (trained, losses) <- training mode trainingConfig (initRNNGModel, optim) indexData (dataForTraining, validationData)
    print losses
    return()

  return ()