{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE BlockArguments #-}


module RNNG where
import Model.RNNG
import Data.RNNGSentence
import Util
import Torch hiding (foldLoop, take, repeat)
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
import Debug.Trace
import System.Random
import Control.Monad


-- | 推論したactionのサイズと異なる場合、-1で埋めて調整する
aligned :: [Action] -> [Action] -> ([Action], [Action])
aligned prediction answer = 
  let predictionSize = length prediction
      answerSize = length answer
      alignedPrediction = if predictionSize < answerSize
                          then prediction ++ (replicate (answerSize - predictionSize) ERROR)
                          else prediction
      alignedAnswer = if predictionSize > answerSize
                        then answer ++ (replicate (predictionSize - answerSize) ERROR)
                        else answer
  in (alignedPrediction, alignedAnswer)

classificationReport :: [[Action]] -> [[Action]] -> T.Text
classificationReport predictions answers =
  let (alignedPredictions, alignedAnswers) = unzip $ map (\(prediction, answer) -> aligned prediction answer) (zip predictions answers)
  in showClassificationReport 10 $ zip (join alignedPredictions) (join alignedAnswers)

data TrainingConfig = TrainingConfig {
  iter :: Int, -- epoch数
  validationStep :: Int,
  learningRate :: LearningRate
}

training ::
  (Optimizer o) =>
  Device ->
  TrainingConfig ->
  (RNNG, o) ->
  IndexData ->
  -- | trainingData, validationData
  ([RNNGSentence], [RNNGSentence]) ->
  IO (RNNG, [[Float]])
training device TrainingConfig{..} (rnng, optim) IndexData {..} (trainingData, validationData) = do
    ((trained, _), losses) <- mapAccumM [1..iter] (rnng, (optim, optim, optim)) epochStep
    return (trained, losses)
    where
      batches = makeBatch validationStep trainingData
      epochStep :: (Optimizer o) => Int -> (RNNG, (o, o, o)) -> IO ((RNNG, (o, o, o)), [Float])
      epochStep epoch model = mapAccumM [1..(length batches)] model batchStep
      -- | TODO:: Batch処理にする
      batchStep :: (Optimizer o) => Int -> (RNNG, (o, o, o)) -> IO ((RNNG, (o, o, o)), Float)
      batchStep idx (batchRNNG, batchOpts) = do
        let batch = batches !! (idx - 1)
        ((batchRNNG', batchOpts'), batchLoss) <- mapAccumM batch (batchRNNG, batchOpts) step
        
        let trainingLoss = sum batchLoss / (fromIntegral (length batch)::Float)
        putStrLn $ "#" ++ show idx 
        putStrLn $ "Training Loss: " ++ show trainingLoss

        (validationLoss, validationPrediction) <- evaluate device batchRNNG' IndexData {..}  validationData
        putStrLn $ "Validation Loss(To not be any help): " ++ show validationLoss
        sampleRandomData 5 (zip validationData validationPrediction) >>= printResult  
        putStrLn "======================================"
        return ((batchRNNG', batchOpts'), validationLoss)
      step :: (Optimizer o) => RNNGSentence -> (RNNG, (o, o, o)) -> IO ((RNNG, (o, o, o)), Float)
      step rnngSentence@(RNNGSentence (_, actions)) (stepRNNG@(RNNG actionPredictRNNG parseRNNG compRNNG), (stepOpt1, stepOpt2, stepOpt3)) = do
        let answer = toDevice device $ asTensor $ fmap actionIndexFor actions
            output = rnngForward device Train stepRNNG IndexData {..}  rnngSentence
        dropoutOutput <- forM output (dropout 0.2 True) 
        let loss = nllLoss' answer (Torch.stack (Dim 0) dropoutOutput)
        -- | パラメータ更新
        (updatedActionPredictRNNG, stepOpt1') <- runStep actionPredictRNNG stepOpt1 loss learningRate
        (updatedParseRNNG, stepOpt2') <- runStep parseRNNG stepOpt2 loss learningRate
        (updatedCompRNNG, stepOpt3') <- if length (filter (== REDUCE) actions) > 1
                                        then runStep compRNNG stepOpt3 loss learningRate
                                      else return (compRNNG, stepOpt3)
        return ((RNNG updatedActionPredictRNNG updatedParseRNNG updatedCompRNNG, (stepOpt1', stepOpt2', stepOpt3')), (asValue loss::Float))


evaluate ::
  Device ->
  RNNG ->
  IndexData ->
  [RNNGSentence] ->
  -- | (loss, predicted action sequence)
  IO (Float, [[Action]])
evaluate device rnng IndexData {..} rnngSentences = do
  (_, result) <- mapAccumM rnngSentences rnng step
  let (losses, predictions) = unzip result
      size = fromIntegral (length rnngSentences)::Float
  return ((sum losses) / size, reverse predictions)
  where
    step :: RNNGSentence -> RNNG -> IO (RNNG, (Float, [Action]))
    step rnngSentence@(RNNGSentence (_, actions)) rnng' = do
      let answer = toDevice device $ asTensor $ fmap actionIndexFor actions
          output = rnngForward device Train rnng IndexData {..} rnngSentence
          predictionTensor = Torch.stack (Dim 0) $ fmap (argmax (Dim 0) RemoveDim) output
          prediction = fmap indexActionFor (asValue predictionTensor::[Int])
          -- | lossの計算は答えと推論結果の系列長が同じ時だけ
          loss = if length prediction == length prediction
                  then nllLoss' answer (Torch.stack (Dim 0) output)
                  else asTensor (0::Float)          
      return (rnng', ((asValue loss::Float), prediction))

{-

  Util function for RNNG

-}

sampleRandomData ::
  -- | 取り出したいデータ数
  Int ->
  -- | データ
  [a] ->
  -- | サンプル結果
  IO [a]
sampleRandomData size xs = do
  gen <- newStdGen
  let randomIdxes = Data.List.take size $ nub $ randomRs (0, (length xs) - 1) gen
  return $ map (xs !!) randomIdxes


printResult ::
  [(RNNGSentence, [Action])] ->
  IO ()
printResult result = forM_ result \(RNNGSentence (words, actions), predition) -> do
  putStrLn "----------------------------------"
  putStrLn $ "Sentence: " ++ show words
  putStrLn $ "Actions:    " ++ show actions
  putStrLn $ "Prediction: " ++ show predition
  putStrLn "----------------------------------"
  return ()


main :: IO()
main = do
  -- experiment setting
  config <- configLoad
  putStrLn "Experiment Setting: "
  print config
  putStrLn "======================================"

  let mode = modeConfig config
      wordEmbedSize = fromIntegral (wordEmbedSizeConfig config)::Int
      actionEmbedSize = fromIntegral (actionEmbedSizeConfig config)::Int
      hiddenSize = fromIntegral (hiddenSizeConfig config)::Int
      numLayer = fromIntegral (numOfLayerConfig config)::Int
      modelName = modelNameConfig config
      modelFilePath = "models/" ++ modelName
      modelSpecPath = "models/" ++ modelName ++ "-spec"
      graphFilePath = "imgs/" ++ modelName ++ ".png"
      reportFilePath = "reports/" ++ modelName ++ ".txt"
      device = Device CUDA 0
      trainingConfig = TrainingConfig {
        iter = fromIntegral (epochConfig config)::Int,
        learningRate = toDevice device $ asTensor (learningRateConfig config),
        validationStep = fromIntegral (validationStepConfig config)::Int
      }
      optim = GD
  -- data
  trainingData <- loadActionsFromBinary $ trainingDataPathConfig config
  validationData <- loadActionsFromBinary $ validationDataPathConfig config
  evaluationData <- loadActionsFromBinary $ evaluationDataPathConfig config
  let dataForTraining = trainingData
  putStrLn $ "Training Data Size: " ++ show (length trainingData)
  putStrLn $ "Validation Data Size: " ++ show (length validationData)
  putStrLn $ "Evaluation Data Size: " ++ show (length evaluationData)
  putStrLn "======================================"

  -- create index data
  let (wordIndexFor, indexWordFor, wordEmbDim) = indexFactory (buildVocab dataForTraining 0 toWordList) (T.pack "unk") Nothing
      (actionIndexFor, indexActionFor, actionEmbDim) = indexFactory (buildVocab dataForTraining 0 toActionList) ERROR Nothing
      (ntIndexFor, indexNTFor, ntEmbDim) = indexFactory (buildVocab dataForTraining 0 toNTList) (T.pack "unk") Nothing
      indexData = IndexData wordIndexFor indexWordFor actionIndexFor indexActionFor ntIndexFor indexNTFor
  putStrLn $ "WordEmbDim: " ++ show wordEmbDim
  putStrLn $ "ActionEmbDim: " ++ show actionEmbDim
  putStrLn $ "NTEmbDim: " ++ show ntEmbDim
  putStrLn "======================================"

  when (mode == "Train") $ do
    let rnngSpec = RNNGSpec device wordEmbedSize actionEmbedSize wordEmbDim actionEmbDim ntEmbDim hiddenSize
    initRNNGModel <- toDevice device <$> sample rnngSpec

    -- | training
    (trained, losses) <- training device trainingConfig (initRNNGModel, optim) indexData (trainingData, validationData)

    -- | model保存
    B.encodeFile modelSpecPath rnngSpec
    Torch.Train.saveParams trained modelFilePath
    drawLearningCurve graphFilePath "Learning Curve" [("", reverse $ concat losses)]

  -- | evaluation
  rnngSpec <- (B.decodeFile modelSpecPath)::(IO RNNGSpec)
  print rnngSpec
  rnngModel <- Torch.Train.loadParams rnngSpec modelFilePath

  (_, evaluationPrediction) <- evaluate device rnngModel indexData evaluationData
  let answers = fmap (\(RNNGSentence (_, actions)) -> actions) evaluationData
  T.writeFile reportFilePath $ classificationReport evaluationPrediction answers
  sampleRandomData 10 (zip evaluationData evaluationPrediction) >>= printResult
  return ()
