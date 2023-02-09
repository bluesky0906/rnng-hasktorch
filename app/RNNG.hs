{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE BlockArguments #-}


module RNNG where
import Model.RNNG
import Data.RNNGSentence
import Data.SyntaxTree
import Data.CCG
import Util
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
import System.Directory (createDirectory, doesDirectoryExist)
import Debug.Trace
import Control.Monad


data TrainingConfig = TrainingConfig {
  iter :: Int, -- epoch数
  validationStep :: Int,
  learningRate :: LearningRate,
  modelName :: String
}


-- | 推論したactionのサイズと異なる場合、-1で埋めて調整する
aligned :: [Action] -> [Action] -> ([Action], [Action])
aligned prediction answer = 
  let predictionSize = length prediction
      answerSize = length answer
      alignedPrediction = if predictionSize < answerSize
                          then prediction ++ replicate (answerSize - predictionSize) ERROR
                          else prediction
      alignedAnswer = if predictionSize > answerSize
                        then answer ++ replicate (predictionSize - answerSize) ERROR
                        else answer
  in (alignedPrediction, alignedAnswer)


{-

functions for analysis

-}

checkCorrect ::
  [(Action, Action)] ->
  Bool
checkCorrect [] = True
checkCorrect ((answer, prediction):rest)
  | answer == prediction = checkCorrect rest
  | otherwise            = False

unknownActions ::
  (Action -> Int) ->
  [Action] ->
  [Action]
unknownActions actionIndexFor [] = []
unknownActions actionIndexFor (action:rest)
  | actionIndexFor action == 0 = action:unknownActions actionIndexFor rest
  | otherwise                  = unknownActions actionIndexFor rest

classificationReport :: 
  [[Action]] ->
  [[Action]] -> 
  T.Text
classificationReport predictions answers =
  let (alignedPredictions, alignedAnswers) = unzip $ zipWith aligned predictions answers
  in showClassificationReport 20 $ zip (join alignedPredictions) (join alignedAnswers)

filterByMask ::
  [a] ->
  [Bool] ->
  [a]
filterByMask lst mask = map fst $ filter snd (zip lst mask)

showResult ::
  RNNGSentence ->
  [Action] ->
  String
showResult (RNNGSentence (words, actions)) predition = 
  unlines [
      "----------------------------------",
      "Sentence: " ++ show words,
      "Actions:    " ++ show actions,
      "Prediction: " ++ show predition
    ]

reportResult ::
  -- | validな木かどうか
  Bool ->
  -- | 答えが合っているかどうか
  Bool ->
  -- | 正解
  (RNNGSentence, Tree) ->
  -- | 予測
  ([Action], Tree, Either String Bool) ->
  String
reportResult isValid isCorrect (RNNGSentence (words, actions), correctTree) (prediction, predictedTree, ccgError) = 
  unlines $ [
      "----------------------------------",
      "Sentence:         " ++ show words,
      "Actions:          " ++ show actions,
      "Prediction:       " ++ show prediction,
      "Valid or Invalid: " ++ if isValid then "Valid" else "Invalid",
      "Right or Wrong:   " ++ if isCorrect then "Right" else "Wrong"
    ] ++ [
      "Correct tree:\n" ++ show correctTree | isValid
    ] ++ if not isCorrect && isValid 
          then [
                  "Predicted Tree:\n" ++ show predictedTree,
                  "Reason for invalid CCG:\n" ++ case ccgError of
                                                    Left str -> str
                                                    Right _ -> "Valid"
                ]
          else []


{-

main functions

-}

training ::
  (Optimizer o) =>
  Mode ->
  TrainingConfig ->
  (RNNG, o) ->
  IndexData ->
  -- | trainingData, validationData
  ([RNNGSentence], [RNNGSentence]) ->
  IO (RNNG, [[Float]])
training mode@Mode{..} TrainingConfig{..} (rnng, optim) IndexData {..} (trainingData, validationData) = do
  existCheckpointDir <- doesDirectoryExist checkpointDirectory 
  unless existCheckpointDir $ createDirectory checkpointDirectory
  ((trained, _), losses) <- mapAccumM [1..iter] (rnng, (optim, optim, optim)) epochStep
  return (trained, losses)
  where
    checkpointDirectory = "models/" ++ modelName ++ "-checkpoints"
    batches = makeBatch validationStep trainingData
    epochStep :: (Optimizer o) => Int -> (RNNG, (o, o, o)) -> IO ((RNNG, (o, o, o)), [Float])
    epochStep epoch model = do
      trained@((trainedModel, _), losses) <- mapAccumM [1..(length batches)] model batchStep
      -- | 1stepごとにcheckpointとしてモデル保存
      -- | TODO::　optimizerも保存
      Torch.Train.saveParams trainedModel (checkpointDirectory ++ "/epoch-" ++ show epoch)
      return trained
    -- | TODO:: Batch処理にする
    batchStep :: (Optimizer o) => Int -> (RNNG, (o, o, o)) -> IO ((RNNG, (o, o, o)), Float)
    batchStep idx (batchRNNG, batchOpts) = do
      let batch = batches !! (idx - 1)
      ((batchRNNG', batchOpts'), batchLoss) <- mapAccumM batch (batchRNNG, batchOpts) step
      
      let trainingLoss = sum batchLoss / (fromIntegral (length batch)::Float)
      putStrLn $ "#" ++ show idx 
      putStrLn $ "Training Loss: " ++ show trainingLoss

      (validationLoss, validationPrediction) <- evaluate mode batchRNNG' IndexData {..}  validationData
      putStrLn $ "Validation Loss(To not be any help): " ++ show validationLoss
      sampledData <- sampleRandomData 5 (zip validationData validationPrediction)
      putStr $ unlines $ map (uncurry showResult) sampledData
      putStrLn "======================================"
      return ((batchRNNG', batchOpts'), validationLoss)
    step :: (Optimizer o) => RNNGSentence -> (RNNG, (o, o, o)) -> IO ((RNNG, (o, o, o)), Float)
    step rnngSentence@(RNNGSentence (_, actions)) (stepRNNG@(RNNG actionPredictRNNG parseRNNG compRNNG), (stepOpt1, stepOpt2, stepOpt3)) = do
      let answer = toDevice device $ asTensor $ fmap actionIndexFor actions
          output = rnngForward mode stepRNNG IndexData {..} rnngSentence
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
  Mode ->
  RNNG ->
  IndexData ->
  [RNNGSentence] ->
  -- | (loss, predicted action sequence)
  IO (Float, [[Action]])
evaluate mode@Mode{..} rnng IndexData {..} rnngSentences = do
  (_, result) <- mapAccumM rnngSentences rnng step
  let (losses, predictions) = unzip result
      size = fromIntegral (length rnngSentences)::Float
  return ((sum losses) / size, reverse predictions)
  where
    step :: RNNGSentence -> RNNG -> IO (RNNG, (Float, [Action]))
    step rnngSentence@(RNNGSentence (_, actions)) rnng' = do
      let answer = toDevice device $ asTensor $ fmap actionIndexFor actions
          output = rnngForward mode rnng IndexData {..} rnngSentence
          predictionTensor = Torch.stack (Dim 0) $ fmap (argmax (Dim 0) RemoveDim) output
          prediction = fmap indexActionFor (asValue predictionTensor::[Int])
          -- | lossの計算は答えと推論結果の系列長が同じ時だけ
          loss = if length prediction == length prediction
                  then nllLoss' answer (Torch.stack (Dim 0) output)
                  else asTensor (0::Float)
      return (rnng', (asValue loss::Float, prediction))


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
  modelName <- modelNameConfig True config
  let modelFilePath = "models/" ++ modelName
      modelSpecPath = "models/" ++ modelName ++ "-spec"
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
  let (wordIndexFor, indexWordFor, wordEmbDim) = indexFactory (buildVocab dataForTraining 0 toWordList) (T.pack "unk") Nothing
      (actionIndexFor, indexActionFor, actionEmbDim) = indexFactory (buildVocab dataForTraining 0 toActionList) ERROR Nothing
      (ntIndexFor, indexNTFor, ntEmbDim) = indexFactory (buildVocab dataForTraining 0 toNTList) (T.pack "unk") Nothing
      indexData = IndexData wordIndexFor indexWordFor actionIndexFor indexActionFor ntIndexFor indexNTFor
  putStrLn $ "WordEmbDim: " ++ show wordEmbDim
  putStrLn $ "ActionEmbDim: " ++ show actionEmbDim
  putStrLn $ "NTEmbDim: " ++ show ntEmbDim
  putStrLn "======================================"

  when (mode == "Train") $ do
    -- model spec
    let wordEmbedSize = fromIntegral (wordEmbedSizeConfig config)::Int
        actionEmbedSize = fromIntegral (actionEmbedSizeConfig config)::Int
        hiddenSize = fromIntegral (hiddenSizeConfig config)::Int
        numLayer = fromIntegral (numOfLayerConfig config)::Int
        device = Device CUDA 0
        rnngSpec = RNNGSpec device posMode wordEmbedSize actionEmbedSize wordEmbDim actionEmbDim ntEmbDim hiddenSize
    initRNNGModel <- toDevice device <$> sample rnngSpec
    -- | spec保存
    B.encodeFile modelSpecPath rnngSpec

    -- | training
    let trainingConfig = TrainingConfig {
                           iter = fromIntegral (epochConfig config)::Int,
                           learningRate = toDevice device $ asTensor (learningRateConfig config),
                           validationStep = fromIntegral (validationStepConfig config)::Int,
                           modelName = modelName
                         }
        optim = GD
        mode = Mode {
                      device = device,
                      parsingMode = Point,
                      posMode = posMode
                    }
    (trained, losses) <- training mode trainingConfig (initRNNGModel, optim) indexData (trainingData, validationData)

    -- | model保存
    Torch.Train.saveParams trained modelFilePath
    drawLearningCurve ("imgs/" ++ modelName ++ ".png") "Learning Curve" [("", reverse $ concat losses)]


  {-

    evaluation

  -}
  -- | model読み込み
  rnngSpec <- B.decodeFile modelSpecPath::(IO RNNGSpec)
  rnngModel <- Torch.Train.loadParams rnngSpec modelFilePath
  let answers = fmap (\(RNNGSentence (_, actions)) -> actions) evaluationData

  -- | training dataにないactionとその頻度
  putStrLn "Unknown Actions and their frequency: "
  print $ counts $ unknownActions actionIndexFor $ toActionList evaluationData
  
  -- | trainingデータの低頻度ラベル
  putStr "Num of ow frequency (<= 10) label: "
  print $ length $ takeWhile (\x -> snd x <= 10) $ sortOn snd $ counts $ toNTList trainingData

  let mode = Mode {
                    device = modelDevice rnngSpec,
                    parsingMode = parsingMode,
                    posMode = modelPosMode rnngSpec
                  }
  (_, evaluationPrediction) <- evaluate mode rnngModel indexData evaluationData

  -- | ちゃんと木になってる予測を抜き出してくる
  let predictedRNNGSentences = zipWith (curry insertDifferentActions) evaluationData evaluationPrediction
      predictionTrees = fromRNNGSentences predictedRNNGSentences
      correctTrees = fromRNNGSentences evaluationData
      validTreeMask = map (not . isErr) predictionTrees
  putStr "Num of Valid Trees: "
  print $ length $ filterByMask predictionTrees validTreeMask

  -- | CCGの時はちゃんとCCGになってる木を取り出してくる
  let checkedValidCCG = map checkValidCCG predictionTrees
      validCCGMask = map isRight checkedValidCCG
  putStr "Num of Valid CCGtrees: "
  print $ length $ filterByMask predictionTrees validCCGMask

  -- | 全て正解の文を抜き出してくる
  let correctAnswerMask = zipWith ((checkCorrect .) . zip) answers evaluationPrediction
      correctAnswers = filterByMask evaluationData correctAnswerMask
  putStr "Num of Correct Answers: "
  print $ length correctAnswers
  putStrLn "Correct Answers: "
  print correctAnswers

  -- | 分析結果を出力
  writeFile ("reports/" ++ modelName ++ "-" ++ show parsingMode ++ "-result.txt") $ unlines $ zipWith4 reportResult validTreeMask correctAnswerMask (zip evaluationData correctTrees) (zip3 evaluationPrediction predictionTrees checkedValidCCG)
  T.writeFile ("reports/" ++ modelName ++ "-" ++ show parsingMode ++ "-classification.txt") $ classificationReport evaluationPrediction answers
