{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE BlockArguments #-}


module RNNG where
import Model.RNNG
import Data.RNNGSentence
import Util
import Torch hiding (foldLoop, take, repeat)
-- | hasktorch-tools
import Torch.Layer.RNN (RNNParams, rnnLayer)
import Torch.Layer.BiLSTM (BiLstmParams, biLstmLayers)
import Torch.Layer.LSTM (LstmParams, lstmLayer)
import Torch.Layer.Linear (linearLayer)
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

{-
mask Forbidden actions
-}

checkNTForbidden ::
  RNNGState ->
  Bool
checkNTForbidden RNNGState {..} =
  numOpenParen > 100 -- 開いているNTが多すぎる時
  || length textBuffer == 0 -- 単語が残っていない時
  
checkREDUCEForbidden ::
  RNNGState ->
  Bool
checkREDUCEForbidden RNNGState {..} =
  length textStack < 2  -- first action must be NT and don't predict POS
  || (numOpenParen == 1 && length textBuffer > 0) -- bufferに単語が残ってるのに木を一つにまとめることはできない
  || ((previousAction /= SHIFT) && (previousAction /= REDUCE)) -- can't REDUCE after NT
  where 
    previousAction = if length textActionHistory > 0
                      then head textActionHistory
                      else ERROR

checkSHIFTForbidden ::
  RNNGState ->
  Bool
checkSHIFTForbidden RNNGState{..} =
  length textStack == 0 -- first action must be NT
  || length textBuffer == 0 -- Buffer isn't empty

maskTensor :: 
  Device ->
  RNNGState ->
  [Action] ->
  Tensor
maskTensor device rnngState actions = toDevice device $ asTensor $ map mask actions
  where 
    ntForbidden = checkNTForbidden rnngState
    reduceForbidden = checkREDUCEForbidden rnngState
    shiftForbidden = checkSHIFTForbidden rnngState
    mask :: Action -> Bool
    mask ERROR = True
    mask (NT _) = ntForbidden
    mask REDUCE = reduceForbidden
    mask SHIFT = shiftForbidden

maskImpossibleAction ::
  Device ->
  Tensor ->
  RNNGState ->
  IndexData ->
  Tensor
maskImpossibleAction device prediction rnngState IndexData {..}  =
  let actions = map indexActionFor [0..((shape prediction !! 0) - 1)]
      boolTensor = maskTensor device rnngState actions
  in maskedFill prediction boolTensor (1e-38::Float)


{-
  RNNG Forward
-}

predictNextAction ::
  Device -> 
  RNNG ->
  -- | data
  RNNGState ->
  IndexData ->
  -- | possibility of actions
  Tensor
predictNextAction device (RNNG PredictActionRNNG {..} _ _) RNNGState {..} indexData = 
  let (_, bufferEmbedding) = rnnLayer bufferRNN (toDependent $ bufferh0) buffer
      stackEmbedding = snd $ head hiddenStack
      actionEmbedding = hiddenActionHistory
      ut = Torch.tanh $ ((cat (Dim 0) [stackEmbedding, bufferEmbedding, actionEmbedding]) `matmul` (toDependent w) + (toDependent c))
      actionLogit = linearLayer linearParams ut
      maskedAction = maskImpossibleAction device actionLogit RNNGState {..} indexData
  in logSoftmax (Dim 0) $ maskedAction


stackLSTMForward :: 
  LstmParams ->
  -- | [(ci, hi)]
  [(Tensor, Tensor)] ->
  -- | new Element
  Tensor ->
  -- | (cn, hn)
  (Tensor, Tensor)
stackLSTMForward stackLSTM stack newElem = last $ lstmLayer stackLSTM (head stack) [newElem]

actionRNNForward ::
  RNNParams ->
  -- | h_n
  Tensor ->
  -- | new Element
  Tensor ->
  -- | h_n+1
  Tensor
actionRNNForward actionRNN hn newElem = snd $ rnnLayer actionRNN hn [newElem]

parse ::
  Device ->
  -- | model
  RNNG ->
  IndexData ->
  -- | new RNNGState
  RNNGState ->
  Action ->
  RNNGState
parse device (RNNG _ ParseRNNG {..} _) IndexData {..} RNNGState {..} (NT label) =
  let nt_embedding = embedding' (toDependent ntEmbedding) ((toDevice device . asTensor . ntIndexFor) label)
      textAction = NT label
      action_embedding = embedding' (toDependent actionEmbedding) ((toDevice device . asTensor . actionIndexFor) textAction)
  in RNNGState {
      stack = nt_embedding:stack,
      textStack = ((T.pack "<") `T.append` label):textStack,
      hiddenStack = (stackLSTMForward stackLSTM hiddenStack nt_embedding):hiddenStack,
      buffer = buffer,
      textBuffer = textBuffer,
      actionHistory = action_embedding:actionHistory,
      textActionHistory = textAction:textActionHistory,
      hiddenActionHistory = actionRNNForward actionRNN hiddenActionHistory action_embedding,
      numOpenParen = numOpenParen + 1
    }
parse device (RNNG _ ParseRNNG {..} _) IndexData {..} RNNGState {..} SHIFT =
  let textAction = SHIFT
      action_embedding = embedding' (toDependent actionEmbedding) ((toDevice device . asTensor . actionIndexFor) textAction)
  in RNNGState {
      stack = (head buffer):stack,
      textStack = (head textBuffer):textStack,
      hiddenStack = (stackLSTMForward stackLSTM hiddenStack (head buffer)):hiddenStack,
      buffer = tail buffer,
      textBuffer = tail textBuffer,
      actionHistory = action_embedding:actionHistory,
      textActionHistory = textAction:textActionHistory,
      hiddenActionHistory = actionRNNForward actionRNN hiddenActionHistory action_embedding,
      numOpenParen = numOpenParen
    }
parse device (RNNG _ ParseRNNG {..} CompRNNG {..}) IndexData {..} RNNGState {..} REDUCE =
  let textAction = REDUCE
      action_embedding = embedding' (toDependent actionEmbedding) ((toDevice device . asTensor . actionIndexFor) textAction)
      -- | 開いたlabelのidxを特定する
      (Just idx) = findIndex (\elem -> (T.isPrefixOf (T.pack "<") elem) && not (T.isSuffixOf (T.pack ">") elem)) textStack
      -- | popする
      (textSubTree, newTextStack) = splitAt (idx + 1) textStack
      (subTree, newStack) = splitAt (idx + 1) stack
      (_, newHiddenStack) = splitAt (idx + 1) hiddenStack
      -- composeする
      composedSubTree = snd $ last $ biLstmLayers compLSTM (toDependent $ compc0, toDependent $ comph0) (reverse subTree)
  in RNNGState {
      stack = composedSubTree:newStack,
      textStack = (T.intercalate (T.pack " ") (reverse $ (T.pack ">"):textSubTree)):newTextStack,
      hiddenStack = (stackLSTMForward stackLSTM newHiddenStack composedSubTree):newHiddenStack,
      buffer = buffer,
      textBuffer = textBuffer,
      actionHistory = action_embedding:actionHistory,
      textActionHistory = textAction:textActionHistory,
      hiddenActionHistory = actionRNNForward actionRNN hiddenActionHistory action_embedding,
      numOpenParen = numOpenParen - 1
    }

predict ::
  Device ->
  RuntimeMode ->
  RNNG ->
  IndexData ->
  -- | actions
  [Action] ->
  -- | predicted history
  [Tensor] ->
  -- | init RNNGState
  RNNGState ->
  -- | (predicted actions, new rnngState)
  ([Tensor], RNNGState)
predict device Train _ _ [] results rnngState = (reverse results, rnngState)
predict device Train rnng indexData (action:rest) predictionHitory rnngState =
  let prediction = predictNextAction device rnng rnngState indexData
      newRNNGState = parse device rnng indexData rnngState action
  in predict device Train rnng indexData rest (prediction:predictionHitory) newRNNGState

predict device Eval rnng IndexData {..} _ predictionHitory RNNGState {..} =
  if ((length textStack) == 1) && ((length textBuffer) == 0)
    then (reverse predictionHitory, RNNGState {..} )
  else
    let prediction = predictNextAction device rnng RNNGState {..} IndexData {..}
        action = indexActionFor (asValue (argmax (Dim 0) RemoveDim prediction)::Int)
        newRNNGState = parse device rnng IndexData {..} RNNGState {..} action
    in predict device Eval rnng IndexData {..}  [] (prediction:predictionHitory) newRNNGState


rnngForward ::
  Device ->
  RuntimeMode ->
  RNNG ->
  -- | functions to convert text to index
  IndexData ->
  RNNGSentence ->
  [Tensor]
rnngForward device runTimeMode (RNNG predictActionRNNG ParseRNNG {..} compRNNG) IndexData {..} (RNNGSentence (sents, actions)) =
  let sentsTensor = fmap ((embedding' (toDependent wordEmbedding)) . toDevice device . asTensor . wordIndexFor) sents
      initRNNGState = RNNGState {
        stack = [toDependent stackGuard],
        textStack = [],
        hiddenStack = [stackLSTMForward stackLSTM [(toDependent stackh0, toDependent stackc0)] (toDependent stackGuard)],
        buffer = sentsTensor ++ [toDependent bufferGuard],
        textBuffer = sents,
        actionHistory = [toDependent actionStart],
        textActionHistory = [],
        hiddenActionHistory = actionRNNForward actionRNN (toDependent $ actionh0) (toDependent actionStart),
        numOpenParen = 0
      }
  in fst $ predict device runTimeMode (RNNG predictActionRNNG ParseRNNG {..} compRNNG) IndexData {..} actions [] initRNNGState



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

evaluate ::
  Device ->
  RNNG ->
  IndexData ->
  [RNNGSentence] ->
  -- | (loss, predicted action sequence)
  IO (Float, [[Action]])
evaluate device rnng IndexData {..} batches = do
  (_, result) <- mapAccumM batches rnng step
  let (losses, predictions) = unzip result
      size = fromIntegral (length batches)::Float
  return ((sum losses) / size, reverse predictions)
  where
    step :: RNNGSentence -> RNNG -> IO (RNNG, (Float, [Action]))
    step batch rnng' = do
      let RNNGSentence (_, actions) = batch
          answer = toDevice device $ asTensor $ fmap actionIndexFor actions
          output = rnngForward device Train rnng IndexData {..} batch
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
      iter = fromIntegral (epochConfig config)::Int
      wordEmbedSize = fromIntegral (wordEmbedSizeConfig config)::Int
      actionEmbedSize = fromIntegral (actionEmbedSizeConfig config)::Int
      hiddenSize = fromIntegral (hiddenSizeConfig config)::Int
      numLayer = fromIntegral (numOfLayerConfig config)::Int
      device = Device CUDA 0
      learningRate = toDevice device $ asTensor (learningRateConfig config)
      modelName = modelNameConfig config
      modelFilePath = "models/" ++ modelName
      modelSpecPath = "models/" ++ modelName ++ "-spec"
      graphFilePath = "imgs/" ++ modelName ++ ".png"
      reportFilePath = "reports/" ++ modelName ++ ".txt"
      validationStep = fromIntegral (validationStepConfig config)::Int
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
    ((trained, _), losses) <- mapAccumM [1..iter] (initRNNGModel, (optim, optim, optim)) $ 
      \epoch (rnng, (opt1, opt2, opt3)) -> do
        let batches = take 2 $ makeBatch validationStep dataForTraining

        ((updated, opts), batchLosses) <- mapAccumM [1..(length batches)] (rnng, (opt1, opt2, opt3)) $
          -- | batch処理はできていないが、便宜上
          \index (rnng', (opt1', opt2', opt3')) -> do
            let batch = batches !! (index-1)
            ((updated', opts'), batchLoss) <- mapAccumM batch (rnng', (opt1', opt2', opt3')) $ 
              \rnngSentence (rnng'', (opt1'', opt2'', opt3'')) -> do
                let RNNGSentence (sents, actions) = rnngSentence
                    answer = toDevice device $ asTensor $ fmap actionIndexFor actions
                    output = rnngForward device Train rnng'' indexData (RNNGSentence (sents, actions))
                dropoutOutput <- forM output (dropout 0.2 True) 
                let loss = nllLoss' answer (Torch.stack (Dim 0) dropoutOutput)
                    RNNG actionPredictRNNG parseRNNG compRNNG = rnng''
                -- | パラメータ更新
                (updatedActionPredictRNNG, opt1''') <- runStep actionPredictRNNG opt1'' loss learningRate
                (updatedParseRNNG, opt2''') <- runStep parseRNNG opt2'' loss learningRate
                (updatedCompRNNG, opt3''') <- if length (filter (== REDUCE) actions) > 1
                                              then runStep compRNNG opt3'' loss learningRate
                                            else return (compRNNG, opt3'')
                return ((RNNG updatedActionPredictRNNG updatedParseRNNG updatedCompRNNG, (opt1''', opt2''', opt3''')), (asValue loss::Float))

            let trainingLoss = sum batchLoss / (fromIntegral (length batch)::Float)
            putStrLn $ "#" ++ show index 
            putStrLn $ "Training Loss: " ++ show trainingLoss

            (validationLoss, validationPrediction) <- evaluate device updated' indexData validationData
            putStrLn $ "Validation Loss(To not be any help): " ++ show validationLoss
            sampleRandomData 5 (zip validationData validationPrediction) >>= printResult  
            putStrLn "======================================"
            return ((updated', opts'), validationLoss)
        return ((updated, opts), batchLosses)

    -- | model保存
    B.encodeFile modelSpecPath rnngSpec
    Torch.Train.saveParams trained modelFilePath
    drawLearningCurve graphFilePath "Learning Curve" [("", reverse $ concat losses)]

  -- | evaluation
  -- | model読み込み
  rnngSpec <- (B.decodeFile modelSpecPath)::(IO RNNGSpec)
  print rnngSpec
  rnngModel <- Torch.Train.loadParams rnngSpec modelFilePath

  (_, evaluationPrediction) <- evaluate device rnngModel indexData evaluationData
  let answers = fmap (\(RNNGSentence (_, actions)) -> actions) evaluationData
  T.writeFile reportFilePath $ classificationReport evaluationPrediction answers
  sampleRandomData 10 (zip evaluationData evaluationPrediction) >>= printResult
  return ()
