{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE GADTs #-}


module RNNG where
import PTB
import Util
import Torch hiding (foldLoop)
-- | hasktorch-tools
import Torch.Layer.RNN (RNNHypParams(..), RNNParams, rnnLayer)
import Torch.Layer.BiLSTM (BiLstmHypParams(..), BiLstmParams, biLstmLayers)
import Torch.Layer.LSTM (LstmHypParams(..), LstmParams, lstmLayer)
import Torch.Layer.Linear
import Torch.Control (mapAccumM, foldLoop)
import Torch.Train (update, saveParams, loadParams)
-- | nlp-tools
import ML.Exp.Chart (drawLearningCurve)

import GHC.Generics
import qualified Data.Text as T 
import Data.List
import Debug.Trace



data RNNGSpec = RNNGSpec {
  modelDevice :: Device,
  wordEmbedSize :: Int,
  actionEmbedSize :: Int,
  wordNumEmbed :: Int,
  actionNumEmbed :: Int,
  hiddenSize :: Int
} deriving (Show, Eq)

data RNNG where
  RNNG ::
    {
      wordEmbedding :: Parameter,
      -- ntEmbedding :: Parameter,
      -- actionEmbedding :: Parameter,
      bufferRNN :: RNNParams,
      actionRNN :: RNNParams,
      stackLSTM :: LstmParams,
      h0 :: Parameter,
      stackh0 :: Parameter,
      stackc0 :: Parameter,
      w :: Parameter,
      c :: Parameter,
      linearParams :: LinearParams,
      actionStart :: Parameter, -- dummy for empty action history
      bufferGuard :: Parameter, -- dummy for empty buffer
      stackGuard :: Parameter  -- dummy for empty stack
    } ->
    RNNG
  deriving (Show, Generic, Parameterized)

instance
  Randomizable
    RNNGSpec
    RNNG
  where
    sample RNNGSpec {..} =
      RNNG
        <$> (makeIndependent =<< randnIO' [wordNumEmbed, wordEmbedSize])
        <*> sample (RNNHypParams modelDevice wordEmbedSize hiddenSize)
        <*> sample (RNNHypParams modelDevice wordEmbedSize hiddenSize)
        <*> sample (LstmHypParams modelDevice hiddenSize)
        <*> (makeIndependent =<< randnIO' [hiddenSize])
        <*> (makeIndependent =<< randnIO' [hiddenSize])
        <*> (makeIndependent =<< randnIO' [hiddenSize])
        <*> (makeIndependent =<< randnIO' [hiddenSize * 3, hiddenSize])
        <*> (makeIndependent =<< randnIO' [hiddenSize])
        <*> sample (LinearHypParams modelDevice hiddenSize actionNumEmbed)
        <*> (makeIndependent =<< randnIO' [actionEmbedSize])
        <*> (makeIndependent =<< randnIO' [wordEmbedSize])
        <*> (makeIndependent =<< randnIO' [wordEmbedSize])

data RNNGParseSpec = RNNGParseSpec {
  modelDevice2 :: Device,
  wordEmbedSize2 :: Int,
  ntNumEmbed2 :: Int,
  actionNumEmbed2 :: Int,
  hiddenSize2 :: Int
} deriving (Show, Eq)

data RNNGParse where
  RNNGParse ::
    {
      ntEmbedding :: Parameter,
      actionEmbedding :: Parameter,
      compLSTM :: BiLstmParams,
      comph0 :: Parameter,
      compc0 :: Parameter
    } ->
    RNNGParse
  deriving (Show, Generic, Parameterized)

instance
  Randomizable
    RNNGParseSpec
    RNNGParse
  where
    sample RNNGParseSpec {..} =
      RNNGParse
        <$> (makeIndependent =<< randnIO' [ntNumEmbed2, wordEmbedSize2])
        <*> (makeIndependent =<< randnIO' [actionNumEmbed2, wordEmbedSize2])
        <*> sample (BiLstmHypParams modelDevice2 1 hiddenSize2)
        <*> (makeIndependent =<< randnIO' [hiddenSize2])
        <*> (makeIndependent =<< randnIO' [hiddenSize2])


data IndexData = IndexData {
    wordIndexFor :: T.Text -> Int,
    indexWordFor :: Int -> T.Text,
    actionIndexFor :: Action -> Int,
    indexActionFor :: Int -> Action,
    ntIndexFor :: T.Text -> Int,
    indexNTFor :: Int -> T.Text
  }

data RNNGState where
  RNNGState ::
    {
      stack :: [Tensor],
      textStack :: [T.Text],
      buffer :: [Tensor],
      textBuffer :: [T.Text],
      actionHistory :: [Tensor],
      textActionHistory :: [Action]
    } ->
    RNNGState 
  deriving (Show)

predictNextAction ::
  RNNG ->
  -- | data
  RNNGState ->
  -- | possibility of actions
  Tensor
predictNextAction RNNG {..} RNNGState {..} = 
  let (_, b_embedding) = rnnLayer bufferRNN (toDependent $ h0) buffer
      s_embedding = snd $ last $ lstmLayer stackLSTM (toDependent $ stackc0, toDependent $ stackh0) stack
      (_, a_embedding) = rnnLayer actionRNN (toDependent $ h0) actionHistory
      ut = Torch.tanh $ ((cat (Dim 0) [s_embedding, b_embedding, a_embedding]) `matmul` (toDependent w) + (toDependent c))
  in logSoftmax (Dim 0) $ linearLayer linearParams ut

push ::
  [a] ->
  a ->
  [a]
push list d = list ++ [d]


parse ::
  -- | model
  RNNGParse ->
  IndexData ->
  -- | new RNNGState
  RNNGState ->
  Action ->
  RNNGState
parse RNNGParse {..} IndexData {..} RNNGState {..} (NT label) =
  let nt_embedding = embedding' (toDependent ntEmbedding) ((asTensor . ntIndexFor) label)
      textAction = NT label
      action_embedding = embedding' (toDependent actionEmbedding) ((asTensor . actionIndexFor) textAction)
  in RNNGState {
      stack = push stack nt_embedding,
      textStack = push textStack ((T.pack "(") `T.append` label),
      buffer = buffer,
      textBuffer = textBuffer,
      actionHistory = push actionHistory action_embedding,
      textActionHistory = push textActionHistory  textAction
    }
parse RNNGParse {..} IndexData {..} RNNGState {..} SHIFT =
  let textAction = SHIFT
      action_embedding = embedding' (toDependent actionEmbedding) ((asTensor . actionIndexFor) textAction)
  in RNNGState {
      stack = push stack (head buffer),
      textStack = push textStack (head textBuffer),
      buffer = tail buffer,
      textBuffer = tail textBuffer,
      actionHistory = push actionHistory action_embedding,
      textActionHistory = push textActionHistory textAction
    }
parse RNNGParse {..} IndexData {..} RNNGState {..} REDUCE =
  let textAction = REDUCE
      action_embedding = embedding' (toDependent actionEmbedding) ((asTensor . actionIndexFor) textAction)
      (Just idx) = findIndex (\elem -> (T.isPrefixOf (T.pack "(") elem) && not (T.isSuffixOf (T.pack ")") elem)) (reverse textStack)
      splitedIdx = (length textStack) - idx - 1
      (newTextStack, textSubTree) = splitAt splitedIdx textStack
      (newStack, subTree) = splitAt splitedIdx stack
      composedSubTree = fst $ last $ biLstmLayers compLSTM (toDependent $ compc0, toDependent $ comph0) subTree
  in RNNGState {
      stack = push newStack composedSubTree,
      textStack = push newTextStack (T.intercalate (T.pack " ") (push textSubTree (T.pack ")"))),
      buffer = buffer,
      textBuffer = textBuffer,
      actionHistory = push actionHistory action_embedding,
      textActionHistory = push textActionHistory textAction
    }


predict ::
  RuntimeMode ->
  (RNNG, RNNGParse) ->
  IndexData ->
  -- | actions
  [Action] ->
  -- | predicted history
  [Tensor] ->
  -- | init RNNGState
  RNNGState ->
  -- | (predicted actions, new rnngState)
  ([Tensor], RNNGState)
predict Train _ _ [] results rnngState = (reverse results, rnngState)
predict Train (rnng, rnngParse) indexData (action:rest) predictedHitory rnngState =
  let predicted = predictNextAction rnng rnngState
      newRNNGState = parse rnngParse indexData rnngState action
  in predict Train (rnng, rnngParse) indexData rest (predicted:predictedHitory) newRNNGState

predict Eval (rnng, rnngParse) IndexData {..} _ predictedHitory RNNGState {..} =
  if ((length textStack) == 1) && ((length textBuffer) == 0)
    then (reverse predictedHitory, RNNGState {..} )
  else
    let predicted = predictNextAction rnng RNNGState {..}
        action = indexActionFor (asValue (argmax (Dim 0) RemoveDim predicted)::Int)
    -- 不正なactionを予測した場合
    -- TODO: valid　actionのみに絞る
    in if (action == REDUCE && (length textStack == 1)) || (action == SHIFT && (length textBuffer) == 0)
         then (reverse predictedHitory, RNNGState {..} )
       else
         let newRNNGState = parse rnngParse IndexData {..} RNNGState {..} action
         in predict Eval (rnng, rnngParse) IndexData {..}  [] (predicted:predictedHitory) newRNNGState

rnngForward ::
  RuntimeMode ->
  (RNNG, RNNGParse) ->
  -- | functions to convert text to index
  IndexData ->
  RNNGSentence ->
  [Tensor]
rnngForward runTimeMode (RNNG {..}, rnngParse) IndexData {..} (RNNGSentence (sents, actions)) =
  let sentsTensor = fmap ((embedding' (toDependent wordEmbedding)) . asTensor . wordIndexFor) sents
      initRNNGState = RNNGState {
        stack = [toDependent stackGuard],
        textStack = [],
        buffer = sentsTensor ++ [toDependent bufferGuard],
        textBuffer = sents,
        actionHistory = [toDependent actionStart],
        textActionHistory = []
      }
  in fst $ predict runTimeMode (RNNG {..}, rnngParse) IndexData {..} actions [] initRNNGState

evaluate ::
  (RNNG, RNNGParse) ->
  IndexData ->
  [RNNGSentence] ->
  -- | action answers
  [[Action]] ->
  -- | (accuracy, loss, predicted action sequence)
  ([Float], Float, [[Action]])
evaluate (rnng, rnngParse) IndexData {..} batches answers =
  let (acc, loss, ans) = foldLoop (zip batches answers) ([], 0, []::[[Action]]) $ 
        \(batch, answerActions) (acc', loss', ans') ->
          let answer = asTensor $ fmap actionIndexFor answerActions
              output = rnngForward Eval (rnng, rnngParse) IndexData {..} batch
              predicted = Torch.stack (Dim 0) $ fmap (argmax (Dim 0) RemoveDim) output 
              predictedSize = shape predicted !! 0
              answerSize = shape answer !! 0
              alignedOutput = if predictedSize < answerSize
                                  then output ++ replicate (answerSize - predictedSize) (zeros' [shape (head output) !! 0])
                                  else output 
              alignedPredicted = if predictedSize < answerSize
                                  then cat (Dim 0) [predicted, asTensor ((replicate (answerSize - predictedSize) (-1))::[Int])]
                                  else predicted
              alignedAnswer = if predictedSize > answerSize
                                then cat (Dim 0) [answer, asTensor ((replicate (predictedSize - answerSize) (-1))::[Int])]
                                else answer
              numCorrect = asValue (sumAll $ eq alignedAnswer alignedPredicted)::Int
              accuracy = (fromIntegral numCorrect::Float) / fromIntegral (Prelude.max answerSize predictedSize)::Float
              batchLoss = asValue (nllLoss' alignedAnswer (Torch.stack (Dim 0) alignedOutput))::Float
              predictedActions = fmap indexActionFor (asValue predicted::[Int])
          in (accuracy:acc', loss' + (Debug.Trace.traceShow batchLoss batchLoss), predictedActions:ans')
      size = fromIntegral (length batches)::Float
  in (acc, loss/size, ans)


main :: IO()
main = do
  -- experiment setting
  config <- configLoad
  let iter = fromIntegral (getEpoch config)::Int
      wordEmbedSize = fromIntegral (getWordEmbedSize config)::Int
      actionEmbedSize = fromIntegral (getActionEmbedSize config)::Int
      hiddenSize = fromIntegral (getHiddenSize config)::Int
      numLayer = fromIntegral (getNumLayer config)::Int
      modelFilePath = getModelFilePath config
      graphFilePath = getGraphFilePath config
      device = Device CPU 0
      optim = GD
  -- data
  trainingData <- loadActionsFromBinary $ getTrainingDataPath config
  validationData <- loadActionsFromBinary $ getValidationDataPath config
  evaluationData <- loadActionsFromBinary $ getEvaluationDataPath config

  let dataForTraining = trainingData

  let (wordIndexFor, indexWordFor, wordEmbDim) = indexFactory (buildVocab dataForTraining 3 toWordList) (T.pack "unk") Nothing
      (actionIndexFor, indexActionFor, actionEmbDim) = indexFactory (buildVocab dataForTraining 1 toActionList) ERROR Nothing
      (ntIndexFor, indexNTFor, ntEmbDim) = indexFactory (buildVocab dataForTraining 1 toNTList) (T.pack "unk") Nothing
      indexData = IndexData wordIndexFor indexWordFor actionIndexFor indexActionFor ntIndexFor indexNTFor

  let batches = Data.List.take 1 dataForTraining
      rnngSpec = RNNGSpec device wordEmbedSize actionEmbedSize wordEmbDim actionEmbDim hiddenSize
      rnngParseSpec = RNNGParseSpec device wordEmbedSize ntEmbDim actionEmbDim hiddenSize
  initRNNGModel <- sample $ rnngSpec
  initRNNGParseModel <- sample $ rnngParseSpec

  -- | training
  (trained, losses) <- mapAccumM [1..iter] ((initRNNGModel, optim), (initRNNGParseModel, optim)) $ 
    \epoc ((rnng, opt1), (rnngParse, opt2)) -> do
      (updated, batchLosses) <- mapAccumM batches ((rnng, opt1), (rnngParse, opt2)) $ 
        \batch ((rnng', opt1'), (rnngParse', opt2')) -> do
          let RNNGSentence (sents, actions) = batch
          print actions
          print sents
          let answer = asTensor $ fmap actionIndexFor actions
              output = rnngForward Train (rnng', rnngParse') indexData (RNNGSentence (sents, actions))
              loss = nllLoss' answer (Torch.stack (Dim 0) output)
          updatedRNNG <- runStep rnng' opt1' loss 1e-2
          updatedRNNGParse <- runStep rnngParse' opt2' loss 1e-2
          return ((updatedRNNG, updatedRNNGParse), (asValue loss::Float))

      let loss = sum batchLosses / (fromIntegral (length batches)::Float)
      return (updated, loss)

  -- | model保存
  let ((rnngModel, _), (rnngParseModel, _)) = trained
  Torch.Train.saveParams rnngModel (modelFilePath ++ "-rnng")
  Torch.Train.saveParams rnngParseModel (modelFilePath ++ "-rnng-parse")
  drawLearningCurve graphFilePath "Learning Curve" [("", reverse losses)]
  print losses
  
  -- | model読み込み
  rnngModel <- Torch.Train.loadParams rnngSpec (modelFilePath ++ "-rnng")
  rnngParseModel <- Torch.Train.loadParams rnngParseSpec (modelFilePath ++ "-rnng-parse")

  -- let RNNGSentence (s, actions) = head evaluationData
  -- print wordEmbDim
  -- print $ shape $ toDependent $ wordEmbedding rnngModel
  -- print $ fmap (embedding' (toDependent (wordEmbedding rnngModel)) . asTensor . wordIndexFor) s

  let answers = fmap (\(RNNGSentence (_, actions)) -> actions) evaluationData
      (acc, loss, predicted) = evaluate (rnngModel, rnngParseModel) indexData evaluationData answers
  
  print $ "acc: " ++ show (sum acc / (fromIntegral (length acc)::Float))
  print $ "loss: " ++ show loss
  return ()
