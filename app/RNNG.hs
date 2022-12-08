{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE GADTs #-}


module RNNG where
import PTB
import Util
import Torch
-- | hasktorch-tools
import Torch.Layer.RNN (RNNHypParams(..), RNNParams, rnnLayer)
import Torch.Layer.BiLSTM (BiLstmHypParams(..), BiLstmParams, biLstmLayers)
import Torch.Layer.LSTM (LstmHypParams(..), LstmParams, lstmLayer)
import Torch.Layer.Linear
import Torch.Control (mapAccumM)
import Torch.Train (update, saveParams)
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
    actionIndexFor :: T.Text -> Int,
    ntIndexFor :: T.Text -> Int
  }

data RNNGState where
  RNNGState ::
    {
      stack :: [Tensor],
      textStack :: [T.Text],
      buffer :: [Tensor],
      textBuffer :: [T.Text],
      actionHistory :: [Tensor],
      textActionHistory :: [T.Text]
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
  RNNGParse ->
  -- | new RNNGState
  IndexData ->
  RNNGState ->
  Action ->
  RNNGState
parse RNNGParse {..} IndexData {..} RNNGState {..} (NT label) =
  let nt_embedding = embedding' (toDependent ntEmbedding) ((asTensor . ntIndexFor) label)
      textAction = showAction (NT label)
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
  let textAction = showAction SHIFT
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
  let textAction = showAction REDUCE
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
  (RNNG, RNNGParse) ->
  IndexData ->
  -- | init RNNGState
  RNNGState ->
  -- | actions
  [Action] ->
  -- | predicted history
  [Tensor] ->
  -- | (predicted actions, new rnngState)
  ([Tensor], RNNGState)
predict _ _ rnngState [] results = (reverse results, rnngState)
predict (rnng, rnngParse) indexData rnngState (action:rest) predictedHitory =
  let predicted = predictNextAction rnng rnngState
      newRNNGState = parse rnngParse indexData rnngState action
  in predict (rnng, rnngParse) indexData newRNNGState rest (predicted:predictedHitory)

rnngForward ::
  (RNNG, RNNGParse) ->
  -- | functions to convert text to index
  IndexData ->
  RNNGSentence ->
  [Tensor]
rnngForward (RNNG {..}, rnngParse) IndexData {..} (RNNGSentence (sents, actions)) =
  let sentsTensor = fmap ((embedding' (toDependent wordEmbedding)) . asTensor . wordIndexFor) sents
      initRNNGState = RNNGState {
        stack = [toDependent stackGuard],
        textStack = [],
        buffer = sentsTensor ++ [toDependent bufferGuard],
        textBuffer = sents,
        actionHistory = [toDependent actionStart],
        textActionHistory = []
      }
  in fst $ predict (RNNG {..}, rnngParse) IndexData {..} initRNNGState actions []

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

  let (wordIndexFor, wordEmbDim) = indexFactory (buildVocab dataForTraining 3 toWordList) False
      (actionIndexFor, actionEmbDim) = indexFactory (buildVocab dataForTraining 1 toActionList) False
      (ntIndexFor, ntEmbDim) = indexFactory (buildVocab dataForTraining 1 toNTList) False
      indexData = IndexData wordIndexFor actionIndexFor ntIndexFor
  
  let batches = dataForTraining

  initRNNGModel <- sample $ RNNGSpec device wordEmbedSize actionEmbedSize wordEmbDim actionEmbDim hiddenSize
  initRNNGParseModel <- sample $ RNNGParseSpec device wordEmbedSize ntEmbDim actionEmbDim hiddenSize

  (trained, losses) <- mapAccumM [1..iter] ((initRNNGModel, optim), (initRNNGParseModel, optim)) $ 
    \epoc ((rnng, opt1), (rnngParse, opt2)) -> do
      (updated, batchLosses) <- mapAccumM batches ((rnng, opt1), (rnngParse, opt2)) $ 
        \batch ((rnng', opt1'), (rnngParse', opt2')) -> do
          let RNNGSentence (sents, actions) = batch
          print actions
          print sents
          let answer = asTensor $ fmap (actionIndexFor . showAction) actions
              output = rnngForward (rnng', rnngParse') indexData (RNNGSentence (sents, actions))
              loss = nllLoss' answer (Torch.stack (Dim 0) output)
          updatedRNNG <- runStep rnng' opt1' loss 1e-2
          updatedRNNGParse <- runStep rnngParse' opt2' loss 1e-2
          return ((updatedRNNG, updatedRNNGParse), (asValue loss::Float))

      let loss = sum batchLosses / (fromIntegral (length batches)::Float)
      return (updated, loss)

  let ((rnngModel, _), (rnngParseModel, _)) = trained
  Torch.Train.saveParams rnngModel (modelFilePath ++ "-rnng")
  Torch.Train.saveParams rnngParseModel (modelFilePath ++ "-rnng-parse")
  drawLearningCurve graphFilePath "Learning Curve" [("", reverse losses)]
  print losses
  return ()
