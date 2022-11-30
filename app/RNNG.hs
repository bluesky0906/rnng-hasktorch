{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE GADTs #-}

module RNNG where
import PTB
import Util
import qualified Data.Text as T 
import Data.List.Split (chunksOf, splitEvery) --split
import Data.List
import Torch.Control (mapAccumM)
import Torch.Train (update)
import GHC.Generics
import Torch
import Torch.Layer.LSTM (LstmHypParams(..), LstmParams, lstmLayer)
import Torch.Layer.BiLSTM (BiLstmHypParams(..), BiLstmParams, biLstmLayers)
import Torch.Layer.RNN (RNNHypParams(..), RNNParams, rnnLayer)
import Torch.Layer.Linear


data RNNGSpec = RNNGSpec {
  modelDevice :: Device,
  wordEmbedSize :: Int,
  actionEmbedSize :: Int,
  wordNumEmbed :: Int,
  ntNumEmbed :: Int,
  actionNumEmbed :: Int,
  hiddenSize :: Int
} deriving (Show, Eq)


data RNNG where
  RNNG ::
    {
      wordEmbedding :: Parameter,
      ntEmbedding :: Parameter,
      actionEmbedding :: Parameter,
      h0 :: Parameter,
      c0 :: Parameter,
      bufferRNN :: RNNParams,
      actionRNN :: RNNParams,
      stackLSTM :: LstmParams,
      compLSTM :: BiLstmParams,
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
    sample RNNGSpec {..}  =
      RNNG
        <$> (makeIndependent =<< randnIO' [wordNumEmbed, wordEmbedSize])
        <*> (makeIndependent =<< randnIO' [ntNumEmbed, wordEmbedSize])
        <*> (makeIndependent =<< randnIO' [actionNumEmbed, wordEmbedSize])
        <*> (makeIndependent =<< randnIO' [hiddenSize])
        <*> (makeIndependent =<< randnIO' [hiddenSize])
        <*> sample (RNNHypParams modelDevice wordEmbedSize hiddenSize)
        <*> sample (RNNHypParams modelDevice wordEmbedSize hiddenSize)
        <*> sample (LstmHypParams modelDevice hiddenSize)
        <*> sample (BiLstmHypParams modelDevice 1 hiddenSize)
        <*> (makeIndependent =<< randnIO' [hiddenSize * 3, hiddenSize])
        <*> (makeIndependent =<< randnIO' [hiddenSize])
        <*> sample (LinearHypParams modelDevice hiddenSize actionNumEmbed)
        <*> (makeIndependent =<< randnIO' [actionEmbedSize])
        <*> (makeIndependent =<< randnIO' [wordEmbedSize])
        <*> (makeIndependent =<< randnIO' [wordEmbedSize])

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
    RNNGState deriving(Show)

data IndexData where
  IndexData ::
    {
      wordIndexFor :: T.Text -> Int,
      actionIndexFor :: T.Text -> Int,
      ntIndexFor :: T.Text -> Int
    } ->
    IndexData


push ::
  [a] ->
  a ->
  [a]
push list d = list ++ [d]

predictNextAction ::
  RNNG ->
  -- | data
  RNNGState ->
  -- | possibility of actions
  Tensor
predictNextAction RNNG {..} RNNGState {..} = 
  let (_, b_embedding) = rnnLayer bufferRNN (toDependent $ h0) buffer
      s_embedding = fst $ head $ lstmLayer stackLSTM (toDependent $ c0, toDependent $ h0) stack
      (_, a_embedding) = rnnLayer actionRNN (toDependent $ h0) actionHistory
      ut = Torch.tanh $ ((cat (Dim 0) [s_embedding, b_embedding, a_embedding]) `matmul` (toDependent w)) + (toDependent c)
  in logSoftmax (Dim 0) $ linearLayer linearParams ut

parse ::
  RNNG ->
  -- | new RNNGState
  IndexData ->
  RNNGState ->
  Action ->
  RNNGState
parse RNNG {..} IndexData {..} RNNGState {..} (NT label) =
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
parse RNNG {..} IndexData {..} RNNGState {..} SHIFT =
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
parse RNNG {..} IndexData {..} RNNGState {..} REDUCE =
  let textAction = showAction REDUCE
      action_embedding = embedding' (toDependent actionEmbedding) ((asTensor . actionIndexFor) textAction)
      (Just idx) = findIndex (\elem -> (T.isPrefixOf (T.pack "(") elem) && not (T.isSuffixOf (T.pack ")") elem)) (reverse textStack)
      splitedIdx = (length textStack) - idx - 1
      (newTextStack, textSubTree) = splitAt splitedIdx textStack
      (newStack, subTree) = splitAt splitedIdx stack
      composedSubTree = fst $ last $ biLstmLayers compLSTM (toDependent $ c0, toDependent $ h0) subTree
  in RNNGState {
      stack = push newStack composedSubTree,
      textStack = push newTextStack (T.intercalate (T.pack " ") (push textSubTree (T.pack ")"))),
      buffer = buffer,
      textBuffer = textBuffer,
      actionHistory = push actionHistory action_embedding,
      textActionHistory = push textActionHistory textAction
    }

predict ::
  RNNG ->
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
predict rnng indexData rnngState (action:rest) predictedHitory =
  let predicted = predictNextAction rnng rnngState
      newRNNGState = parse rnng indexData rnngState action
  in predict rnng indexData newRNNGState rest (predicted:predictedHitory)


rnng ::
  -- | model
  RNNG ->
  -- | input
  RNNGSentence ->
  -- | functions to convert text to index
  IndexData ->
  -- | correct Data
  Tensor ->
  Tensor
rnng RNNG {..} (RNNGSentence (sents, actions)) IndexData {..} answer =
  let sentenceIdx = fmap (asTensor . wordIndexFor) sents
      initRNNGState = RNNGState {
        stack = [toDependent stackGuard],
        textStack = [],
        buffer = (fmap (embedding' (toDependent wordEmbedding)) sentenceIdx) ++ [toDependent bufferGuard],
        textBuffer = sents,
        actionHistory = [toDependent actionStart],
        textActionHistory = []
      }
      (predicted, lastState) = predict RNNG {..} IndexData {..} initRNNGState actions []
  in nllLoss' answer (Torch.stack (Dim 0) predicted)


main :: IO()
main = do
  -- hyper parameter
  config <- configLoad
  let iter = fromIntegral (getTrial config)::Int
      wordEmbedSize = fromIntegral (getWordEmbedSize config)::Int
      actionEmbedSize = fromIntegral (getActionEmbedSize config)::Int
      hiddenSize = fromIntegral (getHiddenSize config)::Int
      numLayer = fromIntegral (getNumLayer config)::Int
  -- data
  trainingData <- loadActionsFromBinary $ getTrainingDataPath config
  validationData <- loadActionsFromBinary $ getValidationDataPath config
  evaluationData <- loadActionsFromBinary $ getEvaluationDataPath config

  -- 出現頻度が3回以上の語彙に絞る
  let (wIndexFor, wordEmbDim) = indexFactory (buildVocab evaluationData 3 toWordList) False
      (aIndexFor, actionEmbDim) = indexFactory (buildVocab evaluationData 1 toActionList) False
      (nIndexFor, ntEmbDim) = indexFactory (buildVocab evaluationData 1 toNTList) False
      indexData = IndexData {
                    wordIndexFor = wIndexFor,
                    actionIndexFor = aIndexFor,
                    ntIndexFor = nIndexFor
                  }

  initModel <- sample $ RNNGSpec (Device CPU 0) wordEmbedSize actionEmbedSize wordEmbDim ntEmbDim actionEmbDim hiddenSize
  ((trainedModel, _),losses) <- mapAccumM [1..iter] (initModel, GD) $ \epoc (model,opt) -> do
    (updated, loss) <- mapAccumM evaluationData (model, opt) $ \batch (model',opt') -> do
      let RNNGSentence (sents, actions) = batch
          answer = asTensor $ fmap (aIndexFor . showAction) (Data.List.take 1 actions)
          loss = rnng initModel (RNNGSentence (Data.List.take 1 sents, Data.List.take 1 actions)) indexData answer
      print loss
      u <- update model' opt' loss 5e-4
      return (u, loss)
    return (updated, loss)
  return ()

{-
config <- configLoad
wordEmbedSize = fromIntegral (getWordEmbedSize config)::Int
actionEmbedSize = fromIntegral (getActionEmbedSize config)::Int
hiddenSize = fromIntegral (getHiddenSize config)::Int
evaluationData <- loadActionsFromBinary $ getEvaluationDataPath config

vocab = buildVocab evaluationData 3
(wIndexFor, wordEmbDim) = indexFactory (buildVocab evaluationData 3 toWordList) False
(aIndexFor, actionEmbDim) = indexFactory (buildVocab evaluationData 1 toActionList) False
(nIndexFor, ntEmbDim) = indexFactory (buildVocab evaluationData 1 toNTList) False

initModel <- sample $ RNNGSpec (Device CPU 0) wordEmbedSize actionEmbedSize wordEmbDim ntEmbDim actionEmbDim hiddenSize
indexData = IndexData {wordIndexFor = wIndexFor, actionIndexFor = aIndexFor, ntIndexFor = nIndexFor}
RNNGSentence (sents, actions) = (head evaluationData) 
answer =  asTensor $ fmap (aIndexFor . showAction) actions
loss = rnng initModel (head evaluationData) indexData answer
-}