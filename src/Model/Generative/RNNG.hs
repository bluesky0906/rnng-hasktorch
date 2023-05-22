{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE StandaloneDeriving #-}

module Model.Generative.RNNG where
import Data.CFG
import Data.RNNGSentence
import Data.RNNG
import Model.Generative.RNNGCell
import Model.Generative.LSTMComposition
import Model.Generative.Embedding
import Model.Generative.FixedStack
import Model.Device

import Torch hiding (take)
-- | hasktorch-tools
import Torch.Layer.RNN (RnnHypParams(..), RnnParams, rnnLayers)
import Torch.Layer.LSTM (LstmHypParams(..), LstmParams, lstmLayers)
import Torch.Layer.Linear (LinearHypParams(..), LinearParams, linearLayer)
import Torch.Layer.NonLinear (ActName(..))
import Torch.Tensor.Util (unstack)
import Torch.Control (mapAccumM)
import Util

import GHC.Generics
import qualified Data.Text as T
import qualified Data.Map as M
import Data.List
import Data.Binary
import Debug.Trace
import System.IO.Unsafe (unsafePerformIO) --base
import Control.Monad


data RNNGSpec = RNNGSpec {
  dev :: Device,
  numLayers :: Int,
  numWords :: Int,
  numActions :: Int,
  numNts :: Int,
  wordDim :: Int,
  hiddenDim :: Int
} deriving (Show, Eq, Generic, Binary)


{-

RNNG model

-}


data RNNG where
  RNNG ::
    {
      wordEmbedding :: Embedding,
      -- actionEmbedding :: Embedding,
      rnng :: RNNGCell,
      -- stackToHidden :: LinearParams,
      vocabMLP :: LinearParams,
      actionMLP :: LinearParams
    } ->
    RNNG
  deriving (Show, Generic, Parameterized)


instance
  Randomizable
    RNNGSpec
    RNNG
  where
    sample RNNGSpec {..} = RNNG
      <$> (sample $ EmbeddingSpec dev numWords wordDim)
      -- <*> (sample $ EmbeddingSpec dev numActions wordDim)
      <*> (sample $ RNNGCellSpec dev numNts numLayers wordDim hiddenDim)
      -- <*> (sample $ LinearHypParams dev True hiddenDim wordDim)
      <*> (sample $ LinearHypParams dev True wordDim numWords)
      <*> (sample $ LinearHypParams dev True wordDim numActions)

-- instance Show RNNGState where
--   show RNNGState {..} = unlines [
--       "textStack: " ++ show textStack,
--       "textBuffer: " ++ show textBuffer,
--       "textActionHistory: " ++ show textActionHistory,
--       "numOpenParen: " ++ show numOpenParen
--     ]


{-

  Data Structure for RNNG

-}


rnngForward ::
  Mode ->
  RNNG ->
  IndexData ->
  [[Int]] ->
  [[Int]] ->
  -- | loss, action loss, word loss
  (Tensor, Tensor, Tensor)
rnngForward mode@Mode{..} model@RNNG{..} indexData@IndexData{..} sents actions = unsafePerformIO $ do
  -- initialize
  let sentsIdxTensor = toDevice device $ asTensor sents
      wordVecs = embedding'' wordEmbedding dropoutProb sentsIdxTensor
      wDim = last $ shape wordVecs
      -- stackSize = 10
      stackSize = Prelude.max 100 (size 1 sentsIdxTensor + 10)
      initialHC = initialize rnng dropoutProb sentsIdxTensor 
      initialStack = initializeStack stackSize wDim initialHC

  -- unroll states
  let hs0 = select 1 (-1) $ hiddenHead initialStack 0
      actionIdxTensor = toDevice device $ asTensor actions
      -- 1 stepごとのかたまりにする[Tensor<batchSize>]
      stepActionIdxTensor = map (squeezeDim 1) $ split 1 (Dim 1) actionIdxTensor
  (stack, hsRest) <- mapAccumM (init stepActionIdxTensor) initialStack (rnngCellForward rnng indexData dropoutProb wordVecs)  
  let hs = Torch.stack (Dim 0) (hs0:hsRest)
      actionContexts = rnngCellOutput rnng dropoutProb $ contiguous $ Torch.transpose (Dim 0) (Dim 1) hs
      aLoss = actionLoss model actionIdxTensor actionContexts
      wLoss = wordLoss model indexData sentsIdxTensor actionIdxTensor actionContexts
      loss = (Torch.sumAll aLoss) + (Torch.sumAll wLoss)
  return (loss, aLoss, wLoss)

actionLoss ::
  RNNG ->
  -- <bSize, actionLen>
  Tensor ->
  -- <bSize, actionLen, wDim>
  Tensor ->
  -- loss
  Tensor
actionLoss RNNG{..} actions hiddens =
  let actions' = view [-1] actions -- 一次元にする
      hiddens' = view [size 0 actions', -1] hiddens
      actionMask = actions' /=. 1 -- <batchSize, actionLen> (padding idxは 1)
      idx = squeezeDim 1 $ nonzero actionMask
      nonPadActions = actions' ! idx
      nonPadHiddens = hiddens' ! idx
      logit = linearLayer actionMLP nonPadHiddens
      loss = nllLoss ReduceNone 1 nonPadActions (logSoftmax (Dim 1) logit)
  in loss

wordLoss ::
  RNNG ->
  IndexData ->
  Tensor ->
  Tensor ->
  Tensor ->
  Tensor
wordLoss RNNG{..} IndexData{..} x actions hiddens =
  let actions' = view [-1] actions
      hiddens' = view [size 0 actions', -1] hiddens
      actionMask = actions' ==. (asTensor' (actionIndexFor SHIFT) (withDevice (Torch.device actions) defaultOpts))
      idx = squeezeDim 1 $ nonzero actionMask
      shiftHiddens = hiddens' ! idx
      x' = view [-1] x
      nonPadX = x' ! (x' /=. 1)
      logit = linearLayer vocabMLP shiftHiddens
      loss = nllLoss ReduceNone 1 nonPadX (logSoftmax (Dim 1) logit)
  in loss
