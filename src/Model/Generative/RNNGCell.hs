{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE StandaloneDeriving #-}

module Model.Generative.RNNGCell where
import Model.Generative.Embedding
import Model.Generative.LSTMComposition
import Model.Generative.MultiLayerLSTMCell
import Model.Generative.FixedStack
import Model.Device
import Data.RNNG (IndexData(..))
import Data.RNNGSentence
import Util

import GHC.Generics
import Data.Binary
import Data.List.Index (ifor)
import System.IO.Unsafe (unsafePerformIO) --base
import Control.Monad


import Torch hiding (take, repeat)
import Torch.Functional.Internal (gather)
-- | hasktorch-tools
import Torch.Layer.LSTM (LstmHypParams(..), LstmParams(..), SingleLstmParams(..), lstmLayers)
import Torch.Layer.Linear (LinearParams(..), LinearHypParams(..), linearLayer)
import Torch.Tensor.Util (unstack)

data RNNGCellSpec = RNNGCellSpec {
  dev :: Device,
  numNts :: Int,
  numLayers :: Int,
  inputDim :: Int,
  hiddenDim :: Int
} deriving (Show, Eq, Generic, Binary)

data RNNGCell where
  RNNGCell ::
    {
      ntEmbedding :: Embedding,
      stackRNN :: MultiLayerLSTMCell,
      composition :: LSTMComposition,
      initialEmb :: Embedding,
      linear :: LinearParams
    } ->
    RNNGCell
  deriving (Show, Generic, Parameterized)


instance
  Randomizable
    RNNGCellSpec
    RNNGCell
  where
    sample RNNGCellSpec {..} = RNNGCell
      <$> (sample $ EmbeddingSpec dev numNts inputDim)
      <*> (sample $ MultiLayerLSTMCellSpec dev numLayers inputDim hiddenDim)
      <*> (sample $ LSTMCompositionSpec dev numLayers inputDim)
      <*> (sample $ EmbeddingSpec dev 1 inputDim)
      <*> (sample $ LinearHypParams dev True hiddenDim inputDim)


-- getInitialHidden
initialize ::
  RNNGCell ->
  Maybe Double ->
  Tensor ->
  -- (<bSize,numLayers,hDim>, <bSize,numLayers,hDim>)
  (Tensor, Tensor)
initialize RNNGCell{..} dropoutProb x =
  let iemb = embedding'' initialEmb dropoutProb (zeros [shape x !! 0] (withDType Int64 $ withDevice (device x) defaultOpts))
  in multiLayerLSTMCellForward stackRNN dropoutProb Nothing iemb

rnngCellForward ::
  RNNGCell ->
  IndexData ->
  Maybe Double ->
  -- | word vecs: <bSize, seqLen, wDim>
  Tensor ->
  -- | action Idx:  <bSize>
  Tensor ->
  FixedStack ->
  -- | (stack, h): (FixedStack, <bSize,numLayers,hDim>)
  IO (FixedStack, Tensor)
rnngCellForward RNNGCell{..} indexData@IndexData{..} dropoutProb wordVecs actionIdxTensor stack@FixedStack{..} = do
  let wDim = last $ shape wordVecs
      dev = device actionIdxTensor
      shiftBatches = squeezeDim 1 $ nonzero $ actionIdxTensor ==. (asTensor' (actionIndexFor SHIFT) (withDType Int64 $ withDevice dev defaultOpts)) -- <numShift>
      -- 0, 1はunkとpad,　頻度順に並んでいるので2:REDUCE, 3:SHIFT
      ntBatches = squeezeDim 1 $ nonzero $ actionIdxTensor >=. (asTensor' (4::Int) (withDType Int64 $ withDevice (device actionIdxTensor) defaultOpts)) -- <numNT>
      reduceBatches = squeezeDim 1 $ nonzero $ actionIdxTensor ==. (asTensor' (actionIndexFor REDUCE) (withDType Int64 $ withDevice dev defaultOpts)) -- <numREDUCE>
  
      batchSize = size 0 actionIdxTensor
      newInput = zeros [batchSize, wDim] (withDevice dev defaultOpts) -- <batchSize, wDim>
  -- print actionIdxTensor
  -- shift
  --  TODO: wordvec idxの動作確認
  (shiftedStack, shiftedInput) <- 
    if (size 0 shiftBatches > 0) 
      then do
        let shiftBatchPointer = pointer ! shiftBatches -- <numShift>
            shiftIdx = expand (view [-1, 1, 1] shiftBatchPointer) False [-1, 1, wDim] -- <numShift, 1, wDim>
            shiftBatchWordVecs = wordVecs ! shiftBatches -- <numShift, seqLen, wDim> 
            shiftEmbs = squeezeDim 1 $ gather shiftBatchWordVecs 1 shiftIdx False -- <numShift, wDim>
            shiftedStack = doShift stack shiftBatches shiftEmbs
            shiftedInput = subst newInput [shiftBatches] shiftEmbs
        return (shiftedStack, shiftedInput)
      else return (stack, newInput)
  
  -- nt
  (ntedStack, ntedInput) <- 
    if (size 0 ntBatches > 0) 
      then do
        let ntIds = (actionIdxTensor ! ntBatches) - (asTensor' (2::Int) (withDType Int64 $ withDevice dev defaultOpts)) -- <numNt>
            ntEmbs = embedding'' ntEmbedding dropoutProb ntIds
            ntedStack = doNt shiftedStack ntBatches ntEmbs ntIds
            ntedInput = subst shiftedInput [ntBatches] ntEmbs
        -- putStrLn "ntEmbs"
        -- print ntEmbs
        return (ntedStack, ntedInput)
      else return (shiftedStack, shiftedInput)

  -- reduce
  (reducedStack, reducedInput) <-
    if (size 0 reduceBatches > 0)
      then do
        let (reducedChildren, childLength, reducedNts, reducedNtIds) = collectReducedChildren stack reduceBatches
            newChild = lstmCompositionForward composition dropoutProb reducedChildren childLength reducedNts reducedNtIds
            reducedStack = doReduce ntedStack reduceBatches newChild
            reducedInput = subst ntedInput [reduceBatches] newChild
        return (reducedStack, reducedInput)
      else return (ntedStack, ntedInput)

  let (newHidden, newCell) = multiLayerLSTMCellForward stackRNN dropoutProb (Just (hiddenHead reducedStack 1, cellHead reducedStack 1)) (view [-1, wDim] reducedInput)
      newStack = updateHidden reducedStack newHidden newCell
      h = select 1 (-1) $ hiddenHead newStack 0

  -- putStrLn "--------------------------------------------"
  -- putStrLn ""
  return (newStack, h)


rnngCellOutput RNNGCell{..} dropoutProb input =
  let dropoutLayer = case dropoutProb of
                      (Just prob) -> unsafePerformIO . (dropout prob True)
                      Nothing -> id
  in relu $ linearLayer linear $ dropoutLayer input
