{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}

module Model.Generative.FixedStack where
import Data.RNNGSentence
import Util

import qualified Data.Text as T
import Debug.Trace
import Data.List
import System.IO.Unsafe (unsafePerformIO) --base

import Torch hiding (take)
import Torch.Functional.Internal (gather)
-- | hasktorch-tools
import Torch.Tensor.Util (unstack)

data FixedStack where
  FixedStack ::
    {
      -- word pointer: <batchSize>
      pointer :: Tensor, 
      -- stack top position: <batchSize>
      topPosition :: Tensor,
      -- 
      -- h: <stackSize+1,bSize,numLayers,hDim>  pytorch版では(batch,stackSize+1, hidden_size, num_layers) 
      hiddens :: Tensor,
      -- c: <stackSize,numLayers,bSize,hDim>　
      cells :: Tensor,
      -- <stackSize, batchSize, inputSize>
      trees :: Tensor,
      -- treesのどこがntか (1-idx) <stackSize, batchSize>
      ntIndex :: Tensor,
      -- ntIndexが持っているのはどのntか
      -- <stackSize, batchSize>
      ntIds :: Tensor,
      -- ntIndexの何番目が最近のntか (0-idx)　default is -1 (0 means zero-dim exists): <batchSize>
      ntIndexPos :: Tensor,
      -- <batchSize>
      batchIndex :: Tensor,

      stackSize :: Tensor
    } ->
    FixedStack 
  deriving (Show)


initializeStack ::
  -- stack size
  Int ->
  -- input size
  Int ->
  -- initial (h,c): (<bSize,numLayers,hDim>, <bSize,numLayers,hDim>)
  (Tensor, Tensor) ->
  FixedStack
initializeStack stackSize inputSize (h, c) =
  let numLayers = size 1 h
      hiddenSize = size 2 h
      batchSize = size 0 h
      dev = device h
  in FixedStack {
      pointer = zeros [batchSize] (withDType Int64 $ withDevice dev defaultOpts),
      topPosition = zeros [batchSize] (withDType Int64 $ withDevice dev defaultOpts),
      hiddens = cat (Dim 0) $ [unsqueeze (Dim 0) h, zeros [stackSize + 1, batchSize, numLayers, hiddenSize] (withDevice dev defaultOpts)],
      cells = cat (Dim 0) $ [unsqueeze (Dim 0) c, zeros [stackSize + 1, batchSize, numLayers, hiddenSize] (withDevice dev defaultOpts)],
      trees = zeros [stackSize, batchSize, inputSize] (withDevice dev defaultOpts),
      ntIndex = zeros [stackSize, batchSize] (withDType Int64 $ withDevice dev defaultOpts),
      ntIds = zeros [stackSize, batchSize] (withDType Int64 $ withDevice dev defaultOpts),
      ntIndexPos = unsafePerformIO $ clone $ expand (asTensor' (-1::Int) (withDType Int64 $ withDevice dev defaultOpts)) False [batchSize],
      batchIndex = arange 0 batchSize 1 (withDType Int64 $ withDevice dev defaultOpts),
      stackSize = asTensor' stackSize (withDType Int64 $ withDevice dev defaultOpts)
    }

-- 先頭のhiddenを返す
hiddenHead ::
  FixedStack ->
  -- | offset
  Int -> 
  -- | <bSize,numLayers,hDim>
  Tensor
hiddenHead FixedStack{..} offset = 
  hiddens !. [(view [-1] topPosition) - (asTensor offset), batchIndex]
  -- Torch.stack (Dim 1) $ zipWith (select 1) batchIndex (unstack (cells ! (view [-1] topPosition - 1))) 

-- 先頭のhiddenを返す
cellHead ::
  FixedStack ->
  -- | offset
  Int ->
  -- | <numLayers,bSize,hDim>
  Tensor
cellHead FixedStack{..} offset =
  cells !. [(view [-1] topPosition) - (asTensor offset), batchIndex]


doShift ::
  FixedStack ->
  -- shiftBatches: <numShift>
  Tensor ->
  -- shiftEmbs: <numShift, wDim>
  Tensor ->
  FixedStack
doShift FixedStack{..} shiftBatches shiftEmbs = 
  -- stackSizeのうち、topPositionのTensorを取り出す
  let newTrees = subst trees [(topPosition ! shiftBatches), shiftBatches] shiftEmbs
      -- shiftされているところだけ、pointerを１進める
      newPointer = subst pointer [shiftBatches] ((pointer + 1) ! shiftBatches)
      newTopPosition = subst topPosition [shiftBatches] ((topPosition + 1) ! shiftBatches)
  in FixedStack {
      pointer = newPointer,
      topPosition = newTopPosition,
      hiddens = hiddens,
      cells = cells,
      trees = newTrees,
      ntIndex = ntIndex,
      ntIds = ntIds,
      ntIndexPos = ntIndexPos,
      batchIndex = batchIndex,
      stackSize = stackSize
  }

doNt ::
  FixedStack ->
  -- shiftBatches: <numShift>
  Tensor ->
  -- shiftEmbs: <numShift, wDim>
  Tensor ->
  -- batchedNtIds: -- <numNt>
  Tensor ->
  FixedStack
doNt FixedStack{..} ntBatches ntEmbs batchedNtIds = 
  let newTrees = subst trees [topPosition ! ntBatches, ntBatches] ntEmbs
      newNtIndexPos = subst ntIndexPos [ntBatches] ((ntIndexPos + 1) ! ntBatches)
      newNtIds = subst ntIds [newNtIndexPos ! ntBatches, ntBatches] batchedNtIds
      newTopPosition = subst topPosition [ntBatches] ((topPosition + 1) ! ntBatches)
      newNtIndex = subst ntIndex [(newNtIndexPos ! ntBatches), ntBatches] (newTopPosition ! ntBatches)
  in FixedStack {
      pointer = pointer,
      topPosition = newTopPosition,
      hiddens = hiddens,
      cells = cells,
      trees = newTrees,
      ntIndex = newNtIndex,
      ntIds = newNtIds,
      ntIndexPos = newNtIndexPos,
      batchIndex = batchIndex,
      stackSize = stackSize
    }

doReduce ::
  FixedStack ->
  -- reduceBatches: <numShift>
  Tensor ->
  -- <numReduce, wDim>
  Tensor ->
  FixedStack
doReduce FixedStack{..} reduceBatches newChild =
  let prevNtPosition = ntIndex !. [ntIndexPos ! reduceBatches, reduceBatches]
      newTrees = subst trees [prevNtPosition - 1, reduceBatches] newChild
      newNtIndexPos = subst ntIndexPos [reduceBatches] (ntIndexPos ! reduceBatches - 1)
      newTopPosition = subst topPosition [reduceBatches] prevNtPosition
  in FixedStack {
      pointer = pointer,
      topPosition = newTopPosition,
      hiddens = hiddens,
      cells = cells,
      trees = newTrees,
      ntIndex = ntIndex,
      ntIds = ntIds,
      ntIndexPos = newNtIndexPos,
      batchIndex = batchIndex,
      stackSize = stackSize
    }

collectReducedChildren :: 
  FixedStack ->
  -- <numREDUCE>
  Tensor ->
  -- (reduced children embeddings, child length, reduced NT embeddings, reduced NT ids): (<batchSize, maxNumChild, wDim>, <batchSize>, <numReduce, wDim>, <numReduce>)
  (Tensor, Tensor, Tensor, Tensor)
collectReducedChildren FixedStack{..} reduceBatches =
  let ntIndexPos' = ntIndexPos ! reduceBatches
      prevNtPosition = ntIndex !. [ntIndexPos', reduceBatches]  -- <numReduce>
      reducedNtIds = ntIds !. [ntIndexPos', reduceBatches] -- <numReduce>
      reducedNts = trees !. [prevNtPosition - 1, reduceBatches] -- <numReduce, wDim>
      childLength = (topPosition ! reduceBatches) `sub` prevNtPosition -- <batchSize>
      maxChLength = asValue (Torch.max childLength)::Int
      childIdx'' = unsqueeze (Dim 1) prevNtPosition + arange 0 maxChLength 1 (withDType Int64 $ withDevice (Torch.device prevNtPosition) defaultOpts) -- <numReduce, childLength>
      -- ceiled at maximum stack size (exceeding this may occur for some batches, but those should be ignored safely.)
      exceedStackSize = squeezeDim 1 $ nonzero (childIdx'' >=. stackSize)
      childIdx' = subst childIdx'' [select 1 0 exceedStackSize, select 1 1 exceedStackSize] (Torch.repeat [asValue childLength::Int] (stackSize - 1))
      childIdx = expand (unsqueeze (Dim (-1)) $ childIdx') False [-1, -1, size (-1) trees] -- <numReduce, maxNumChild, wDim>
      reducedChildren = gather (Torch.transpose (Dim 0) (Dim 1) $ indexSelect 1 reduceBatches trees) 1 childIdx False -- <numReduce, maxNumChild, wDim>
  in (reducedChildren, childLength, reducedNts, reducedNtIds)

updateHidden ::
  FixedStack ->
  -- <numLayers, bSize, hDim>
  Tensor ->
  -- <numLayers, bSize, hDim>
  Tensor ->
  FixedStack
updateHidden FixedStack{..} newHidden newCell =
  let pos = unsafePerformIO $ clone $ reshape [-1] topPosition
      newHiddens = subst hiddens [pos, batchIndex] newHidden
      newCells = subst hiddens [pos, batchIndex] newCell
  in FixedStack {
      pointer = pointer,
      topPosition = topPosition,
      hiddens = newHiddens,
      cells = newCells,
      trees = trees,
      ntIndex = ntIndex,
      ntIds = ntIds,
      ntIndexPos = ntIndexPos,
      batchIndex = batchIndex,
      stackSize = stackSize
    }
