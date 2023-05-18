{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}


module Model.Generative.LSTMComposition where
import GHC.Generics
import Data.Binary
import Data.List.Index (ifor)

import Model.Device
import Util

import Torch hiding (take, repeat)
import Torch.Functional.Internal (gather)
-- | hasktorch-tools
import Torch.Layer.LSTM (LstmHypParams(..), LstmParams(..), SingleLstmParams(..), lstmLayers)
import Torch.Layer.Linear (LinearHypParams(..), LinearParams(..), linearLayer)
import Torch.Tensor.Util (unstack)
import Debug.Trace

import System.IO.Unsafe (unsafePerformIO) --base


data LSTMCompositionSpec = LSTMCompositionSpec {
  dev :: Device,
  numLayers :: Int,
  dim :: Int
} deriving (Show, Eq, Generic, Binary)

data LSTMComposition where
  LSTMComposition ::
    {
      lstm :: LstmParams,
      linear :: LinearParams
      -- h0 :: Parameter,
      -- c0 :: Parameter
      -- batchIndex :: Tensor
    } ->
    LSTMComposition
  deriving (Show, Generic, Parameterized)


instance
  Randomizable
    LSTMCompositionSpec
    LSTMComposition
  where
    sample LSTMCompositionSpec {..} = LSTMComposition
      <$> (sample $ LstmHypParams dev True dim dim numLayers True Nothing)
      <*> (sample $ LinearHypParams dev True (dim * 2) dim)
      -- <*> (makeIndependent =<< zeros [2 * numLayers, batchSize, hiddenDim] (withDevice dev defaultOpts))
      -- <*> (makeIndependent =<< zeros [2 * numLayers, batchSize, hiddenDim] (withDevice dev defaultOpts))
      -- <*> (return $ arange 0 10000 1 (withDType Int64 $ withDevice dev defaultOpts))

lstmCompositionForward ::
  LSTMComposition ->
  Maybe Double ->
  -- | children: <bSize, seqLen, iDim>
  Tensor ->
  -- | chLengths: <batchSize>
  Tensor ->
  -- | nt: <numReduce, wDim>
  Tensor ->
  -- | ntId: <numReduce>
  Tensor ->
  -- | <numReduce, wDim>
  Tensor
lstmCompositionForward LSTMComposition{..} dropoutProb children chLengths nt ntId =
  let batchIndex = arange 0 10000 1 (withDType Int64 $ withDevice (device children) defaultOpts)
      batchSize = size 0 children
      numLayers = length (restLstmParams lstm) + 1
      (hiddenDim:_) = shape $ toDependent $ Torch.Layer.Linear.weight $ forgetGate $ firstLstmParams lstm
      dev = device children
      lengths = chLengths + 2
      nt' = unsqueeze (Dim 1) nt
      -- 直接ここで[nt', children, nt']にしちゃだめ？
      elems' = cat (Dim 1) [nt', children, zerosLike nt']
      elems = subst elems' [batchIndex ! Slice (None, size 0 elems'), lengths - 1] nt -- <numReduce, childLengths + 2, wDim>
      h0c0 = (zeros [2 * numLayers, batchSize, hiddenDim] (withDevice dev defaultOpts), zeros [2 * numLayers, batchSize, hiddenDim] (withDevice dev defaultOpts))
      (h, _) = lstmLayers lstm dropoutProb True h0c0 elems -- <numReduce, childLength + 2, 2 * wDim>
      gatherIdx = unsqueeze (Dim 1) $ expand (unsqueeze (Dim 1) (lengths - 2)) False [-1, size (-1) h] -- <numReduce, 1, 2 * wDim>
      fwd = indexSelect 1 (arange 0 hiddenDim 1 (withDType Int64 $ withDevice dev defaultOpts)) (squeezeDim 1 $ gather h 1 gatherIdx False) -- <numReduce, wDim>
      bwd = indexSelect 1 (arange hiddenDim (size 2 h) 1 (withDType Int64 $ withDevice dev defaultOpts)) (select 1 1 h) -- <numReduce, wDim>
      c = Torch.cat (Dim 1) [fwd, bwd] -- <numReduce, wDim>
      dropoutLayer = case dropoutProb of
                      (Just prob) -> unsafePerformIO . (dropout prob True)
                      Nothing -> id
      output = relu $ linearLayer linear $ dropoutLayer c -- <numReduce, wDim>
  in output

