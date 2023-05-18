{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE StandaloneDeriving #-}

module Model.Generative.MultiLayerLSTMCell where
import Model.Device
import Util

import GHC.Generics
import Data.Binary
import Data.List.Index (ifor)
import Torch hiding (take, repeat)
-- | hasktorch-tools
import Torch.Layer.LSTM (LstmHypParams(..), LstmParams(..), SingleLstmParams(..), lstmLayers)
import Torch.Layer.Linear (LinearParams(..))


data MultiLayerLSTMCellSpec = MultiLayerLSTMCellSpec {
  dev :: Device,
  numLayers :: Int,
  inputDim :: Int,
  hiddenDim :: Int
} deriving (Show, Eq, Generic, Binary)

data MultiLayerLSTMCell where
  MultiLayerLSTMCell ::
    {
      lstm :: LstmParams
    } ->
    MultiLayerLSTMCell
  deriving (Show, Generic, Parameterized)


instance
  Randomizable
    MultiLayerLSTMCellSpec
    MultiLayerLSTMCell
  where
    sample MultiLayerLSTMCellSpec {..} = MultiLayerLSTMCell
      <$> (sample $ LstmHypParams dev False inputDim hiddenDim numLayers True Nothing)

-- TODO: pytorchと同じく<bSize,numLayers,hDim>にする
multiLayerLSTMCellForward ::
  MultiLayerLSTMCell ->
  Maybe Double ->
  -- | (h_n, c_n): (<bSize,numLayers,hDim>, <bSize,numLayers,hDim>)
  Maybe (Tensor, Tensor) -> 
  -- | input: <bSize, iDim>
  Tensor ->
  -- | (h_n+1, c_n+1): (<bSize,numLayers,hDim>, <bSize,numLayers,hDim>)
  (Tensor, Tensor)
multiLayerLSTMCellForward MultiLayerLSTMCell{..} dropout prevHC input =
  let (batchSize:_) = shape input
      numLayers = length (restLstmParams lstm) + 1
      (hiddenDim:_) = shape $ toDependent $ Torch.Layer.Linear.weight $ forgetGate $ firstLstmParams lstm
      prev = case prevHC of
                    Just (h, c) -> (transpose (Dim 0) (Dim 1) h, transpose (Dim 0) (Dim 1) c)
                    Nothing -> (zeros [numLayers, batchSize, hiddenDim] (withDevice (device input) defaultOpts), zeros [numLayers, batchSize, hiddenDim] (withDevice (device input) defaultOpts))
      (h, c) = snd $ lstmLayers lstm dropout True prev (unsqueeze (Dim 1) input)
  in (transpose (Dim 0) (Dim 1) h, transpose (Dim 0) (Dim 1) c)

