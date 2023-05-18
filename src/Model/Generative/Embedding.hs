{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE StandaloneDeriving #-}

module Model.Generative.Embedding where
import GHC.Generics
import Data.Binary
import System.IO.Unsafe (unsafePerformIO) --base

import Model.Device

import Torch hiding (take, repeat)
-- | hasktorch-tools
import Torch.Layer.LSTM (LstmHypParams(..), LstmParams, lstmLayers)

data EmbeddingSpec = EmbeddingSpec {
  dev :: Device,
  numEmbed :: Int,
  dim :: Int
} deriving (Show, Eq, Generic, Binary)

data Embedding where
  Embedding ::
    {
      emb :: Parameter
    } ->
    Embedding
  deriving (Show, Generic, Parameterized)


instance
  Randomizable
    EmbeddingSpec
    Embedding
  where
    sample EmbeddingSpec {..} = Embedding
      <$> (makeIndependent =<< randnIO' [numEmbed, dim])

embedding'' ::
  Embedding ->
  Maybe Double ->
  Tensor ->
  Tensor
embedding'' Embedding{..} dropoutProb idxTensor =
  dropoutLayer dropoutProb $ embedding' (toDependent emb) idxTensor
  where 
    dropoutLayer (Just prob) = unsafePerformIO . (dropout prob False)
    dropoutLayer Nothing = id
  

