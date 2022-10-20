{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}

module RNN where
import Torch.Typed
import qualified Torch as D
import GHC.TypeNats
import GHC.Generics

data 
  RNNSpec
    (inputSize :: Nat)
    (hiddenSize :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  = RNNSpec
  deriving (Show, Eq)

data 
  RNN
    (inputSize :: Nat)
    (hiddenSize :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
    RNN ::
      forall inputSize hiddenSize dtype device.
      { inh :: Linear inputSize hiddenSize dtype device,
        hh :: Linear hiddenSize hiddenSize dtype device
      } ->
      RNN inputSize hiddenSize dtype device
    deriving (Show, Generic, Parameterized)

instance
  ( KnownNat inputSize,
    KnownNat hiddenSize,
    KnownDType dtype,
    KnownDevice device,
    RandDTypeIsValid device dtype
  ) =>
  Randomizable
    (RNNSpec inputSize hiddenSize dtype device)
    (RNN inputSize hiddenSize dtype device)
  where
  sample RNNSpec =
    RNN
      <$> sample LinearSpec
      <*> sample LinearSpec

rnnCellForward ::
  ( KnownNat inputSize,
    KnownNat hiddenSize,
    KnownDType dtype,
    KnownDevice device,
    RandDTypeIsValid device dtype,
    StandardFloatingPointDTypeValidation device dtype
  ) =>
  RNN inputSize hiddenSize dtype device ->
  -- h_t-1
  Tensor device dtype '[hiddenSize] ->
  -- x_t
  Tensor device dtype '[inputSize] ->
  -- h_t
  Tensor device dtype '[hiddenSize]
rnnCellForward RNN {..} h x =
  Torch.Typed.tanh ( linearForward inh x + linearForward hh h )


-- TODO : Batch対応
-- TODO : Tensorのままtraverse
rnn ::
  forall  inputSize hiddenSize dtype device.
  ( KnownNat inputSize,
    KnownNat hiddenSize,
    KnownDType dtype,
    KnownDevice device,
    RandDTypeIsValid device dtype,
    StandardFloatingPointDTypeValidation device dtype
  ) =>
  -- | model
  RNN inputSize hiddenSize dtype device ->
  -- | input
  [Tensor device dtype '[inputSize]] ->
  -- | (output, h_t)
  ([Tensor device dtype '[hiddenSize]], Tensor device dtype '[hiddenSize])
rnn model input =
  let hidden = scanl (rnnCellForward model) (zeros @'[hiddenSize] @dtype @device) input
  in (tail hidden, last hidden)
