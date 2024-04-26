{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
-- {-# OPTIONS_GHC -fdefer-type-errors #-}

-- :seti -XTypeApplications -XDataKinds -XScopedTypeVariables -XAllowAmbiguousTypes -XKindSignatures -XFlexibleContexts -XGADTs -XTypeOperators　-XConstraintKinds -XRecordWildCards -XNoStarIsType

module RNNG where
import PTB
import Util
import Data.List.Split (chunksOf, splitEvery) --split
import Data.Proxy
import Torch.Control (mapAccumM)
import GHC.Generics
import GHC.TypeNats
import qualified GHC.Natural as N
import Torch.Typed
import qualified Torch.Tensor as D
import Unsafe.Coerce
import Data.Constraint


type EmbedSize = 10

type HiddenSize = 5

type BatchSize = 2

type NumEmbed = 10

type NumLayers = 2

type ModelDevice = '( 'CPU, 0)

data RNNGSpec 
  = RNNGSpec
  deriving (Show, Eq)

data RNNG where
  RNNG ::
    {
      embedding :: Embedding ('Just 1) NumEmbed EmbedSize 'Learned 'Float ModelDevice,
      lstmWithInit :: LSTMWithInit EmbedSize HiddenSize NumLayers 'Bidirectional 'LearnedInitialization 'Float '( 'CPU, 0),
      linear :: Linear (HiddenSize * (NumberOfDirections 'Bidirectional)) 3 'Float ModelDevice
    } ->
    RNNG
  deriving (Show, Generic, Parameterized)

instance
  Randomizable
    RNNGSpec
    RNNG
  where
    sample RNNGSpec =
      RNNG
        <$> sample (LearnedEmbeddingWithRandomInitSpec @('Just 1) @NumEmbed @EmbedSize @'Float @ModelDevice)
        <*> sample (LSTMWithLearnedInitSpec (
                      LSTMSpec @EmbedSize @HiddenSize @NumLayers @'Bidirectional @'Float @ModelDevice (DropoutSpec 0.1)
                    ) zeros zeros)
        <*> sample (LinearSpec @(HiddenSize * (NumberOfDirections 'Bidirectional)) @3 @'Float @ModelDevice)

rnng :: forall shape batchSize seqLen.
  ( KnownShape shape,
    KnownNat batchSize,
    KnownNat seqLen,
    InRange shape 0 (seqLen-1)
  ) =>
  RNNG -> 
  Tensor ModelDevice 'Int64 shape 
  -> IO()
rnng RNNG {..} tensor = do
  let input = embed @('Just 1) @shape embedding tensor
      (lstmOutput, _, _) = lstmForwardWithDropout @'BatchFirst @batchSize @seqLen lstmWithInit (unsafeCoerce input)
      -- output = forward linear lstmOutput
  print lstmOutput
  print $ select @0 @(seqLen -1) lstmOutput

  return ()



-- initEmbedding paddingIdx numEmbeds = 
--   case someNatVal paddingIdx of
--     (SomeNat (Proxy :: Proxy paddingIdx)) -> case someNatVal numEmbeds of
--       (SomeNat (Proxy :: Proxy numEmbeds)) -> 
--         f @paddingIdx @numEmbeds
--   where
--     f ::
--       forall paddingIdx numEmbeds.(
--         paddingIdx <= numEmbeds,
--         1 <= numEmbeds - paddingIdx,
--         (((numEmbeds - paddingIdx) - 1) + (1 + paddingIdx)) ~ numEmbeds,
--         (paddingIdx + 1) <= numEmbeds,
--         KnownNat paddingIdx,
--         KnownNat numEmbeds
--       ) =>
--       -- Dict ((paddingIdx <=? numEmbeds) ~ 'True) -> Dict ((1 <=? numEmbeds) ~ 'True) -> Dict ((((numEmbeds - paddingIdx) - 1) + (1 + paddingIdx))  numEmbeds) -> Dict (((paddingIdx + 1) <= numEmbeds) ~ 1)
--       IO ()
--     f = do 
--       let embeddingSpec = LearnedEmbeddingWithRandomInitSpec @('Just paddingIdx) @numEmbeds @EmbedSize @'Float @ModelDevice
--       wordEmb <- sample embeddingSpec
--       print wordEmb



-- program paddingIdx numEmbeds = 
--   case someNatVal paddingIdx of
--     (SomeNat (Proxy :: Proxy paddingIdx)) -> case someNatVal numEmbeds of
--       (SomeNat (Proxy :: Proxy numEmbeds)) -> f @paddingIdx @numEmbeds 
--   where
--     f ::
--       forall paddingIdx numEmbeds shape shape'.(
--         KnownNat paddingIdx,
--         KnownNat numEmbeds,
--         paddingIdx <= numEmbeds,
--         1 <= numEmbeds - paddingIdx,
--         (((numEmbeds - paddingIdx) - 1) + (1 + paddingIdx)) ~ numEmbeds,
--         (paddingIdx + 1) <= numEmbeds,
--         shape' ~ Reverse (EmbedSize ': (Reverse shape))
--       ) =>
--       Tensor ModelDevice 'Int64 shape -> IO (Tensor ModelDevice 'Float shape')
--     f indices = do
--       wordEmb <- wordEmbIO
--       let input = embed wordEmb indices
--       return (input)
--       where
--         embeddingSpec = LearnedEmbeddingWithRandomInitSpec @('Just paddingIdx) @numEmbeds @EmbedSize @'Float @ModelDevice
--         wordEmbIO = sample embeddingSpec

-- -- メインのプログラムになる
-- go :: forall (n :: Nat) (paddingIdx :: Nat).(
--   -- paddingIdx <= n,
--   -- 1 <= n - paddingIdx,
--   -- (((n - paddingIdx) - 1) + (1 + paddingIdx)) ~ n,
--   -- (paddingIdx + 1) <= n,
--   KnownNat n,
--   KnownNat paddingIdx
--   -- KnownNat numEmbeds
--   -- KnownDevice device
--   ) => EmbeddingSpec (Just paddingIdx) n 256 'Learned 'Float '( 'CPU, 0) -> IO(Int)
-- go  = do 
--   let embeddingSpec = LearnedEmbeddingWithRandomInitSpec @('Just paddingIdx) @n @256 @'Float @'( 'CPU, 0)
--   wordEmb <- sample embeddingSpec
--   -- let a = embed wordEmb (zeros :: CPUTensor 'Int64 '[2,3])
-- -- -- モデル初期化
-- -- let spec = LSTMSpec @256 @128 @2 @'Bidirectional @'Float @'( 'CPU, 0) (DropoutSpec 0.1)
-- --     spec'' = LSTMWithLearnedInitSpec spec zeros zeros
-- -- model <- sample $ spec''
-- -- let output = lstmForwardWithDropout @'BatchFirst model a
-- -- print(output)
--   return (1)


  -- let embeddingSpec = LearnedEmbeddingWithRandomInitSpec @('Just 1) @numEmbeds @256 @'Float @'( 'CPU, 0)
  -- wordEmb <- sample embeddingSpec
  -- let a = embed wordEmb (zeros :: CPUTensor 'Int64 '[2,3])
-- -- モデル初期化
-- let spec = LSTMSpec @256 @128 @2 @'Bidirectional @'Float @'( 'CPU, 0) (DropoutSpec 0.1)
--     spec'' = LSTMWithLearnedInitSpec spec zeros zeros
-- model <- sample $ spec''
-- let output = lstmForwardWithDropout @'BatchFirst model a
-- print(output)
  

-- 最初に呼び出す
-- program :: IO ()
-- program = case someNatVal 10 of
--   (SomeNat proxy) -> case mkNumEmbedsProof proxy of
--     Just dict -> go dict
--     -- Nothing -> pure ()
--   where 
--     go :: forall (numEmbeds :: Nat).
--       KnownNat numEmbeds =>
--       Dict ((1 <=? numEmbeds) ~ 'True) ->
--       IO ()
--     go Dict = do
--       print 1
--       return ()

--     go :: forall (numEmbeds :: Nat).
--       KnownNat numEmbeds =>
--       Int ->
--       IO ()
--     go numEmbeds = do
--       print 1
--       return ()

-- withEmbedding :: 
--   (forall (numEmbeds :: Nat).
--   KnownNat numEmbeds =>
--   Data.Proxy.Proxy numEmbeds -> r) -> r
-- withEmbedding (p :: (Proxy numEmbeds)) f = 
--   if numEmbeds > 0 
--     then Just (f numEmbeds)
--     else Nothing
--   where
--     numEmbeds = natValI @numEmbeds

-- mkShapewordEmb wordEmb = do
--   let indices = [[0,1,2],[3,4,5]]
--       untypedTensor = D.asTensor (indices :: [[Int]])
--       shape = D.shape untypedTensor
--   return ()


learning ::
  D.Tensor ->
  IO()
learning untypedTensor = do
  let shape = D.shape untypedTensor
      batchSize = N.naturalFromInteger ((fromIntegral (shape !! 0))::Integer)
      seqLen = N.naturalFromInteger ((fromIntegral (shape !! 1))::Integer)
  model <- sample RNNGSpec
  case someShape (D.shape untypedTensor) of
    (SomeShape (Proxy :: Proxy shape)) -> case (someNatVal batchSize) of
      (SomeNat (Proxy :: Proxy batchSize)) -> case (someNatVal seqLen) of
        (SomeNat (Proxy :: Proxy seqLen)) ->
          rnng @shape @batchSize @seqLen model (UnsafeMkTensor @ModelDevice @'Int64 @shape untypedTensor)
  return ()
  where 
    f :: forall shape batchSize seqLen.
      ( KnownShape shape,
        KnownNat batchSize,
        KnownNat seqLen
        -- outputSize ~ (EmbedSize * NumberOfDirections Bidirectional),
        -- outputShape ~ RNNShape BatchFirst seqLen batchSize outputSize,
        -- hxShape ~ '[NumLayers * NumberOfDirections Bidirectional, batchSize, EmbedSize]
      ) =>
        Embedding ('Just 1) NumEmbed EmbedSize 'Learned 'Float ModelDevice ->
        LSTMWithInit EmbedSize HiddenSize NumLayers 'Bidirectional 'LearnedInitialization 'Float ModelDevice ->
        Tensor ModelDevice 'Int64 shape ->
        IO()
    f wordEmb model tensor = do
      let input = embed @('Just 1) @shape wordEmb tensor
          (a, b, c) = lstmForwardWithDropout @'BatchFirst @batchSize @seqLen model (unsafeCoerce input)
      print a
      return ()
{-
:seti -XTypeApplications -XDataKinds -XScopedTypeVariables -XAllowAmbiguousTypes -XKindSignatures -XFlexibleContexts -XGADTs -XTypeOperators　-XConstraintKinds
embeddingSpec = LearnedEmbeddingWithRandomInitSpec @('Just 1) @10 @10 @'Float @'( 'CPU, 0)
wordEmb <- sample $ embeddingSpec

untypedTensor = D.asTensor ( [[0,1,2],[3,4,5]] :: [[Int]])
input = embed wordEmb (UnsafeMkTensor @'( 'CPU, 0) @Int64 @'[2, 3] untypedTensor)

spec = LSTMSpec @10 @5 @2 @'Bidirectional @'Float @'( 'CPU, 0) (DropoutSpec 0.1)
spec'' = LSTMWithLearnedInitSpec spec zeros zeros
model <- sample $ spec''
(a, b, c) = lstmForwardWithDropout @'BatchFirst model input
-}


    --
      -- f @'( 'CPU, 0) @'[2,3] (UnsafeMkTensor @'( 'CPU, 0) @'Int64 @'[2,3] untypedTensor) wordEmb

main :: IO()
main = do
  -- hyper parameter
  config <- configLoad
  let batchSize = fromIntegral (getBatchSize config)::Int
      trial = fromIntegral (getTrial config)::Int
      iter = fromIntegral (getEpoch config)::Int
      embSize = 100
  -- data
  trainingData <- loadActionsFromBinary $ getTrainingDataPath config
  validationData <- loadActionsFromBinary $ getValidationDataPath config
  evaluationData <- loadActionsFromBinary $ getEvaluationDataPath config

  -- 出現頻度が3回以上の語彙に絞る
  -- let vocab = buildVocab evaluationData 3
  --     (wordIndexFor, wordEmbDim) = indexFactory vocab True
  --     (actionIndexFor, actionEmbDim) = indexFactory ["SHIFT", "REDUCE", "NT"] True
  --     embeddingSpec = LearnedEmbeddingWithRandomInitSpec @('Just 1) @NumEmbed @EmbedSize @'Float @ModelDevice
  --     batches = chunksOf batchSize trainingData
  -- 単語埋め込み
  -- wordEmb <- sample $ embeddingSpec
  -- モデル初期化
  -- let spec = LSTMSpec @EmbedSize @HiddenSize @NumLayers @'Bidirectional @'Float @ModelDevice (DropoutSpec 0.1)
  --     spec'' = LSTMWithLearnedInitSpec spec zeros zeros

  -- initModel <- sample $ spec''
  -- ((trainedModel,_),losses) <- mapAccumM [1..iter] (initModel, GD) $ \epoc (model,opt) -> do
  --   (updated, batchLosses) <- mapAccumM batches (model, opt) $ \batch (model',opt') -> do
  --     let (words, actions) = unzip $ fmap unpackRNNGSentence batch
  --         indices = indexForBatch wordIndexFor words
  --         teacher = indexForBatch actionIndexFor $ fmap (fmap actionDic) actions
  --         indicesTensor = D.asTensor (indices :: [[Int]])
  --     -- learning wordEmb model indicesTensor
  --         -- feedForward model' input
  --         -- input = embed wordEmb $ UnsafeMkTensor $ D.asTensor (indices :: [[Int]])
  --         -- lstmForward
  --     -- print index
  --     return ((model', opt'), 0::Float)
  --   return ((model,opt), 0::Float)
  return ()

-- feedForward :: LSTMWithInit inputSize hiddenSize numLayers directionality initialization dtype device -> D.Tensor -> IO()
-- feedForward wordEmb model indices = do
--   withTensor indices go
--   where
--     go :: Tensor device dtype shape -> IO()
--     go indices = do
--       let input = embed wordEmb indices
--       print (input)
--       let (a, b, c) = lstmForwardWithDropout @'BatchFirst model input
--       -- print (a, b, c)
--       return ()