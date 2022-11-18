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
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE RankNTypes #-}

-- {-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
-- {-# OPTIONS_GHC -fdefer-type-errors #-}

-- :seti -XTypeApplications -XDataKinds -XScopedTypeVariables -XAllowAmbiguousTypes -XKindSignatures -XFlexibleContexts -XGADTs -XTypeOperators　-XConstraintKinds -XRecordWildCards -XNoStarIsType

module RNNG where
import RNN
import PTB
import Util
import Data.List.Split (chunksOf, splitEvery) --split
import Data.Proxy
import Torch.Control (mapAccumM)
import Torch.Train (update)
import GHC.Generics
import GHC.TypeNats
import GHC.Exts
import Torch.HList
import qualified GHC.Natural as N
import Torch.Typed
import qualified Torch.Tensor as D
import Unsafe.Coerce
import Data.Constraint

type PaddingIdx = 'Nothing :: Maybe Nat

type EmbedSize = 10

type HiddenSize = 5

type NumEmbed = 2006

type NumLayers = 2

type ModelDevice = '( 'CPU, 0)

data RNNGSpec 
  = RNNGSpec
  deriving (Show, Eq)

data RNNG where
  RNNG ::
    {
      wordEmbedding :: Embedding PaddingIdx NumEmbed EmbedSize 'Learned 'Float ModelDevice,
      actionEmbedding :: Embedding PaddingIdx 150 EmbedSize 'Learned 'Float ModelDevice,
      bufferRNN :: RNN EmbedSize EmbedSize 'Float ModelDevice,
      historyRNN :: RNN EmbedSize EmbedSize 'Float ModelDevice,
      stackLSTM :: LSTMWithInit EmbedSize HiddenSize NumLayers 'Bidirectional 'LearnedInitialization 'Float ModelDevice,
      compLSTM :: LSTMWithInit EmbedSize HiddenSize NumLayers 'Bidirectional 'LearnedInitialization 'Float ModelDevice,
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
        <$> sample (LearnedEmbeddingWithRandomInitSpec @PaddingIdx @NumEmbed @EmbedSize @'Float @ModelDevice)
        <*> sample (LearnedEmbeddingWithRandomInitSpec @PaddingIdx @150 @EmbedSize @'Float @ModelDevice)
        <*> sample (RNNSpec @EmbedSize @EmbedSize @'Float @ModelDevice)
        <*> sample (RNNSpec @EmbedSize @EmbedSize @'Float @ModelDevice)
        <*> sample (LSTMWithLearnedInitSpec (
                      LSTMSpec @EmbedSize @HiddenSize @NumLayers @'Bidirectional @'Float @ModelDevice (DropoutSpec 0.1)
                    ) zeros zeros)
        <*> sample (LSTMWithLearnedInitSpec (
                      LSTMSpec @EmbedSize @HiddenSize @NumLayers @'Bidirectional @'Float @ModelDevice (DropoutSpec 0.1)
                    ) zeros zeros)
        <*> sample (LinearSpec @(HiddenSize * (NumberOfDirections 'Bidirectional)) @3 @'Float @ModelDevice)

data RNNGData where
  RNNGData ::
    {
      sentence :: Sentence,
      sentenceIdx :: [Tensor ModelDevice 'Int64 '[]],
      actions :: [Action],
      actionsIdx :: [Tensor ModelDevice 'Int64 '[]]
    } ->
    RNNGData
  deriving (Show, Generic)

-- rnng :: forall shape dim1.
--   ( KnownShape shape,
--     KnownNat dim1
--   ) =>
--   -- | input as index
--   Tensor ModelDevice 'Int64 shape ->
--   -- | actions
--   [Action] ->
--   -- Tensor ModelDevice 'Int64 shape ->
--   RNNG ->
--   -- (Tensor ModelDevice 'Int64 _, Tensor ModelDevice 'Float '[])
--   IO()
-- rnng indices actions RNNG {..} = do
--   let input = embed @PaddingIdx @shape embedding indices
--       (lstmOutput, _, _) = lstmForwardWithoutDropout @'SequenceFirst @1 @dim1 stackLSTM (unsafeCoerce input)
--       output = select @1 @0 $ logSoftmax @2 $ forward linear lstmOutput
--       -- loss = nllLoss @ReduceNone @dim1 @3 @'[] (ones @'[3] @'Float @ModelDevice) (-100) output (unsafeCoerce actions)
--   print output
toHList (first:[]) = HNil
toHList (first:rest) = 
  case fromList [first] :: (Maybe (HList '[Tensor ModelDevice 'Float '[EmbedSize]])) of 
    Just x -> x :. (toHList rest)

parse ::
  RNNG ->
  [Action] ->
  [Tensor ModelDevice 'Float '[EmbedSize]] ->
  [Tensor ModelDevice 'Float '[EmbedSize]] ->
  IO ()
parse RNNG {..} ((NT label):rest) stack buffer = do
  let (_, b_embedding) = rnnForward bufferRNN buffer
  print $ ((fromList ([head buffer])) :: (Maybe (HList '[Tensor ModelDevice 'Float '[EmbedSize]])))
      -- (_, s_embedding) = rnnForward stackLSTM buffer
  print b_embedding
  -- print s_embedding
  return ()

rnng ::
  -- | input
  RNNGData ->
  -- Tensor ModelDevice 'Int64 shape ->
  RNNG ->
  -- (Tensor ModelDevice 'Int64 _, Tensor ModelDevice 'Float '[])
  IO()
rnng RNNGData{..} RNNG {..} = do
  -- [Torch.Typed.Tensor]
  let initBuffer = fmap (embed wordEmbedding) sentenceIdx
      -- actions = fmap (embed @PaddingIdx @'[EmbedSize] actionEmbedding) actionsIdx
  parse RNNG {..} actions [] initBuffer
      -- (lstmOutput, _, _) = lstmForwardWithoutDropout @'SequenceFirst @1 @dim1 stackLSTM (unsafeCoerce input)
      -- output = select @1 @0 $ logSoftmax @2 $ forward linear lstmOutput
      -- loss = nllLoss @ReduceNone @dim1 @3 @'[] (ones @'[3] @'Float @ModelDevice) (-100) output (unsafeCoerce actions)
  -- print output
  return ()

learning ::
  -- | input data
  RNNGData ->
  -- | model
  RNNG -> 
  -- IO(Tensor ModelDevice 'Float '[])
  IO ()
learning RNNGData {..} model = do
  rnng RNNGData {..} model
  -- let shape = D.shape sentenceIdx
  --     seqLen = N.naturalFromInteger ((fromIntegral (shape !! 1))::Integer)
  -- case someShape (D.shape sentenceIdx) of
  --   SomeShape (Proxy :: Proxy shape) -> case (someNatVal seqLen) of
  --       (SomeNat (Proxy :: Proxy seqLen)) ->
  --         rnng @shape @seqLen (UnsafeMkTensor @ModelDevice @'Int64 @shape input) (UnsafeMkTensor @ModelDevice @'Int64 @shape actions) model
  return ()

{-
:seti -XTypeApplications -XDataKinds -XScopedTypeVariables -XAllowAmbiguousTypes -XKindSignatures -XFlexibleContexts -XGADTs -XTypeOperators　-XConstraintKinds
embeddingSpec = LearnedEmbeddingWithRandomInitSpec @PaddingIdx @10 @10 @'Float @'( 'CPU, 0)
wordEmb <- sample $ embeddingSpec

untypedTensor = D.asTensor ( [0,1,2, 3,4,5] :: [Int])
input = embed wordEmb (UnsafeMkTensor @'( 'CPU, 0) @Int64 @'[2, 3] untypedTensor)

spec = LSTMSpec @10 @5 @2 @'Bidirectional @'Float @'( 'CPU, 0) (DropoutSpec 0.1)
spec'' = LSTMWithLearnedInitSpec spec zeros zeros
model <- sample $ spec''
(a, b, c) = lstmForwardWithDropout @'BatchFirst model input
-}


main :: IO()
main = do
  -- hyper parameter
  config <- configLoad
  let trial = fromIntegral (getTrial config)::Int
      iter = fromIntegral (getEpoch config)::Int
      embSize = 100
  -- data
  trainingData <- loadActionsFromBinary $ getTrainingDataPath config
  validationData <- loadActionsFromBinary $ getValidationDataPath config
  evaluationData <- loadActionsFromBinary $ getEvaluationDataPath config

  -- 出現頻度が3回以上の語彙に絞る
  let (wordIndexFor, wordEmbDim) = indexFactory (buildVocab evaluationData 3 toWordList) False
      (actionIndexFor, actionEmbDim) = indexFactory (buildVocab evaluationData 1 toActionList) False
      -- (ntIndexFor, ntEmbDim) = indexFactory (buildVocab evaluationData 1 toNTList) False
  print (head $ evaluationData)
  initModel <- sample RNNGSpec
  ((trainedModel, _),losses) <- mapAccumM [1..iter] (initModel, GD) $ \epoc (model,opt) -> do
    (updated, loss) <- mapAccumM evaluationData (model, opt) $ \batch (model',opt') -> do
      let (words, actions) = unpackRNNGSentence batch
          indices = D.asTensor $ fmap wordIndexFor words
          -- actionIndices = D.asTensor $ fmap actionIndexFor $ fmap actionDic actions
      -- loss <- learning indices actions initModel
      -- learning indices actionIndices initModel
      -- u <- runStep model' opt' loss 5e-4
      return ((model',opt'), 0::Float)
    return (updated, loss)
  return ()

{-
:seti -XTypeApplications -XDataKinds -XScopedTypeVariables -XAllowAmbiguousTypes -XKindSignatures -XFlexibleContexts -XGADTs -XTypeOperators　-XConstraintKinds -XRecordWildCards -XNoStarIsType
config <- configLoad

evaluationData <- loadActionsFromBinary $ getEvaluationDataPath config
vocab = buildVocab evaluationData 3
(wordIndexFor, wordEmbDim) = indexFactory (buildVocab evaluationData 3 toWordList) False
(actionIndexFor, actionEmbDim) = indexFactory (buildVocab evaluationData 1 toActionList) False

initModel <- sample RNNGSpec

RNNGSentence (sent, acts) = head evaluationData
sentIdx = fmap (UnsafeMkTensor @ModelDevice @'Int64 @'[] . D.asTensor . wordIndexFor) sent
actsIdx = fmap (UnsafeMkTensor @ModelDevice @'Int64 @'[] . D.asTensor . actionIndexFor) (fmap showAction acts)
rnngData = RNNGData {sentence = sent, sentenceIdx = sentIdx, actions = acts, actionsIdx = actsIdx}

rnng rnngData initModel

initBuffer = fmap (embed (wordEmbedding initModel)) (sentenceIdx rnngData)
(fromList ([head initBuffer])) :: (Maybe (HList '[Tensor ModelDevice 'Float '[EmbedSize]]))
-}

