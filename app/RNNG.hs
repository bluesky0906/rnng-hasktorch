{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
-- :seti -XTypeApplications -XDataKinds -XScopedTypeVariables -XAllowAmbiguousTypes -XKindSignatures
module RNNG where
import PTB
import Util
import Data.List.Split (chunksOf, splitEvery) --split
import Data.Proxy
import Torch.Control (mapAccumM)
import GHC.TypeNats
import Torch.Typed
import Torch.Tensor (asTensor)
import Unsafe.Coerce


mkEmbedSizeProof ::
  forall (n :: Nat)
         (paddingIdx :: Nat)
         (numEmbeds :: Nat)
         (embedSize :: Nat)
         (embeddingType :: EmbeddingType)
         (dtype :: DType)
         (device :: (DeviceType, Nat))
         p.
  Proxy n -> Maybe (EmbeddingSpec (Just paddingIdx) numEmbeds n embeddingType dtype device)
mkEmbedSizeProof (proxy :: (Proxy n)) = Just (unsafeCoerce $ LearnedEmbeddingWithRandomInitSpec @('Just 1) @3 @n @'Float @'( 'CPU, 0))

-- メインのプログラムになる
go :: forall (n :: Nat).(
  -- Proxy n
  KnownNat n
  -- KnownNat paddingIdx
  -- KnownNat numEmbeds
  -- KnownDevice device
  ) => EmbeddingSpec (Just 1) 3 n 'Learned 'Float '( 'CPU, 0) -> IO(Int)
go embeddingSpec = do 
  init <- sample $ embeddingSpec
  return (1)

-- 最初に呼び出す
program :: Integer-> IO (Int)
program embSize = case GHC.TypeLits.someNatVal embSize of
    Just (SomeNat (proxy)) -> case mkEmbedSizeProof proxy of
      Just embeddingSpec -> go embeddingSpec

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
  let vocab = buildVocab evaluationData 3
      (wordIndexFor, wordEmbDim) = indexFactory vocab True
      (actionIndexFor, actionEmbDim) = indexFactory ["SHIFT", "REDUCE", "NT"] True
      -- embeddingSpec = case GHC.TypeNats.someNatVal embSize of
      --   SomeNat (proxy :: Proxy p) -> LearnedEmbeddingWithRandomInitSpec @('Just 1) @3 @p @'Float @'( 'CPU, 0)
      embeddingSpec = LearnedEmbeddingWithRandomInitSpec @('Just 1) @2006 @100 @'Float @'( 'CPU, 0)
      batches = chunksOf batchSize trainingData
  -- 単語埋め込み
  wordEmb <- sample $ embeddingSpec
  -- モデル初期化
  let spec = LSTMSpec @100 @128 @2 @'Bidirectional @'Float @'( 'CPU, 0) (DropoutSpec 0.1)
      spec'' = LSTMWithLearnedInitSpec spec zeros zeros

  initModel <- sample $ spec''
  ((trainedModel,_),losses) <- mapAccumM [1..iter] (initModel, GD) $ \epoc (model,opt) -> do
    (updated, batchLosses) <- mapAccumM batches (model, opt) $ \batch (model',opt') -> do
      let (words, actions) = unzip $ fmap unpackRNNGSentence batch
          indices = indexForBatch wordIndexFor words
          teacher = indexForBatch actionIndexFor $ fmap (fmap actionDic) actions
          input = embed wordEmb $ UnsafeMkTensor $ asTensor (indices :: [[Int]])
          -- lstmForward
      -- print index
      return ((model', opt'), 0::Float)
    return ((model,opt), 0::Float)
  return ()

