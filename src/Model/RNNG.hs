{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE StandaloneDeriving #-}

module Model.RNNG where
import Data.CFG
import Data.RNNGSentence
import Torch hiding (take)
-- | hasktorch-tools
import Torch.Layer.RNN (RnnHypParams(..), RnnParams, rnnLayers)
import Torch.Layer.LSTM (LstmHypParams(..), LstmParams, lstmLayers)
import Torch.Layer.Linear (LinearHypParams(..), LinearParams, linearLayer)
import Torch.Layer.NonLinear (ActName(..))
import GHC.Generics
import qualified Data.Text as T
import qualified Data.Map as M
import Data.List
import Data.Binary
import Debug.Trace

deriving instance Generic DeviceType
deriving instance Binary DeviceType
deriving instance Generic Device
deriving instance Binary Device

data RNNGSpec = RNNGSpec {
  modelDevice :: Device,
  modelPosMode :: Bool,
  numLayers :: Int,
  wordEmbedSize :: Int,
  actionEmbedSize :: Int,
  wordNumEmbed :: Int,
  actionNumEmbed :: Int,
  ntNumEmbed :: Int,
  hiddenSize :: Int
} deriving (Show, Eq, Generic, Binary)


{-

RNNG model

-}


data ParsingMode = Point | All deriving (Show)


data PredictActionRNNG where
  PredictActionRNNG :: {
    w :: Parameter,
    c :: Parameter,
    linearParams :: LinearParams
    } ->
    PredictActionRNNG
  deriving (Show, Generic, Parameterized)

data ParseRNNG where
  ParseRNNG ::
    {
      wordEmbedding :: Parameter,
      ntEmbedding :: Parameter,
      actionEmbedding :: Parameter,
      bufferRNN :: RnnParams,
      bufferh0 :: Parameter,
      stackLSTM :: LstmParams,
      stackh0 :: Parameter,
      stackc0 :: Parameter,
      actionRNN :: RnnParams,
      actionh0 :: Parameter,
      actionStart :: Parameter, -- dummy for empty action history
      bufferGuard :: Parameter, -- dummy for empty buffer
      stackGuard :: Parameter  -- dummy for empty stack
    } ->
    ParseRNNG
  deriving (Show, Generic, Parameterized)

data CompRNNG where
  CompRNNG ::
    {
      compForLSTM :: LstmParams,
      compForh0 :: Parameter,
      compForc0 :: Parameter,
      compRevLSTM :: LstmParams,
      compRevh0 :: Parameter,
      compRevc0 :: Parameter,
      compW :: Parameter,
      compC :: Parameter
    } ->
    CompRNNG
  deriving (Show, Generic, Parameterized)


data RNNG where
  RNNG ::
    {
      predictActionRNNG :: PredictActionRNNG,
      parseRNNG :: ParseRNNG,
      compRNNG :: CompRNNG
    } ->
    RNNG
  deriving (Show, Generic, Parameterized)


instance
  Randomizable
    RNNGSpec
    RNNG
  where
    sample RNNGSpec {..} = do
      predictActionRNNG <- PredictActionRNNG
        <$> (makeIndependent =<< randnIO' [hiddenSize * 3, hiddenSize])
        <*> (makeIndependent =<< randnIO' [hiddenSize])
        <*> sample (LinearHypParams modelDevice True hiddenSize actionNumEmbed)
      parseRNNG <- ParseRNNG
        -- wordEmbedding
        <$> (makeIndependent =<< randnIO' [wordNumEmbed, wordEmbedSize])
        -- ntEmbedding
        <*> (makeIndependent =<< randnIO' [ntNumEmbed, wordEmbedSize])
        -- actionEmbedding
        <*> (makeIndependent =<< randnIO' [actionNumEmbed, wordEmbedSize])
        -- bufferRNN
        <*> sample (RnnHypParams modelDevice False actionEmbedSize hiddenSize numLayers True)
        -- bufferh0
        <*> (makeIndependent =<< randnIO' [numLayers, hiddenSize])
        -- stackLSTM
        <*> sample (LstmHypParams modelDevice False hiddenSize hiddenSize numLayers True Nothing)
        -- stackh0
        <*> (makeIndependent =<< randnIO' [numLayers, hiddenSize])
        -- stackc0
        <*> (makeIndependent =<< randnIO' [numLayers, hiddenSize])
        -- actionRNN
        <*> sample (RnnHypParams modelDevice False wordEmbedSize hiddenSize numLayers True)
        -- actionh0
        <*> (makeIndependent =<< randnIO' [numLayers, hiddenSize])
        -- actionStart
        <*> (makeIndependent =<< randnIO' [actionEmbedSize])
        -- bufferGuard
        <*> (makeIndependent =<< randnIO' [wordEmbedSize])
        -- stackGuard
        <*> (makeIndependent =<< randnIO' [wordEmbedSize])
      compRNNG <- CompRNNG
        -- dev, bidirectional, inputSize, hiddenSize, numLayers, hasBias, projSize
        <$> sample (LstmHypParams modelDevice False hiddenSize hiddenSize numLayers True Nothing)
        <*> (makeIndependent =<< randnIO' [numLayers, hiddenSize])
        <*> (makeIndependent =<< randnIO' [numLayers, hiddenSize])
        <*> sample (LstmHypParams modelDevice False hiddenSize hiddenSize numLayers True Nothing)
        <*> (makeIndependent =<< randnIO' [numLayers, hiddenSize])
        <*> (makeIndependent =<< randnIO' [numLayers, hiddenSize])
        -- compW
        <*> (makeIndependent =<< randnIO' [hiddenSize * 2, hiddenSize])
        -- compC
        <*> (makeIndependent =<< randnIO' [hiddenSize])
      return $ RNNG predictActionRNNG parseRNNG compRNNG



{-

  Data Structure for RNNG

-}

data IndexData = IndexData {
    wordIndexFor :: T.Text -> Int,
    indexWordFor :: Int -> T.Text,
    actionIndexFor :: Action -> Int,
    indexActionFor :: Int -> Action,
    ntIndexFor :: T.Text -> Int,
    indexNTFor :: Int -> T.Text
  }

data RNNGState where
  RNNGState ::
    {
      -- | stackをTensor<numLayer, hidden>で保存. 逆順で積まれる
      stack :: [Tensor],
      -- | stackを文字列で保存. 逆順で積まれる
      textStack :: [T.Text],
      -- | TODO: (h, c)の順にする（hasktorch-toolsに追従）
      -- | stack lstmの隠れ状態の記録　(h, c)
      hiddenStack :: [(Tensor, Tensor)],

      -- | bufferをTensor<hidden>で保存. 正順で積まれる
      buffer :: [Tensor],
      -- | bufferを文字列で保存. 正順で積まれる
      textBuffer :: [T.Text],

      -- | action historyをTensor<numLayer, hidden>で保存. 逆順で積まれる
      actionHistory :: [Tensor],
      -- | action historyを文字列で保存. 逆順で積まれる
      textActionHistory :: [Action],
      -- | 現在のaction historyの隠れ層. pushのみ行われるので最終層だけ常に保存.
      hiddenActionHistory :: Tensor,

      -- | 開いているNTの数
      numOpenParen :: Int
    } ->
    RNNGState 

instance Show RNNGState where
  show RNNGState {..} = unlines [
      "textStack: " ++ show textStack,
      "textBuffer: " ++ show textBuffer,
      "textActionHistory: " ++ show textActionHistory,
      "numOpenParen: " ++ show numOpenParen
    ]

data Mode where
  Mode ::
    {
      device :: Device,
      dropoutProb :: Maybe Double,
      parsingMode :: ParsingMode,
      posMode :: Bool
    } ->
    Mode

{-

mask Forbidden actions

-}

checkNTForbidden ::
  Mode ->
  RNNGState ->
  Bool
checkNTForbidden Mode{..} RNNGState {..} =
  numOpenParen > 100 -- 開いているNTが多すぎる時
  || null textBuffer-- 単語が残っていない時
  || if posMode 
      then not (null textActionHistory) && (head textActionHistory == SHIFT) -- 1番初めではなくて、前のアクションがSHIFTの時
      else False

checkREDUCEForbidden ::
  Mode ->
  RNNGState ->
  Bool
checkREDUCEForbidden _ RNNGState {..} =
  length textStack < 2  -- first action must be NT and don't predict POS
  || (numOpenParen == 1 && not (null textBuffer)) -- bufferに単語が残ってるのに木を一つにまとめることはできない
  || ((previousAction /= SHIFT) && (previousAction /= REDUCE)) && (previousAction /= ERROR) -- can't REDUCE after NT
  where 
    previousAction = if not (null textActionHistory)
                      then head textActionHistory
                      else ERROR

checkSHIFTForbidden ::
  Mode ->
  RNNGState ->
  Bool
checkSHIFTForbidden Mode{..} RNNGState{..} =
  null textStack -- first action must be NT
  || null textBuffer -- Buffer isn't empty
  || if posMode && not (null textStack)
      then (previousAction == SHIFT) || (previousAction == REDUCE) || (previousAction == ERROR) -- SHIFT only after NT
      else False
  where 
    previousAction = if not (null textActionHistory)
                      then head textActionHistory
                      else ERROR

maskTensor :: 
  Mode ->
  RNNGState ->
  [Action] ->
  Tensor
maskTensor mode@Mode{..} rnngState actions = toDevice device $ asTensor $ map mask actions
  where 
    ntForbidden = checkNTForbidden mode rnngState
    reduceForbidden = checkREDUCEForbidden mode rnngState
    shiftForbidden = checkSHIFTForbidden mode rnngState
    mask :: Action -> Bool
    mask (NT _) = ntForbidden
    mask REDUCE = reduceForbidden
    mask SHIFT = shiftForbidden

maskImpossibleAction ::
  Mode ->
  Tensor ->
  RNNGState ->
  IndexData ->
  Tensor
maskImpossibleAction mode prediction rnngState IndexData {..}  =
  let actions = map indexActionFor [0..((shape prediction !! 0) - 1)]
      boolTensor = maskTensor mode rnngState actions
  in maskedFill prediction boolTensor (-1e38::Float)

extractLastLayerTensor ::
  -- | an Tensor of <m, n>
  Tensor ->
  -- | an Tensor of <n>
  Tensor
extractLastLayerTensor tensor = tensor ! (head (shape tensor) - 1) 


{-

RNNG Forward

-}

predictNextAction ::
  Mode -> 
  RNNG ->
  -- | data
  RNNGState ->
  IndexData ->
  -- | possibility of actions
  Tensor
predictNextAction mode@Mode{..} (RNNG PredictActionRNNG {..} _ _) RNNGState {..} indexData = 
  let bufferEmbedding = head buffer
      -- | 最終層のhnを取り出す
      stackEmbedding = extractLastLayerTensor $ fst $ head hiddenStack
      actionEmbedding = extractLastLayerTensor hiddenActionHistory
      -- | ut + tanh[W[ot, st, ht] + c]
      ut = Torch.tanh (cat (Dim 0) [stackEmbedding, bufferEmbedding, actionEmbedding] `matmul` toDependent w + toDependent c)
      actionLogit = linearLayer linearParams ut
      maskedAction = maskImpossibleAction mode actionLogit RNNGState {..} indexData
  in logSoftmax (Dim 0) maskedAction

stackLSTMForward ::
  Mode ->
  LstmParams ->
  -- | (hi, ci)
  (Tensor, Tensor) ->
  -- | new Elements
  [Tensor] ->
  -- | (hn, cn)
  (Tensor, Tensor)
stackLSTMForward Mode{..} stackLSTM stack newElem = snd $ lstmLayers stackLSTM dropoutProb stack (Torch.stack (Dim 0) newElem)

actionRNNForward ::
  Mode ->
  RnnParams ->
  -- | h_n
  Tensor ->
  -- | new Elements
  [Tensor] ->
  -- | h_n+1
  Tensor
actionRNNForward Mode{..} actionRNN hn newElem = snd $ rnnLayers actionRNN Tanh dropoutProb hn (Torch.stack (Dim 0) newElem)

parse ::
  Mode ->
  -- | model
  RNNG ->
  IndexData ->
  -- | new RNNGState
  RNNGState ->
  Action ->
  RNNGState
parse mode@Mode {..} (RNNG _ ParseRNNG {..} _) IndexData {..} RNNGState {..} (NT label) =
  let nt_embedding = embedding' (toDependent ntEmbedding) ((toDevice device . asTensor . ntIndexFor) label)
      textAction = NT label
      action_embedding = embedding' (toDependent actionEmbedding) ((toDevice device . asTensor . actionIndexFor) textAction)
  in RNNGState {
      stack = nt_embedding:stack,
      textStack = (T.pack "<" `T.append` label):textStack,
      hiddenStack = stackLSTMForward mode stackLSTM (head hiddenStack) [nt_embedding]:hiddenStack,
      buffer = buffer,
      textBuffer = textBuffer,
      actionHistory = action_embedding:actionHistory,
      textActionHistory = textAction:textActionHistory,
      hiddenActionHistory = actionRNNForward mode actionRNN hiddenActionHistory [action_embedding],
      numOpenParen = numOpenParen + 1
    }
parse mode@Mode {..} (RNNG _ ParseRNNG {..} _) IndexData {..} RNNGState {..} SHIFT =
  let textAction = SHIFT
      action_embedding = embedding' (toDependent actionEmbedding) ((toDevice device . asTensor . actionIndexFor) textAction)
  in RNNGState {
      stack = head buffer:stack,
      textStack = head textBuffer:textStack,
      hiddenStack = stackLSTMForward mode stackLSTM (head hiddenStack) [head buffer]:hiddenStack,
      buffer = tail buffer,
      textBuffer = tail textBuffer,
      actionHistory = action_embedding:actionHistory,
      textActionHistory = textAction:textActionHistory,
      hiddenActionHistory = actionRNNForward mode actionRNN hiddenActionHistory [action_embedding],
      numOpenParen = numOpenParen
    }
parse mode@Mode {..} (RNNG _ ParseRNNG {..} CompRNNG {..}) IndexData {..} RNNGState {..} REDUCE =
  let textAction = REDUCE
      action_embedding = embedding' (toDependent actionEmbedding) ((toDevice device . asTensor . actionIndexFor) textAction)
      -- | 開いたlabelのidxを特定する
      (Just idx) = findIndex (\elem -> T.isPrefixOf (T.pack "<") elem && not (T.isSuffixOf (T.pack ">") elem)) textStack
      -- | popする
      (textSubTree, newTextStack) = splitAt (idx + 1) textStack
      (subTree, newStack) = splitAt (idx + 1) stack
      (_, newHiddenStack) = splitAt (idx + 1) hiddenStack
      label = last subTree
      words = tail subTree
      -- composeする
      -- 最終層のfwdとrevをconcatしてaffine変換する
      composedForhn = select 0 1 $ fst $ lstmLayers compForLSTM dropoutProb (toDependent compForh0, toDependent compForc0) (Torch.stack (Dim 0) $ label:reverse subTree)
      composedRevhn = select 0 1 $ fst $ lstmLayers compRevLSTM dropoutProb (toDependent compRevh0, toDependent compRevc0) (Torch.stack (Dim 0) $ label:subTree)
      lastLayerhn = Torch.cat (Dim 0) [composedForhn, composedRevhn]
      composedSubTree = lastLayerhn `matmul` toDependent compW + toDependent compC
  in RNNGState {
      stack = composedSubTree:newStack,
      textStack = T.intercalate (T.pack " ") (reverse $ T.pack ">":textSubTree):newTextStack,
      hiddenStack = stackLSTMForward mode stackLSTM (head newHiddenStack) [composedSubTree]:newHiddenStack,
      buffer = buffer,
      textBuffer = textBuffer,
      actionHistory = action_embedding:actionHistory,
      textActionHistory = textAction:textActionHistory,
      hiddenActionHistory = actionRNNForward mode actionRNN hiddenActionHistory [action_embedding],
      numOpenParen = numOpenParen - 1
    }

predict ::
  Mode ->
  RNNG ->
  IndexData ->
  -- | actions
  [Action] ->
  -- | predicted history
  [Tensor] ->
  -- | init RNNGState
  RNNGState ->
  -- | (predicted actions, new rnngState)
  ([Tensor], RNNGState)
predict mode@(Mode _ _ Point _) _ _ [] results rnngState = (reverse results, rnngState)
predict mode@(Mode _ _ Point _) rnng indexData (action:rest) predictionHitory rnngState =
  let prediction = predictNextAction mode rnng rnngState indexData
      newRNNGState = parse mode rnng indexData rnngState action
  in predict mode rnng indexData rest (prediction:predictionHitory) newRNNGState

predict mode@(Mode _ _ All _) rnng IndexData {..} _ predictionHitory RNNGState {..} =
  if (length textStack == 1) && (length textBuffer == 0)
    then (reverse predictionHitory, RNNGState {..} )
  else
    let prediction = predictNextAction mode rnng RNNGState {..} IndexData {..}
        action = indexActionFor (asValue (argmax (Dim 0) RemoveDim prediction)::Int)
        newRNNGState = parse mode rnng IndexData {..} RNNGState {..} action
    in predict mode rnng IndexData {..}  [] (prediction:predictionHitory) newRNNGState


rnngForward ::
  Mode ->
  RNNG ->
  -- | functions to convert text to index
  IndexData ->
  RNNGSentence ->
  [Tensor]
rnngForward mode@Mode{..} (RNNG predictActionRNNG ParseRNNG {..} compRNNG) IndexData {..} (RNNGSentence (sents, actions)) =
  let sentsTensor = Torch.stack (Dim 0) $ toDependent bufferGuard : fmap (embedding' (toDependent wordEmbedding) . toDevice device . asTensor . wordIndexFor) (reverse sents)
      initRNNGState = RNNGState {
        stack = [toDependent stackGuard],
        textStack = [],
        hiddenStack = [stackLSTMForward mode stackLSTM (toDependent stackh0, toDependent stackc0) [toDependent stackGuard]],
        buffer = reverse $ fmap (reshape [shape sentsTensor !! 1]) $ split 1 (Dim 0) $ fst $ rnnLayers bufferRNN Tanh dropoutProb (toDependent bufferh0) sentsTensor,
        textBuffer = sents,
        actionHistory = [toDependent actionStart],
        textActionHistory = [],
        hiddenActionHistory = actionRNNForward mode actionRNN (toDependent actionh0) [(toDependent actionStart)],
        numOpenParen = 0
      }
  in fst $ predict mode (RNNG predictActionRNNG ParseRNNG {..} compRNNG) IndexData {..} actions [] initRNNGState
