{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE GADTs #-}


module RNNG where
import PTB
import Util
import Torch hiding (foldLoop)
-- | hasktorch-tools
import Torch.Layer.RNN (RNNHypParams(..), RNNParams, rnnLayer)
import Torch.Layer.BiLSTM (BiLstmHypParams(..), BiLstmParams, biLstmLayers)
import Torch.Layer.LSTM (LstmHypParams(..), LstmParams, lstmLayer)
import Torch.Layer.Linear
import Torch.Control (mapAccumM, foldLoop)
import Torch.Train (update, saveParams, loadParams)
-- | nlp-tools
import ML.Exp.Chart (drawLearningCurve)

import GHC.Generics
import qualified Data.Text as T 
import Data.List
import Debug.Trace



data RNNGSpec = RNNGSpec {
  modelDevice :: Device,
  wordEmbedSize :: Int,
  actionEmbedSize :: Int,
  wordNumEmbed :: Int,
  actionNumEmbed :: Int,
  ntNumEmbed :: Int,
  hiddenSize :: Int
} deriving (Show, Eq)


data PredictActionRNNG where
  PredictActionRNNG :: {
    bufferRNN :: RNNParams,
    actionRNN :: RNNParams,
    h0 :: Parameter,
    w :: Parameter,
    c :: Parameter,
    linearParams :: LinearParams
    } ->
    PredictActionRNNG
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

data ParseRNNG where
  ParseRNNG ::
    {
      wordEmbedding :: Parameter,
      ntEmbedding :: Parameter,
      actionEmbedding :: Parameter,
      stackLSTM :: LstmParams,
      stackh0 :: Parameter,
      stackc0 :: Parameter,
      actionStart :: Parameter, -- dummy for empty action history
      bufferGuard :: Parameter, -- dummy for empty buffer
      stackGuard :: Parameter  -- dummy for empty stack
    } ->
    ParseRNNG
  deriving (Show, Generic, Parameterized)

data CompRNNG where
  CompRNNG ::
    {
      compLSTM :: BiLstmParams,
      comph0 :: Parameter,
      compc0 :: Parameter
    } ->
    CompRNNG
  deriving (Show, Generic, Parameterized)

instance
  Randomizable
    RNNGSpec
    RNNG
  where
    sample RNNGSpec {..} = do
      predictActionRNNG <- PredictActionRNNG
        <$> sample (RNNHypParams modelDevice wordEmbedSize hiddenSize)
        <*> sample (RNNHypParams modelDevice wordEmbedSize hiddenSize)
        <*> (makeIndependent =<< randnIO' [hiddenSize])
        <*> (makeIndependent =<< randnIO' [hiddenSize * 3, hiddenSize])
        <*> (makeIndependent =<< randnIO' [hiddenSize])
        <*> sample (LinearHypParams modelDevice hiddenSize actionNumEmbed)
      parseRNNG <- ParseRNNG
        <$> (makeIndependent =<< randnIO' [wordNumEmbed, wordEmbedSize])
        <*> (makeIndependent =<< randnIO' [ntNumEmbed, wordEmbedSize])
        <*> (makeIndependent =<< randnIO' [actionNumEmbed, wordEmbedSize])
        <*> sample (LstmHypParams modelDevice hiddenSize)
        <*> (makeIndependent =<< randnIO' [hiddenSize])
        <*> (makeIndependent =<< randnIO' [hiddenSize])
        <*> (makeIndependent =<< randnIO' [actionEmbedSize])
        <*> (makeIndependent =<< randnIO' [wordEmbedSize])
        <*> (makeIndependent =<< randnIO' [wordEmbedSize])
      compRNNG <- CompRNNG
        <$> sample (BiLstmHypParams modelDevice 1 hiddenSize)
        <*> (makeIndependent =<< randnIO' [hiddenSize])
        <*> (makeIndependent =<< randnIO' [hiddenSize])
      return $ RNNG predictActionRNNG parseRNNG compRNNG



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
      -- | stackをTensorで保存. 逆順で積まれる
      stack :: [Tensor],
      -- | stackの状態を文字列で保存. 逆順で積まれる
      textStack :: [T.Text],
      -- | TODO: (h, c)の順にする（hasktorch-toolsに追従）
      -- | stack lstmの隠れ状態の記録　(c, h)
      hiddenStack :: [(Tensor, Tensor)],
      buffer :: [Tensor],
      textBuffer :: [T.Text],
      actionHistory :: [Tensor],
      textActionHistory :: [Action]
    } ->
    RNNGState 
  deriving (Show)

predictNextAction ::
  RNNG ->
  -- | data
  RNNGState ->
  -- | possibility of actions
  Tensor
predictNextAction (RNNG PredictActionRNNG {..} _ _) RNNGState {..} = 
  let (_, bufferEmbedding) = rnnLayer bufferRNN (toDependent $ h0) buffer
      stackEmbedding = snd $ head hiddenStack
      (_, actionEmbedding) = rnnLayer actionRNN (toDependent $ h0) actionHistory
      ut = Torch.tanh $ ((cat (Dim 0) [stackEmbedding, bufferEmbedding, actionEmbedding]) `matmul` (toDependent w) + (toDependent c))
  in logSoftmax (Dim 0) $ linearLayer linearParams ut

push ::
  [a] ->
  a ->
  [a]
push list d = list ++ [d]

stackLSTMForward :: 
  LstmParams ->
  -- | [(ci, hi)]
  [(Tensor, Tensor)] ->
  -- | new Element
  Tensor ->
  -- | (cn, hn)
  (Tensor, Tensor)
stackLSTMForward stackLSTM stack newElem = last $ lstmLayer stackLSTM (head stack) [newElem]


parse ::
  -- | model
  RNNG ->
  IndexData ->
  -- | new RNNGState
  RNNGState ->
  Action ->
  RNNGState
parse (RNNG _ ParseRNNG {..} _) IndexData {..} RNNGState {..} (NT label) =
  let nt_embedding = embedding' (toDependent ntEmbedding) ((asTensor . ntIndexFor) label)
      textAction = NT label
      action_embedding = embedding' (toDependent actionEmbedding) ((asTensor . actionIndexFor) textAction)
  in RNNGState {
      stack = nt_embedding:stack,
      textStack = ((T.pack "(") `T.append` label):textStack,
      hiddenStack = (stackLSTMForward stackLSTM hiddenStack nt_embedding):hiddenStack,
      buffer = buffer,
      textBuffer = textBuffer,
      actionHistory = push actionHistory action_embedding,
      textActionHistory = push textActionHistory  textAction
    }
parse (RNNG _ ParseRNNG {..} _) IndexData {..} RNNGState {..} SHIFT =
  let textAction = SHIFT
      action_embedding = embedding' (toDependent actionEmbedding) ((asTensor . actionIndexFor) textAction)
  in RNNGState {
      stack = (head buffer):stack,
      textStack = (head textBuffer):textStack,
      hiddenStack = (stackLSTMForward stackLSTM hiddenStack (head buffer)):hiddenStack,
      buffer = tail buffer,
      textBuffer = tail textBuffer,
      actionHistory = push actionHistory action_embedding,
      textActionHistory = push textActionHistory textAction
    }
parse (RNNG _ ParseRNNG {..} CompRNNG {..}) IndexData {..} RNNGState {..} REDUCE =
  let textAction = REDUCE
      action_embedding = embedding' (toDependent actionEmbedding) ((asTensor . actionIndexFor) textAction)
      -- | 開いたlabelのidxを特定する
      (Just idx) = findIndex (\elem -> (T.isPrefixOf (T.pack "(") elem) && not (T.isSuffixOf (T.pack ")") elem)) textStack
      -- | popする
      (textSubTree, newTextStack) = splitAt (idx + 1) textStack
      (subTree, newStack) = splitAt (idx + 1) stack
      (_, newHiddenStack) = splitAt (idx + 1) hiddenStack
      -- composeする
      composedSubTree = fst $ last $ biLstmLayers compLSTM (toDependent $ compc0, toDependent $ comph0) (reverse subTree)
  in RNNGState {
      stack = composedSubTree:newStack,
      textStack = (T.intercalate (T.pack " ") (reverse $ (T.pack ")"):textSubTree)):newTextStack,
      hiddenStack = (stackLSTMForward stackLSTM newHiddenStack composedSubTree):newHiddenStack,
      buffer = buffer,
      textBuffer = textBuffer,
      actionHistory = push actionHistory action_embedding,
      textActionHistory = push textActionHistory textAction
    }


predict ::
  RuntimeMode ->
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
predict Train _ _ [] results rnngState = (reverse results, rnngState)
predict Train rnng indexData (action:rest) predictedHitory rnngState =
  let predicted = predictNextAction rnng rnngState
      newRNNGState = parse rnng indexData rnngState action
  in predict Train rnng indexData rest (predicted:predictedHitory) (Debug.Trace.traceShow newRNNGState newRNNGState)

predict Eval rnng IndexData {..} _ predictedHitory RNNGState {..} =
  if ((length textStack) == 1) && ((length textBuffer) == 0)
    then (reverse predictedHitory, RNNGState {..} )
  else
    let predicted = predictNextAction rnng RNNGState {..}
        action = indexActionFor (asValue (argmax (Dim 0) RemoveDim predicted)::Int)
    -- 不正なactionを予測した場合
    -- TODO: valid　actionのみに絞る
    in if (action == REDUCE && (length textStack == 1)) || (action == SHIFT && (length textBuffer) == 0)
         then (reverse predictedHitory, RNNGState {..} )
       else
         let newRNNGState = parse rnng IndexData {..} RNNGState {..} action
         in predict Eval rnng IndexData {..}  [] (predicted:predictedHitory) newRNNGState

rnngForward ::
  RuntimeMode ->
  RNNG ->
  -- | functions to convert text to index
  IndexData ->
  RNNGSentence ->
  [Tensor]
rnngForward runTimeMode (RNNG predictActionRNNG ParseRNNG {..} compRNNG) IndexData {..} (RNNGSentence (sents, actions)) =
  let sentsTensor = fmap ((embedding' (toDependent wordEmbedding)) . asTensor . wordIndexFor) sents
      initRNNGState = RNNGState {
        stack = [toDependent stackGuard],
        textStack = [],
        hiddenStack = [stackLSTMForward stackLSTM [(toDependent stackh0, toDependent stackc0)] (toDependent stackGuard)],
        buffer = sentsTensor ++ [toDependent bufferGuard],
        textBuffer = sents,
        actionHistory = [toDependent actionStart],
        textActionHistory = []
      }
  in fst $ predict runTimeMode (RNNG predictActionRNNG ParseRNNG {..} compRNNG) IndexData {..} actions [] initRNNGState

evaluate ::
  RNNG ->
  IndexData ->
  [RNNGSentence] ->
  -- | action answers
  [[Action]] ->
  -- | (accuracy, loss, predicted action sequence)
  ([Float], Float, [[Action]])
evaluate rnng IndexData {..} batches answers =
  let (acc, loss, ans) = foldLoop (zip batches answers) ([], 0, []::[[Action]]) $ 
        \(batch, answerActions) (acc', loss', ans') ->
          let answer = asTensor $ fmap actionIndexFor answerActions
              output = rnngForward Eval rnng IndexData {..} batch
              predicted = Torch.stack (Dim 0) $ fmap (argmax (Dim 0) RemoveDim) output 
              predictedSize = shape predicted !! 0
              answerSize = shape answer !! 0
              alignedOutput = if predictedSize < answerSize
                                  then output ++ replicate (answerSize - predictedSize) (zeros' [shape (head output) !! 0])
                                  else output 
              alignedPredicted = if predictedSize < answerSize
                                  then cat (Dim 0) [predicted, asTensor ((replicate (answerSize - predictedSize) (-1))::[Int])]
                                  else predicted
              alignedAnswer = if predictedSize > answerSize
                                then cat (Dim 0) [answer, asTensor ((replicate (predictedSize - answerSize) (-1))::[Int])]
                                else answer
              numCorrect = asValue (sumAll $ eq alignedAnswer alignedPredicted)::Int
              accuracy = (fromIntegral numCorrect::Float) / fromIntegral (Prelude.max answerSize predictedSize)::Float
              batchLoss = asValue (nllLoss' alignedAnswer (Torch.stack (Dim 0) alignedOutput))::Float
              predictedActions = fmap indexActionFor (asValue predicted::[Int])
          in (accuracy:acc', loss' + batchLoss, predictedActions:ans')
      size = fromIntegral (length batches)::Float
  in (acc, loss/size, ans)


main :: IO()
main = do
  -- experiment setting
  config <- configLoad
  let iter = fromIntegral (getEpoch config)::Int
      wordEmbedSize = fromIntegral (getWordEmbedSize config)::Int
      actionEmbedSize = fromIntegral (getActionEmbedSize config)::Int
      hiddenSize = fromIntegral (getHiddenSize config)::Int
      numLayer = fromIntegral (getNumLayer config)::Int
      modelFilePath = getModelFilePath config
      graphFilePath = getGraphFilePath config
      device = Device CPU 0
      optim = GD
  -- data
  trainingData <- loadActionsFromBinary $ getTrainingDataPath config
  validationData <- loadActionsFromBinary $ getValidationDataPath config
  evaluationData <- loadActionsFromBinary $ getEvaluationDataPath config

  let dataForTraining = trainingData

  let (wordIndexFor, indexWordFor, wordEmbDim) = indexFactory (buildVocab dataForTraining 3 toWordList) (T.pack "unk") Nothing
      (actionIndexFor, indexActionFor, actionEmbDim) = indexFactory (buildVocab dataForTraining 1 toActionList) ERROR Nothing
      (ntIndexFor, indexNTFor, ntEmbDim) = indexFactory (buildVocab dataForTraining 1 toNTList) (T.pack "unk") Nothing
      indexData = IndexData wordIndexFor indexWordFor actionIndexFor indexActionFor ntIndexFor indexNTFor

  let batches = Data.List.take 1 dataForTraining
      rnngSpec = RNNGSpec device wordEmbedSize actionEmbedSize wordEmbDim actionEmbDim ntEmbDim hiddenSize
  initRNNGModel <- sample $ rnngSpec

  -- | training
  ((trained, _), losses) <- mapAccumM [1..iter] (initRNNGModel, (optim, optim, optim)) $ 
    \epoc (rnng, (opt1, opt2, opt3)) -> do
      (updated, batchLosses) <- mapAccumM batches (rnng, (opt1, opt2, opt3)) $ 
        \batch (rnng', (opt1', opt2', opt3')) -> do
          let RNNGSentence (sents, actions) = batch
          let answer = asTensor $ fmap actionIndexFor actions
              output = rnngForward Train rnng' indexData (RNNGSentence (sents, actions))
              loss = nllLoss' answer (Torch.stack (Dim 0) output)
              RNNG actionPredictRNNG parseRNNG compRNNG = rnng'
          (updatedActionPredictRNNG, opt1'') <- runStep actionPredictRNNG opt1' loss 1e-2
          (updatedParseRNNG, opt2'') <- runStep parseRNNG opt2' loss 1e-2
          -- (updatedcompRNNG, opt3'') <- if length (filter (== REDUCE) actions) > 1
          --                                then runStep compRNNG opt3' loss 1e-2
          --                              else return (compRNNG, opt3')
          -- updatedParse <- runStep rnngParse opt' loss 1e-2
          return ((RNNG updatedActionPredictRNNG updatedParseRNNG compRNNG, (opt1'', opt2'', opt3')), (asValue loss::Float))

      let loss = sum batchLosses / (fromIntegral (length batches)::Float)
      return (updated, loss)

  -- | model保存
  -- Torch.Train.saveParams trained (modelFilePath ++ "-rnng")
  -- drawLearningCurve graphFilePath "Learning Curve" [("", reverse losses)]
  -- print losses
  
  -- -- | model読み込み
  -- rnngModel <- Torch.Train.loadParams rnngSpec (modelFilePath ++ "-rnng")

  -- -- let RNNGSentence (s, actions) = head evaluationData
  -- -- print wordEmbDim
  -- -- print $ shape $ toDependent $ wordEmbedding rnngModel
  -- -- print $ fmap (embedding' (toDependent (wordEmbedding rnngModel)) . asTensor . wordIndexFor) s

  -- let answers = fmap (\(RNNGSentence (_, actions)) -> actions) evaluationData
  --     (acc, loss, predicted) = evaluate rnngModel indexData evaluationData answers
  
  -- print $ "acc: " ++ show (sum acc / (fromIntegral (length acc)::Float))
  -- print $ "loss: " ++ show loss
  return ()
