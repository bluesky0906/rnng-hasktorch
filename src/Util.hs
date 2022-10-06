{-# LANGUAGE DeriveGeneric #-}

module Util where
import PTB 
import qualified Data.Text as T  
import qualified Data.Set as S
import qualified Data.Map.Strict as M
import Data.List.Split (chunksOf, splitEvery) --split
import Data.List as L
import Data.Ord
import Torch.Util.Dict (sortWords)

import System.Directory.ProjectRoot (getProjectRootWeightedCurrent)
import Dhall hiding ( map )

extractSentences :: [RNNGSentence] -> [Sentence]
extractSentences [] = []
extractSentences ((RNNGSentence (words, _)):rest) = words:(extractSentences rest)

maxLengthFor :: [[a]] -> Int
maxLengthFor list = length $ L.maximumBy (comparing length) list

padding :: Int -> [Int] -> [Int]
padding maxLength list = list ++ take (maxLength - (length list)) (repeat 1)

indexForBatch :: 
  (a -> Int) ->
  [[a]] ->
  [[Int]]
indexForBatch indexFor input = 
  fmap (padding $ maxLengthFor input) indices
  where
    indices = fmap (fmap indexFor) input

toWordList :: [RNNGSentence] -> [T.Text]
toWordList [] = []
toWordList ((RNNGSentence (words, _)):rest) = words ++ toWordList rest

buildVocab ::
  -- | training data
  [RNNGSentence] ->
  -- | 出現頻度threshold
  Int ->
  -- | 一意な語彙リスト
  [T.Text]
buildVocab actionData freq = sortWords freq (toWordList actionData)

indexFactory :: (Ord a, Eq a) =>
  -- | 単語列
  [a] ->
  -- | paddingをindexに含むか否か(含む場合は1がpadding idx)
  Bool ->
  -- | 単語のindexを返す関数（未知語は0）, リストのサイズ
  (a -> Int, Int)
indexFactory dic padding =
  case padding of
    False -> (factory (M.fromList (zip dic [1..])), dic_size + 1)
    True -> (factory (M.fromList (zip dic [2..])), dic_size + 2)
  where
    dic_size = length dic
    factory hash wrd = M.findWithDefault 0 wrd hash

actionDic :: Action -> String
actionDic (NT _) = "NT"
actionDic SHIFT = "SHIFT"
actionDic REDUCE = "REDUCE"

getProjectRoot :: IO (String)
getProjectRoot = do
  projectRoot <- getProjectRootWeightedCurrent
  return (case projectRoot of
                Nothing -> "./"
                Just a -> a)


data Config = Config { 
  getTrainingDataPath :: String,
  getValidationDataPath :: String,
  getEvaluationDataPath :: String,
  getTrial :: Natural,
  getEpoch :: Natural,
  getLstmDim :: Natural,
  getLearningRate :: Double,
  getBatchSize :: Natural,
  getGraphFilePath :: String,
  getModelFilePath :: String
  } deriving (Generic, Show)

instance FromDhall Config

configLoad :: IO Config
configLoad = do
  projectRoot <- getProjectRoot
  x <- input auto $ T.pack (projectRoot ++ "/config.dhall")
  return (x:: Config)