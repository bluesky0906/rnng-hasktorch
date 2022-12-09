{-# LANGUAGE DeriveGeneric #-}

module Util where
import PTB 
import qualified Data.Text as T  
import qualified Data.Set as S
import qualified Data.Map.Strict as M
import Data.List.Split (chunksOf, splitEvery) --split
import Data.List as L
import Data.Ord
import ML.Util.Dict (sortWords)

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

toActionList :: [RNNGSentence] -> [Action]
toActionList [] = []
toActionList ((RNNGSentence (_, actions)):rest) = actions ++ toActionList rest

extractNT :: [Action] -> [T.Text]
extractNT [] = []
extractNT ((NT label):rest) = label:(extractNT rest)
extractNT (_:rest) = extractNT rest

toNTList :: [RNNGSentence] -> [T.Text]
toNTList [] = []
toNTList ((RNNGSentence (_, actions)):rest) = extractNT actions ++ toNTList rest

buildVocab :: 
  (Ord a) =>
  -- | training data
  [RNNGSentence] ->
  -- | 出現頻度threshold
  Int ->
  -- | 語彙リストを作る関数
  ([RNNGSentence] -> [a])
  -- | 一意な語彙リスト
  -> [a]
buildVocab rnngData freq toList = sortWords freq (toList rnngData)


indexFactory :: (Ord a, Eq a) =>
  -- | 単語列
  [a] ->
  -- | 未知語を表すa型の要素(idxは0)
  a ->
  -- | paddingを表すa型の要素(含む場合は1がpadding idx)
  Maybe a ->
  -- | 単語のindexを返す関数（未知語は0）, indexから単語を返す関数, リストのサイズ
  (a -> Int, Int -> a, Int)
indexFactory dic unk padding =
  case padding of
    Nothing -> (wordToIndexFactory (M.fromList (zip dic [1..])), indexToWordFactory (M.fromList (zip [0..] (unk:dic))), dic_size + 1)
    (Just pad) -> (wordToIndexFactory (M.fromList (zip dic [2..])), indexToWordFactory (M.fromList (zip [0..] (unk:pad:dic))), dic_size + 2)
  where
    dic_size = length dic
    wordToIndexFactory map wrd = M.findWithDefault 0 wrd map
    indexToWordFactory map idx = M.findWithDefault unk idx map

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
  getActionEmbedSize :: Natural,
  getWordEmbedSize :: Natural,
  getHiddenSize :: Natural,
  getNumLayer :: Natural,
  getLearningRate :: Double,
  getGraphFilePath :: String,
  getModelFilePath :: String
  } deriving (Generic, Show)

instance FromDhall Config

configLoad :: IO Config
configLoad = do
  projectRoot <- getProjectRoot
  x <- input auto $ T.pack (projectRoot ++ "/config.dhall")
  return (x:: Config)