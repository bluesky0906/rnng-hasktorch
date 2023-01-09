{-# LANGUAGE DeriveGeneric #-}

module Util where
import qualified Data.Text as T  
import qualified Data.Set as S
import qualified Data.Map.Strict as M
import Data.List.Split (chunksOf, splitEvery) --split
import Data.List as L
import Data.Ord

import System.Directory.ProjectRoot (getProjectRootWeightedCurrent)
import Dhall hiding ( map )

{-

汎用的な関数　(hasktorch-toolsに移動したい)

-}
maxLengthFor :: [[a]] -> Int
maxLengthFor list = length $ L.maximumBy (comparing length) list

padding :: Int -> [Int] -> [Int]
padding maxLength list = list ++ take (maxLength - (length list)) (repeat 1)

-- |  batchごとに分けられたデータをpaddingしてindex化する
indexForBatch ::
  (a -> Int) ->
  [[a]] ->
  [[Int]]
indexForBatch indexFor input = 
  fmap (padding $ maxLengthFor input) indices
  where
    indices = fmap (fmap indexFor) input

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

{-

for Config file

-}

getProjectRoot :: IO (String)
getProjectRoot = do
  projectRoot <- getProjectRootWeightedCurrent
  return (case projectRoot of
                Nothing -> "./"
                Just a -> a)


data Config = Config { 
  modeConfig :: String, 
  trainingDataPathConfig :: String,
  validationDataPathConfig :: String,
  evaluationDataPathConfig :: String,
  epochConfig :: Natural,
  validationStepConfig :: Natural,
  actionEmbedSizeConfig :: Natural,
  wordEmbedSizeConfig :: Natural,
  hiddenSizeConfig :: Natural,
  numOfLayerConfig :: Natural,
  learningRateConfig :: Double,
  modelNameConfig :: String
  } deriving (Generic, Show)

instance FromDhall Config

configLoad :: IO Config
configLoad = do
  projectRoot <- getProjectRoot
  x <- input auto $ T.pack (projectRoot ++ "/config.dhall")
  return (x:: Config)
