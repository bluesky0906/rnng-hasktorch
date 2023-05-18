{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleInstances #-}
-- {-# LANGUAGE BlockArguments #-}

module Util where
import qualified Data.Text as T  
import qualified Data.Set as S
import qualified Data.Map.Strict as M
import Data.List.Split (chunksOf, splitEvery) --split
import Data.List as L
import Data.Ord
import System.Directory (doesFileExist)
import System.Directory.ProjectRoot (getProjectRootWeightedCurrent)
import System.Random
import Dhall hiding ( map )
import Graphics.Gnuplot.Simple
import Debug.Trace

import Torch 
-- | hasktorch-tools
import Torch.Tensor.Util (unstack)

{-

汎用的な関数　(hasktorch-toolsに移動したい)

-}
maxLengthFor :: [[a]] -> Int
maxLengthFor list = length $ L.maximumBy (comparing length) list

padding :: Int -> [Int] -> [Int]
padding maxLength list = list ++ replicate (maxLength - (length list)) 1

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


replaceAt :: [a] -> (Int, a) ->  [a]
replaceAt [] _ = []
replaceAt (_:xs) (0, y) = y : xs
replaceAt (x:xs) (n, y) = x : replaceAt xs ((n - 1), y)

-- TODO: Tensorのままやりたい
class Replaceable a where
  -- 指定した次元のtensorを置き換える(2次元、３次元をサポート)
  (//) ::
    -- target
    Tensor ->
    -- indice
    [a] ->
    -- output
    Tensor

subst :: Tensor -> [Tensor] -> Tensor -> Tensor
subst target indices alt = stack (Dim 0) $ replace indices
  where
    replace (idx0:[]) = foldl' replaceAt (unstack target) (zip (asValue idx0::[Int]) (unstack alt))
    replace (idx0:idx1:[]) = map (stack (Dim 0)) $ foldl' replaceAt2 (map unstack (unstack target)) (zip3 (asValue idx0::[Int]) (asValue idx1::[Int]) (unstack alt))
    replaceAt1 :: [a] -> (Int, a) ->  [a]
    replaceAt1 [] _ = []
    replaceAt1 (_:xs) (0, y) = y : xs
    replaceAt1 (x:xs) (n, y) = x : replaceAt1 xs ((n - 1), y)
    replaceAt2 :: [[a]] -> (Int, Int, a) ->  [[a]]
    replaceAt2 t (idx0, idx1, x) = replaceAt t (idx0, (replaceAt (t !! idx0) (idx1, x)))

instance Replaceable (Int, Tensor) where
  (//) target indices =
    stack (Dim 0) replaced 
    where
      replaced = foldl replaceAt (unstack target) indices

instance Replaceable (Int, Int, Tensor) where
  (//) target indices =
    stack (Dim 0) (map (stack (Dim 0)) replaced)
    where
      replaced = foldl' subst' (map unstack (unstack target)) indices
      subst' :: [[Tensor]] -> (Int, Int, Tensor) -> [[Tensor]]
      subst' t (idx0, idx1, x) = replaceAt t (idx0, (replaceAt (t !! idx0) (idx1, x)))

(!.) :: Tensor -> [Tensor] -> Tensor
target !. indices =
  case length indices of 
    1 -> target ! (head indices)
    2 -> Torch.stack (Dim 0) $ map ((target !) . toTuple2) idxList
    3 -> Torch.stack (Dim 0) $ map ((target !) . toTuple3) idxList
  where
    idxList = (asValue (stack (Dim 1) indices)::[[Int]])
    toTuple1 (a:[]) = (a)
    toTuple2 (a:b:[]) = (a,b)
    toTuple3 (a:b:c:[]) = (a,b,c)

{-

nlp-toolsに移動したい

-}

counts :: Ord a => [a] -> [(a, Int)]
counts = map count . group . sort
  where count xs = (head xs, length xs)

-- TODO: X軸の表示
-- unused
drawHistgram ::
  (Ord a, Show a) =>
  FilePath ->
  String ->
  [a] ->
  IO()
drawHistgram filepath title lst = do
  let strlst = fmap show lst
      freqlstWithIdx = zip [0..] $ sortOn (Down . snd) $ counts strlst
      dataForPlot = fmap (\(idx, freq) -> (idx, snd freq)) freqlstWithIdx
      xTicks = unwords $ fmap (\(idx, (str, _)) -> "'"++ str ++ "'" ++ " " ++ show idx) freqlstWithIdx
  print $ length $ takeWhile (\x -> snd x > 1) dataForPlot
  plotPathStyle [(PNG filepath), (Title title), (XRange (0, fromIntegral (length dataForPlot)::Double))] (defaultStyle {plotType = Boxes}) dataForPlot


sampleRandomData ::
  -- | 取り出したいデータ数
  Int ->
  -- | データ
  [a] ->
  -- | サンプル結果
  IO [a]
sampleRandomData size xs = do
  gen <- newStdGen
  let randomIdxes = L.take size $ nub $ randomRs (0, (length xs) - 1) gen
  return $ map (xs !!) randomIdxes


{-

for Config file

-}

data DataType = Train | Eval | Valid 

dataFilePath ::
  -- | grammar
  String ->
  -- | pos mode
  Bool ->
  -- | train, eval, valid
  (String, String, String)
dataFilePath grammar posMode = 
  ("data/training" ++ suffix, "data/evaluation" ++ suffix, "data/validation" ++ suffix)
  where
    suffix = grammar ++ if posMode then "POS" else ""

getProjectRoot :: IO String
getProjectRoot = do
  projectRoot <- getProjectRootWeightedCurrent
  return (case projectRoot of
                Nothing -> "./"
                Just a -> a)


modelNameConfig ::
  -- | overwrite
  String ->
  Config ->
  FilePath
modelNameConfig rnngMode Config{..} =
  if modeConfig == "Train"
    then 
      "rnng-" ++ rnngMode ++
      "-" ++ grammarModeConfig ++
      pos ++
      "-layer" ++ show numOfLayerConfig ++
      "-hidden" ++ show hiddenSizeConfig ++
      "-epoch" ++  show epochConfig ++
      "-lr" ++ show learningRateConfig ++
      if modelVersionConfig == "" then "" else "-" ++ modelVersionConfig
    else evalModelNameConfig
  where
    pos = if posModeConfig then "-pos" else ""

data Config =  Config { 
  modeConfig :: String, 
  parsingModeConfig :: String,
  posModeConfig :: Bool,
  grammarModeConfig :: String,
  epochConfig :: Natural,
  validationStepConfig :: Natural,
  batchSizeConfig :: Natural,
  actionEmbedSizeConfig :: Natural,
  wordEmbedSizeConfig :: Natural,
  hiddenSizeConfig :: Natural,
  numOfLayerConfig :: Natural,
  learningRateConfig :: Double,
  modelVersionConfig :: String,
  resumeTrainingConfig :: Bool,
  evalModelNameConfig :: String
  } deriving (Generic, Show)

instance FromDhall Config

configLoad :: IO Config
configLoad = do
  projectRoot <- getProjectRoot
  x <- input auto $ T.pack (projectRoot ++ "/config.dhall")
  return (x:: Config)
