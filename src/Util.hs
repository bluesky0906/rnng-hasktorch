{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}

module Util where
import qualified Data.Text as T  
import qualified Data.Set as S
import qualified Data.Map.Strict as M
import Data.List.Split (chunksOf, splitEvery) --split
import Data.List as L
import Data.Ord
import System.Directory.ProjectRoot (getProjectRootWeightedCurrent)
import System.Random
import Dhall hiding ( map )
import Graphics.Gnuplot.Simple

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
  ("data/traininig" ++ suffix, "data/eval" ++ suffix, "data/valid" ++ suffix)
  where
    suffix = grammar ++ if posMode then "POS" else ""

modelNameConfig ::
  Config ->
  String
modelNameConfig Config{..} =
  "rnng-" ++ grammarModeConfig ++
  pos ++
  "-layer" ++ show numOfLayerConfig ++
  "-hidden" ++ show hiddenSizeConfig ++
  "-epoch" ++  show epochConfig ++
  "-lr" ++ show learningRateConfig
  where
    pos = if posModeConfig then "-pos" else ""

getProjectRoot :: IO String
getProjectRoot = do
  projectRoot <- getProjectRootWeightedCurrent
  return (case projectRoot of
                Nothing -> "./"
                Just a -> a)


data Config = Config { 
  modeConfig :: String, 
  parsingModeConfig :: String,
  posModeConfig :: Bool,
  grammarModeConfig :: String,
  epochConfig :: Natural,
  validationStepConfig :: Natural,
  actionEmbedSizeConfig :: Natural,
  wordEmbedSizeConfig :: Natural,
  hiddenSizeConfig :: Natural,
  numOfLayerConfig :: Natural,
  learningRateConfig :: Double
  } deriving (Generic, Show)

instance FromDhall Config

configLoad :: IO Config
configLoad = do
  projectRoot <- getProjectRoot
  x <- input auto $ T.pack (projectRoot ++ "/config.dhall")
  return (x:: Config)
