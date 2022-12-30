module Preprocessing where

import Data.CFG
import Data.RNNGSentence
import Util (configLoad, Config(..))
import Options.Applicative
import qualified Data.Text as T          --text
import qualified Data.Text.IO as T       --text
import System.FilePath.Posix (takeBaseName) --filepath
import Text.Directory (checkFile, getFileList) --nlp-tools
import System.Directory (doesDirectoryExist, doesFileExist, listDirectory) --directory

{-
  PennTreeBankのデータを前処理する
-}

newtype PreprocessingOpt = PreprocessingOpt
  {
    path :: FilePath
  } deriving (Show)

preprocess :: Parser PreprocessingOpt
preprocess = PreprocessingOpt
  <$> strOption
  ( long "path"
  <> short 'p' 
  )

opts :: ParserInfo PreprocessingOpt
opts = info (preprocess <**> helper)
  ( fullDesc
  <> progDesc "Path of WSJ PennTreeBank data" )

listFiles :: String -> IO [String]
listFiles p = do
  isFile <- doesFileExist p
  if isFile then return [p] else getFileList "mrg" p

saveActionData :: [FilePath] -> FilePath -> IO()
saveActionData dirsPath outputPath = do
  -- 指定されたディレクトリ以下のファイルを取得
  filePaths <- fmap concat $ traverse id $ fmap listFiles dirsPath 
  --readmeは除外
  cfgTreess <- mapM parsePTBfile $ filter (\f -> takeBaseName f /= "readme") filePaths
  let cfgTrees = concat cfgTreess
  let parsedCfgTrees = filter (not . isErr) $ cfgTrees
  mapM_ (putStrLn . show) $ filter isErr $ cfgTrees
  let rnngSentence = traverseCFGs parsedCfgTrees
  saveActionsToBinary outputPath rnngSentence
  -- content <- loadActionsFromBinary outputPath
  return ()

main :: IO()
main = do
  options <- execParser opts
  config <- configLoad
  let wsjDirPath = path options
      trainingDataDirs = fmap (wsjDirPath ++) ["02/", "03/", "04/", "05/", "06/", "07/", "08/", "09/", "10/", "11/", "12/", "13/", "14/", "15/", "16/", "17/", "18/", "19/", "20/", "21/"]
      validationDataDirs = fmap (wsjDirPath ++) ["24/"]
      evaluationDataDirs = fmap (wsjDirPath ++) ["23/"]
  -- saveActionData trainingDataDirs $ trainingDataPath config
  saveActionData validationDataDirs $ getValidationDataPath config
  saveActionData evaluationDataDirs $ getEvaluationDataPath config
  return ()