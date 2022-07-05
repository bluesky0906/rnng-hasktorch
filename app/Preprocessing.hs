module Main where

import PTB 
import Options.Applicative
import qualified Data.Text as T          --text
import qualified Data.Text.IO as T       --text
import System.FilePath.Posix (takeBaseName) --filepath
import Text.Directory (checkFile, getFileList) --nlp-tools
import System.Directory (doesDirectoryExist, doesFileExist, listDirectory) --directory


newtype Preprocess = Preprocess
  {
    path :: FilePath
  } deriving (Show)

preprocess :: Parser Preprocess
preprocess = Preprocess
  <$> strOption
  ( long "path"
  <> short 'p' 
  )

opts :: ParserInfo Preprocess
opts = info (preprocess <**> helper)
  ( fullDesc
  <> progDesc "Load Penn Treebank Data." )

listFiles :: String -> IO [String]
listFiles p = do
  isFile <- doesFileExist p
  if isFile then return [p] else getFileList "mrg" p

main :: IO()
main = do
  options <- execParser opts
  let directoryPath = path options
  filePaths <- listFiles directoryPath -- 指定されたディレクトリ以下のファイルを取得
  cfgTreess <- mapM parsePTBfile $ filter (\f -> takeBaseName f /= "readme") filePaths
  let cfgTrees = concat cfgTreess
  let parsedCfgTrees = filter (not . isErr) $ cfgTrees
  mapM_ (putStrLn . show) $ filter isErr $ cfgTrees
  let cfgActionData = traverseCFGs parsedCfgTrees
  saveActionsToBinary "data/actions" cfgActionData
  content <- loadActionsFromBinary "data/actions"
  return ()