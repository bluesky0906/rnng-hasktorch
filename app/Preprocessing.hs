module Main where

import PTB 
import Options.Applicative
import qualified Data.Text as T          --text
import qualified Data.Text.IO as T       --text


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

main :: IO()
main = do
  options <- execParser opts
  content <- T.readFile $ path options
  let cfgData = map cfgParser (T.lines content)
  printCFGdata $ [head cfgData]
  let cfgActionData = traverseCFGs cfgData
  saveCFGData "test.yaml" cfgActionData
  content <- loadCFGData "test.yaml"
  return ()