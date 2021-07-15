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
  printCFGdata cfgData
  let (stack, actions) = traverseCFG ([], []) (head cfgData)
  print stack
  print actions
  let history = traverseCFG' [] (head cfgData)
  print history
  saveCFGData "test.yaml" cfgData
  content <- loadCFGData "test.yaml"
  print content
  return ()