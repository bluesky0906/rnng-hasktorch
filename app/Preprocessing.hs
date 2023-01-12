module Preprocessing where
import Data.SyntaxTree
import Data.CFG
import Data.CCG
import Data.RNNGSentence
import Util (configLoad, Config(..))
import Options.Applicative
import qualified Data.Text as T          --text
import qualified Data.Text.IO as T       --text
import System.FilePath.Posix (takeBaseName) --filepath
import Text.Directory (checkFile, getFileList) --nlp-tools
import System.Directory (doesDirectoryExist, doesFileExist, listDirectory) --directory

{-

  PennTreeBank, CCGBank のデータを前処理する

-}

data PreprocessingOpt = PreprocessingOpt
  {
    path :: FilePath,
    grammar :: Grammar
  } deriving (Show)

preprocess :: Parser PreprocessingOpt
preprocess = PreprocessingOpt
  <$> strOption ( long "path" <> short 'p' <> help "path to WSJ" )
  <*> option auto ( long "grammar" <> short 'g' <> help "CFG or CCG" )

opts :: ParserInfo PreprocessingOpt
opts = info (preprocess <**> helper)
  ( fullDesc
  <> progDesc "Path of WSJ PennTreeBank data" 
  <> progDesc "Grammar used by RNNG" 
  )

listFiles ::
  Grammar ->
  String ->
  IO [String]
listFiles grammar p = do
  let suffix = case grammar of
                CFG -> "mrg"
                CCG -> "auto"
  isFile <- doesFileExist p
  if isFile then return [p] else getFileList suffix p

saveActionData ::
  Grammar ->
  [FilePath] -> 
  FilePath ->
  IO()
saveActionData grammar dirsPath outputPath = do
  -- 指定されたディレクトリ以下のファイルを取得
  filePaths <- fmap concat $ traverse (listFiles grammar) dirsPath 
  --readmeは除外
  treess <- mapM (parseTreefile grammar) $ filter (\f -> takeBaseName f /= "readme") filePaths
  let trees = concat treess
      parsedTrees = filter (not . isErr) $ trees
  mapM_ print $ filter isErr $ trees
  let rnngSentences = traverseTrees parsedTrees
  saveActionsToBinary outputPath rnngSentences
  -- content <- loadActionsFromBinary outputPath
  return ()
  where
    parseTreefile CFG = parseCFGfile
    parseTreefile CCG = parseCCGfile


main :: IO()
main = do
  options <- execParser opts
  config <- configLoad
  let wsjDirPath = path options
      rnngGrammar = grammar options
      trainingDataDirs = fmap (wsjDirPath ++) ["02/", "03/", "04/", "05/", "06/", "07/", "08/", "09/", "10/", "11/", "12/", "13/", "14/", "15/", "16/", "17/", "18/", "19/", "20/", "21/"]
      validationDataDirs = fmap (wsjDirPath ++) ["24/"]
      evaluationDataDirs = fmap (wsjDirPath ++) ["23/"]
  saveActionData rnngGrammar trainingDataDirs $ "data/training" ++ show rnngGrammar
  saveActionData rnngGrammar evaluationDataDirs $ "data/evaluation" ++ show rnngGrammar
  saveActionData rnngGrammar validationDataDirs $ "data/validation" ++ show rnngGrammar
  return ()
