module Preprocessing where
import Data.SyntaxTree
import Data.CFG
import Data.CCG
import Data.RNNGSentence
import Util.File (dataFilePath)
import Options.Applicative
import qualified Data.Text as T          --text
import qualified Data.Text.IO as T       --text
import Text.Directory (checkFile) --nlp-tools
import System.Directory (doesDirectoryExist, listDirectory) --directory

{-

  PennTreeBank, CCGBank のデータを前処理する

-}

data PreprocessingOpt = PreprocessingOpt
  {
    path :: FilePath,
    grammar :: Grammar,
    pos :: Bool
  } deriving (Show)

preprocess :: Parser PreprocessingOpt
preprocess = PreprocessingOpt
  <$> strOption ( long "path" <> short 'p' <> help "path to WSJ" )
  <*> option auto ( long "grammar" <> short 'g' <> help "CFG or CCG" )
  <*> option auto ( long "pos" <> help "include POS-tag or not" )

opts :: ParserInfo PreprocessingOpt
opts = info (preprocess <**> helper)
  ( fullDesc
  <> progDesc "Path of WSJ PennTreeBank data" 
  <> progDesc "Grammar used by RNNG" 
  )

saveActionData ::
  Grammar ->
  Bool ->
  [FilePath] -> 
  FilePath ->
  IO()
saveActionData grammar posMode dirsPath outputPath = do
  -- 指定されたディレクトリ以下のファイルを取得
  filePaths <- treefiles grammar dirsPath
  treess <- mapM (parseTreefile grammar) filePaths
  let trees = concat treess
      parsedTrees = filter (not . isErr) $ trees
  mapM_ print $ filter isErr $ trees
  let rnngSentences = toRNNGSentences posMode parsedTrees
  saveActionsToBinary outputPath rnngSentences
  return ()
  where
    parseTreefile CFG = parseCFGfile
    parseTreefile CCG = parseCCGfile posMode


main :: IO()
main = do
  options <- execParser opts
  let wsjDirPath = path options
      rnngGrammar = grammar options
      posMode = pos options
      trainingDataDirs = fmap (wsjDirPath ++) ["02/", "03/", "04/", "05/", "06/", "07/", "08/", "09/", "10/", "11/", "12/", "13/", "14/", "15/", "16/", "17/", "18/", "19/", "20/", "21/"]
      validationDataDirs = fmap (wsjDirPath ++) ["24/"]
      evaluationDataDirs = fmap (wsjDirPath ++) ["23/"]
      (trainDataPath, evalDataPath, validDataPath) = dataFilePath (show rnngGrammar) posMode
  saveActionData rnngGrammar posMode trainingDataDirs trainDataPath
  saveActionData rnngGrammar posMode evaluationDataDirs evalDataPath
  saveActionData rnngGrammar posMode validationDataDirs validDataPath
  return ()
