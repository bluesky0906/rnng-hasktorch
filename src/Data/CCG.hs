{-# LANGUAGE ScopedTypeVariables #-}
{-# OPTIONS_GHC -Wno-incomplete-patterns #-}

module Data.CCG where
import Data.CCGRule
import Data.RNNGSentence
import Data.SyntaxTree
import GHC.Generics
import qualified Data.Text as T          --text
import qualified Data.Text.IO as T       --text
import qualified Data.Aeson            as A --aeson
import qualified Data.ByteString.Char8 as B --bytestring 
import qualified Data.ByteString as B --bytestring 
import qualified Data.ByteString.Lazy as BL
import qualified Data.Yaml             as Y --yaml
import Data.List as L
import Text.Parsec
import Text.Parsec.Text --parsec
import Data.Store --seralisation
import Debug.Trace


{-

check whether a Tree is valid CCG

-}

checkValidCCG :: 
  Tree -> 
  Bool
checkValidCCG tree@(Phrase (cat, subtrees)) =
  case length subtrees of 
    1 -> if oneSubtree subtrees
           then all checkValidCCG subtrees
           else error $ show tree
    2 -> if twoSubtrees subtrees
           then all checkValidCCG subtrees
           else error $ show tree
    _ -> error $ show tree
  where
    oneSubtree [Word _] = True
    oneSubtree [first] =
      forwardTypeRaisingRule first tree ||
      backwardTypeRaisingRule first tree ||
      -- non-combinatory rules
      unaryTypeChangingRules first tree
    twoSubtrees [first, second] = 
      forwardFunctionalAplicationRule (first, second) tree ||
      backwardFunctionalAplicationRule (first, second) tree ||
      forwardCompositionRule (first, second) tree ||
      backwardCompositionRule (first, second) tree ||
      forwardCrossingCompositionRule (first, second) tree ||
      backwardCrossingCompositionRule (first, second) tree ||
      generalizedForwardCompositionRule (first, second) tree ||
      generalizedBackwardCompositionRule (first, second) tree ||
      generalizedForwardCrossingCompositionRule (first, second) tree ||
      generalizedBackwardCrossingCompositionRule (first, second) tree ||
      generalizedBackwardCrossingCompositionRule2 (first, second) tree ||
      forwardSubstitutionRule (first, second) tree ||
      backwardSubstitutionRule (first, second) tree ||
      forwardCrossingSubstitutionRule (first, second) tree ||
      backwardCrossingSubstitutionRule (first, second) tree ||
      -- non-combinatory rules
      coordinationRules (first, second) tree ||
      punctuationRule (first, second) tree ||
      binaryTypeChangingRules (first, second) tree
checkValidCCG (Word _) = True

{-
:l src/Data/CCG.hs
import Util
config <- configLoad
posMode = posModeConfig config
grammarMode = grammarModeConfig config

(trainDataPath, evalDataPath, validDataPath) = dataFilePath grammarMode posMode

rnngSentences <- loadActionsFromBinary trainDataPath
trees = fromRNNGSentences rnngSentences
Data.List.all checkValidCCG trees

-}


{-

parse text file to CCGTree

-}

parseCCGfile :: Bool -> FilePath -> IO [Tree]
parseCCGfile posMode filePath = do
  ccg <- T.readFile filePath
  return $ (parseCCGTrees posMode) ccg

parseCCGTrees :: Bool -> T.Text -> [Tree]
parseCCGTrees posMode text =
  case parse (ccgsParser posMode) "" text of
    Left e -> [Err (show e) text]
    Right t -> t

ccgsParser :: Bool -> Parser [Tree]
ccgsParser posMode = many1 (ccgParser posMode)

ccgParser :: Bool -> Parser Tree
ccgParser posMode = do
  ignoreSentenceInfo
  tree <- try nonLeafParser <|> headLeafParser posMode
  optional newline
  return tree

ignoreSentenceInfo :: Parser ()
ignoreSentenceInfo = do
  string "ID=wsj"
  manyTill anyChar (try endOfLine)
  return ()

blank :: Parser ()
blank = do
  _ <- many1 $ oneOf " \t"
  return ()

literal :: Parser T.Text
literal = T.pack <$> (many1 $ noneOf " <>\n\t")

openParen :: Parser T.Text
openParen = T.singleton <$> char '('

closeParen :: Parser T.Text
closeParen = T.singleton <$> char ')'

openAngleBracket :: Parser T.Text
openAngleBracket = T.singleton <$> char '<'

closeAngleBracket :: Parser T.Text
closeAngleBracket = T.singleton <$> char '>'

-- | １単語のみのデータに対応
headLeafParser :: Bool -> Parser Tree
headLeafParser posMode = do
  openParen
  openAngleBracket
  char 'L'
  blank
  category <- literal
  blank
  pos <- literal
  blank
  literal
  blank
  word <- literal
  blank
  literal
  closeAngleBracket
  closeParen
  blank
  if posMode 
    then return $ Phrase (category, [Word word])
    else return $ Phrase (category, [Phrase (pos, [Word word])])

leafParser :: Parser Tree
leafParser = do
  openParen
  openAngleBracket
  char 'L'
  blank
  category <- literal
  blank
  literal
  blank
  literal
  blank
  word <- literal
  blank
  literal
  closeAngleBracket
  closeParen
  blank
  return $ Phrase (category, [Word word])

nonLeafParser :: Parser Tree
nonLeafParser = do
  openParen
  openAngleBracket
  char 'T'
  blank
  category <- literal
  blank
  literal
  blank
  literal
  closeAngleBracket
  blank
  -- child <- manyTill anyChar closeParen
  tree <- many (try nonLeafParser <|> leafParser) 
  closeParen
  blank
  return $ Phrase (category, tree)

sepBy1' :: (Stream s m t) => ParsecT s u m a -> ParsecT s u m sep -> ParsecT s u m [a]
{-# INLINABLE sepBy1' #-}
sepBy1' p sep = do 
  x <- p
  xs <- many $ try (sep >> p)
  return $ x:xs
