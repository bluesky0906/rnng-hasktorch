{-#  LANGUAGE DeriveGeneric #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# OPTIONS_GHC -Wno-incomplete-patterns #-}

module PTB where
import GHC.Generics
import qualified Data.Text as T          --text
import qualified Data.Text.IO as T       --text
import qualified Data.Aeson            as A --aeson
import qualified Data.ByteString.Char8 as B --bytestring 
import qualified Data.ByteString as B --bytestring 
import qualified Data.ByteString.Lazy as BL
import qualified Data.Yaml             as Y --yaml
import Data.List as L
import Debug.Trace
import Text.Parsec
import Text.Parsec.Text --parsec
import Data.Store --seralisation

data CFGtree = 
  Phrase (T.Text, [CFGtree]) 
  | Word T.Text
  | Err String T.Text 
  deriving (Eq, Show, Generic)

instance A.FromJSON CFGtree
instance A.ToJSON CFGtree

data Action = NT T.Text | SHIFT | REDUCE | ERROR deriving (Eq, Show, Generic, Ord)
type Sentence = [T.Text]

showAction :: Action -> T.Text
showAction (NT label) = (T.pack "NT_") <> label
showAction SHIFT = T.pack "SHIFT"
showAction REDUCE = T.pack "REDUCE"
showAction ERROR = T.pack "ERROR"

instance Store Action
instance A.FromJSON Action
instance A.ToJSON Action

newtype RNNGSentence = RNNGSentence (Sentence, [Action]) deriving (Eq, Show, Generic)

instance Store RNNGSentence
instance A.FromJSON RNNGSentence
instance A.ToJSON RNNGSentence

unpackRNNGSentence :: RNNGSentence -> (Sentence, [Action])
unpackRNNGSentence (RNNGSentence (words, actions)) = (words, actions)

saveActionsToBinary :: FilePath -> [RNNGSentence] -> IO()
saveActionsToBinary filepath actions = B.writeFile filepath (encode actions)

loadActionsFromBinary :: FilePath -> IO [RNNGSentence]
loadActionsFromBinary filepath = do
  binary <- B.readFile filepath
  case decode binary of
    Left peek_exception -> error $ "Could not parse dic file " ++ filepath ++ ": " ++ (show peek_exception)
    Right actions -> return actions


traverseCFGs :: [CFGtree] -> [RNNGSentence]
traverseCFGs = map (reverseRNNGSentence . traverseCFG (RNNGSentence ([], [])))

reverseRNNGSentence :: RNNGSentence -> RNNGSentence
reverseRNNGSentence (RNNGSentence (words, actions)) = RNNGSentence ((reverse words), (reverse actions))

traverseCFG :: RNNGSentence -> CFGtree -> RNNGSentence
-- POSタグは無視する
traverseCFG (RNNGSentence (words, actions)) (Phrase (_, Word word:rest)) =
  RNNGSentence (word:words, SHIFT:actions)
traverseCFG (RNNGSentence (words, actions)) (Phrase (label, trees)) =
  RNNGSentence (newWords, REDUCE:newActions)
  where
    RNNGSentence (newWords, newActions) = L.foldl traverseCFG (RNNGSentence (words, NT label:actions)) trees
traverseCFG (RNNGSentence (words, actions)) (Err message text)  = RNNGSentence (words, ERROR:actions)


printCFGtrees :: [CFGtree] -> IO ()
printCFGtrees cfgTree = do
  T.putStrLn $ T.unlines $ map (formatCFGtree 0) cfgTree

formatCFGtree :: Int -> CFGtree -> T.Text
formatCFGtree depth (Phrase (label, (Word word):rest)) =
  T.concat [
    T.replicate depth (T.pack "\t"),
    T.pack " (",
    label,
    T.pack " ",
    word,
    T.pack " )"
  ]
formatCFGtree depth (Phrase (label, tree)) =
  T.concat [
    T.replicate depth (T.pack "\t"),
    T.pack " (",
    label,
    T.pack "\n",
    T.intercalate (T.pack "\n") $ map (formatCFGtree (depth + 1)) tree,
    T.pack " )"
  ]
formatCFGtree depth (Err msg text) =
  T.intercalate (T.pack "\n") [
    T.pack $ "Parse Error: " ++ msg ++ " in ",
    text
  ]

isErr :: CFGtree -> Bool
isErr cfg = case cfg of
  Word _ -> False
  Phrase _ -> False
  Err _ _ -> True

parsePTBfile :: FilePath -> IO [CFGtree]
parsePTBfile ptbFilePath = do
  ptb <- T.readFile ptbFilePath
  return $ parseCFGtrees ptb

parseCFGtrees :: T.Text -> [CFGtree]
parseCFGtrees text =
  case parse cfgsParser "" text of
    Left e -> [Err (show e) text]
    Right t -> t

cfgsParser :: Parser [CFGtree]
cfgsParser = do
  _ <- optional blank
  _ <- optional $ string copyRight
  _ <- optional blank
  trees <- sepBy1' cfgParser blank
  return trees

cfgParser :: Parser CFGtree
cfgParser = do
  openParen
  optional blank
  tree <- (phraseParser <|> wordParser)
  _ <- char ')' <|> (blank >> char ')')
  return tree

blank :: Parser ()
blank = do
  _ <- many1 $ oneOf " \t\n"
  return ()

literal :: Parser T.Text
literal = T.pack <$> (many1 $ noneOf " ()\n\t")

openParen :: Parser T.Text
openParen = T.singleton <$> char '('

closeParen :: Parser T.Text
closeParen = T.singleton <$> char ')'

phraseParser :: Parser CFGtree
phraseParser = do
  openParen
  label <- literal
  blank
  tree <- (phraseParser <|> wordParser) `sepBy1'` blank
  closeParen <|> (blank >> closeParen)
  return $ Phrase (label, tree)

wordParser :: Parser CFGtree
wordParser = do
  word <- literal
  return $ Word word

sepBy1' :: (Stream s m t) => ParsecT s u m a -> ParsecT s u m sep -> ParsecT s u m [a]
{-# INLINABLE sepBy1' #-}
sepBy1' p sep = do 
  x <- p
  xs <- many $ try (sep >> p)
  return $ x:xs

copyRight = "*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*\n\
  \*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*\n\
  \*x*                                                                     *x*\n\
  \*x*            Copyright (C) 1995 University of Pennsylvania            *x*\n\
  \*x*                                                                     *x*\n\
  \*x*    The data in this file are part of a preliminary version of the   *x*\n\
  \*x*    Penn Treebank Corpus and should not be redistributed.  Any       *x*\n\
  \*x*    research using this corpus or based on it should acknowledge     *x*\n\
  \*x*    that fact, as well as the preliminary nature of the corpus.      *x*\n\
  \*x*                                                                     *x*\n\
  \*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*\n\
  \*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*"