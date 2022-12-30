{-#  LANGUAGE DeriveGeneric #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# OPTIONS_GHC -Wno-incomplete-patterns #-}

module Data.CCG where
import Data.RNNGSentence
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

define data structure

-}

data CCGTree = 
  Phrase (T.Text, [CCGTree]) 
  | Word T.Text
  | Err String T.Text 
  deriving (Eq, Show, Generic)

instance A.FromJSON CCGTree
instance A.ToJSON CCGTree


{-

save and load data

-}

saveActionsToBinary :: FilePath -> [RNNGSentence] -> IO()
saveActionsToBinary filepath actions = B.writeFile filepath (encode actions)

loadActionsFromBinary :: FilePath -> IO [RNNGSentence]
loadActionsFromBinary filepath = do
  binary <- B.readFile filepath
  case decode binary of
    Left peek_exception -> error $ "Could not parse dic file " ++ filepath ++ ": " ++ (show peek_exception)
    Right actions -> return actions


{-

util function

-}

isErr :: CCGTree -> Bool
isErr ccg = case ccg of
  Word _ -> False
  Phrase _ -> False
  Err _ _ -> True


showAction :: Action -> T.Text
showAction (NT label) = (T.pack "NT_") <> label
showAction SHIFT = T.pack "SHIFT"
showAction REDUCE = T.pack "REDUCE"
showAction ERROR = T.pack "ERROR"


{-

traverse CCGTree to RNNGSentence

-}

traverseCCGs :: [CCGTree] -> [RNNGSentence]
traverseCCGs = map (reverseRNNGSentence . traverseCCG (RNNGSentence ([], [])))

reverseRNNGSentence :: RNNGSentence -> RNNGSentence
reverseRNNGSentence (RNNGSentence (words, actions)) = RNNGSentence ((reverse words), (reverse actions))

traverseCCG :: RNNGSentence -> CCGTree -> RNNGSentence
traverseCCG (RNNGSentence (words, actions)) (Word word) =
  RNNGSentence (word:words, SHIFT:actions)
traverseCCG (RNNGSentence (words, actions)) (Phrase (label, trees)) =
  RNNGSentence (newWords, REDUCE:newActions)
  where
    RNNGSentence (newWords, newActions) = L.foldl traverseCCG (RNNGSentence (words, NT label:actions)) trees
traverseCCG (RNNGSentence (words, actions)) (Err message text)  = RNNGSentence (words, ERROR:actions)


{-

pretty printing of CCGTree

-}

printCCGTrees :: [CCGTree] -> IO ()
printCCGTrees ccgTree =
  T.putStrLn $ T.unlines $ map (formatCCGTree 0) ccgTree

formatCCGTree :: Int -> CCGTree -> T.Text
formatCCGTree depth (Phrase (label, (Word word):rest)) =
  T.concat [
    T.replicate depth (T.pack "\t"),
    T.pack " (",
    label,
    T.pack " ",
    word,
    T.pack " )"
  ]
formatCCGTree depth (Phrase (label, tree)) =
  T.concat [
    T.replicate depth (T.pack "\t"),
    T.pack " (",
    label,
    T.pack "\n",
    T.intercalate (T.pack "\n") $ map (formatCCGTree (depth + 1)) tree,
    T.pack " )"
  ]
formatCCGTree depth (Err msg text) =
  T.intercalate (T.pack "\n") [
    T.pack $ "Parse Error: " ++ msg ++ " in ",
    text
  ]


{-

parse text file to CCGTree

-}

parseCCGfile :: FilePath -> IO [CCGTree]
parseCCGfile ptbFilePath = do
  ptb <- T.readFile ptbFilePath
  return $ parseCCGTrees ptb

parseCCGTrees :: T.Text -> [CCGTree]
parseCCGTrees text =
  case parse ccgsParser "" text of
    Left e -> [Err (show e) text]
    Right t -> t

ccgsParser :: Parser [CCGTree]
ccgsParser = many1 ccgParser

ccgParser :: Parser CCGTree
ccgParser = do
  ignoreSentenceInfo
  tree <- try nonLeafParser <|> leafParser
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

leafParser :: Parser CCGTree
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
  return $ Word word

nonLeafParser :: Parser CCGTree
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
  tree <- many (try leafParser <|> try nonLeafParser) 
  closeParen
  blank
  return $ Phrase (category, tree)

sepBy1' :: (Stream s m t) => ParsecT s u m a -> ParsecT s u m sep -> ParsecT s u m [a]
{-# INLINABLE sepBy1' #-}
sepBy1' p sep = do 
  x <- p
  xs <- many $ try (sep >> p)
  return $ x:xs
