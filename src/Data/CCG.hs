{-# LANGUAGE ScopedTypeVariables #-}
{-# OPTIONS_GHC -Wno-incomplete-patterns #-}

module Data.CCG where
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
