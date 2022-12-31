{-#  LANGUAGE DeriveGeneric #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# OPTIONS_GHC -Wno-incomplete-patterns #-}

module Data.CFG where
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
import Debug.Trace
import Text.Parsec      --parsec
import Text.Parsec.Text --parsec
import Data.Store --seralisation


{-

parse text file to CFGTree

-}

parseCFGfile :: FilePath -> IO [Tree]
parseCFGfile ptbFilePath = do
  ptb <- T.readFile ptbFilePath
  return $ parseCFGtrees ptb

parseCFGtrees :: T.Text -> [Tree]
parseCFGtrees text =
  case parse cfgsParser "" text of
    Left e -> [Err (show e) text]
    Right t -> t

cfgsParser :: Parser [Tree]
cfgsParser = do
  _ <- optional blank
  _ <- optional $ string copyRight
  _ <- optional blank
  trees <- sepBy1' cfgParser blank
  return trees

cfgParser :: Parser Tree
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

phraseParser :: Parser Tree
phraseParser = do
  openParen
  label <- literal
  blank
  tree <- (phraseParser <|> wordParser) `sepBy1'` blank
  closeParen <|> (blank >> closeParen)
  return $ Phrase (label, tree)

wordParser :: Parser Tree
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