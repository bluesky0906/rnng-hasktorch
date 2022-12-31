{-#  LANGUAGE DeriveGeneric #-}

module Data.RNNGSentence where

import GHC.Generics
import qualified Data.Text as T          --text
import qualified Data.Text.IO as T       --text
import qualified Data.Aeson            as A --aeson
import qualified Data.ByteString.Char8 as B --bytestring 
import qualified Data.ByteString as B --bytestring 
import qualified Data.ByteString.Lazy as BL
import qualified Data.Yaml             as Y --yaml
import Text.Parsec      --parsec
import Text.Parsec.Text --parsec
import Data.Store --seralisation
import ML.Util.Dict (sortWords)


data Grammar = CFG | CCG deriving (Show, Read)


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

reverseRNNGSentence :: RNNGSentence -> RNNGSentence
reverseRNNGSentence (RNNGSentence (words, actions)) = RNNGSentence ((reverse words), (reverse actions))

saveActionsToBinary :: FilePath -> [RNNGSentence] -> IO()
saveActionsToBinary filepath actions = B.writeFile filepath (encode actions)

loadActionsFromBinary :: FilePath -> IO [RNNGSentence]
loadActionsFromBinary filepath = do
  binary <- B.readFile filepath
  case decode binary of
    Left peek_exception -> error $ "Could not parse dic file " ++ filepath ++ ": " ++ (show peek_exception)
    Right actions -> return actions

extractSentences :: [RNNGSentence] -> [Sentence]
extractSentences [] = []
extractSentences ((RNNGSentence (words, _)):rest) = words:(extractSentences rest)

toWordList :: [RNNGSentence] -> [T.Text]
toWordList [] = []
toWordList ((RNNGSentence (words, _)):rest) = words ++ toWordList rest

toActionList :: [RNNGSentence] -> [Action]
toActionList [] = []
toActionList ((RNNGSentence (_, actions)):rest) = actions ++ toActionList rest

extractNT :: [Action] -> [T.Text]
extractNT [] = []
extractNT ((NT label):rest) = label:(extractNT rest)
extractNT (_:rest) = extractNT rest

toNTList :: [RNNGSentence] -> [T.Text]
toNTList [] = []
toNTList ((RNNGSentence (_, actions)):rest) = extractNT actions ++ toNTList rest

buildVocab :: 
  (Ord a) =>
  -- | training data
  [RNNGSentence] ->
  -- | 出現頻度threshold
  Int ->
  -- | 語彙リストを作る関数
  ([RNNGSentence] -> [a])
  -- | 一意な語彙リスト
  -> [a]
buildVocab rnngData freq toList = sortWords freq (toList rnngData)
