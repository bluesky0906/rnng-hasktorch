{-#  LANGUAGE DeriveGeneric #-}

module Data.SyntaxTree where
import Data.RNNGSentence
import GHC.Generics
import qualified Data.Text as T          --text
import qualified Data.Text.IO as T       --text
import qualified Data.Aeson            as A --aeson
import Data.List as L

data Tree = 
  Phrase (T.Text, [Tree]) 
  | Word T.Text
  | Err String T.Text 
  deriving (Eq, Generic)

instance A.FromJSON Tree
instance A.ToJSON Tree


isErr :: 
  Tree -> 
  Bool
isErr cfg = case cfg of
  Word _ -> False
  Phrase _ -> False
  Err _ _ -> True


fromRNNGSentences :: [RNNGSentence] -> [Tree]
fromRNNGSentences = fmap fromRNNGSentence'
  where
    fromRNNGSentence' :: RNNGSentence -> Tree
    fromRNNGSentence' rnngSentence = 
      case fromRNNGSentence [] rnngSentence of
        Left e -> Err e (T.pack $ show rnngSentence)
        Right tree -> tree

fromRNNGSentence :: [Tree] -> RNNGSentence -> Either String Tree
fromRNNGSentence stack (RNNGSentence ([], [])) =
  if length stack == 1
    then Right (head stack)
    else Left "Invalid Tree"
fromRNNGSentence stack  (RNNGSentence (word:words, SHIFT:actions)) =
  fromRNNGSentence (Word word:stack) (RNNGSentence (words, actions))
fromRNNGSentence stack (RNNGSentence (words, (NT label):actions)) =
  fromRNNGSentence (Phrase (label, []):stack) (RNNGSentence (words, actions))
fromRNNGSentence stack (RNNGSentence (words, REDUCE:actions)) =
  case reduce stack [] of
    Just newStack -> fromRNNGSentence newStack (RNNGSentence (words, actions))
    Nothing -> Left "Invalid Tree"
  where
    reduce :: [Tree] -> [Tree] -> Maybe [Tree]
    reduce ((Phrase (label, [])):rest) lst = Just (Phrase (label, lst):rest)
    reduce (tree:rest) lst = reduce rest (tree:lst)
    reduce [] lst = Nothing
fromRNNGSentence _ _  =
  Left "Invalid Tree"

toRNNGSentences :: Bool -> [Tree] -> [RNNGSentence]
toRNNGSentences posMode = map (reverseRNNGSentence . toRNNGSentence posMode (RNNGSentence ([], [])))

toRNNGSentence :: Bool -> RNNGSentence -> Tree -> RNNGSentence
toRNNGSentence True (RNNGSentence (words, actions)) (Phrase (label, Word word:rest)) =
  RNNGSentence (word:words, REDUCE:SHIFT:NT label:actions)
-- POSタグは無視する
toRNNGSentence False (RNNGSentence (words, actions)) (Phrase (_, Word word:rest)) =
  RNNGSentence (word:words, SHIFT:actions)
toRNNGSentence posMode (RNNGSentence (words, actions)) (Phrase (label, trees)) =
  RNNGSentence (newWords, REDUCE:newActions)
  where
    RNNGSentence (newWords, newActions) = L.foldl (toRNNGSentence posMode) (RNNGSentence (words, NT label:actions)) trees
toRNNGSentence _ (RNNGSentence (words, actions)) (Err message text)  = RNNGSentence (words, ERROR:actions)


instance Show Tree where
  show cfgTree = T.unpack $ formatCFGtree 0 cfgTree

formatCFGtree ::
  Int ->
  Tree ->
  T.Text
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
