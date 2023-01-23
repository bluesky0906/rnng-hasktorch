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
  deriving (Eq, Show, Generic)

instance A.FromJSON Tree
instance A.ToJSON Tree


isErr :: 
  Tree -> 
  Bool
isErr cfg = case cfg of
  Word _ -> False
  Phrase _ -> False
  Err _ _ -> True


toRNNGSentences :: Bool -> [Tree] -> [RNNGSentence]
toRNNGSentences posMode = map (reverseRNNGSentence . toRNNGSentence posMode (RNNGSentence ([], [])))

toRNNGSentence :: Bool -> RNNGSentence -> Tree -> RNNGSentence
toRNNGSentence True (RNNGSentence (words, actions)) (Phrase (label, Word word:rest)) =
  RNNGSentence (word:words, SHIFT:NT label:actions)
-- POSタグは無視する
toRNNGSentence False (RNNGSentence (words, actions)) (Phrase (_, Word word:rest)) =
  RNNGSentence (word:words, SHIFT:actions)
toRNNGSentence posMode (RNNGSentence (words, actions)) (Phrase (label, trees)) =
  RNNGSentence (newWords, REDUCE:newActions)
  where
    RNNGSentence (newWords, newActions) = L.foldl (toRNNGSentence posMode) (RNNGSentence (words, NT label:actions)) trees
toRNNGSentence _ (RNNGSentence (words, actions)) (Err message text)  = RNNGSentence (words, ERROR:actions)


printTrees :: 
  [Tree] ->
  IO ()
printTrees cfgTree = do
  T.putStrLn $ T.unlines $ map (formatCFGtree 0) cfgTree

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
