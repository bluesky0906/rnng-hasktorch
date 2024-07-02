{-#  LANGUAGE DeriveGeneric #-}

-- TODO:: library名再考（Treebankファイルを扱うための関数も含む）
module Data.SyntaxTree where
import Data.RNNGSentence
import GHC.Generics
import qualified Data.Text as T          --text
import qualified Data.Text.IO as T       --text
import qualified Data.Aeson            as A --aeson
import Data.List as L
import Text.Directory (getFileList) --nlp-tools
import System.Directory (doesFileExist) --directory
import System.FilePath.Posix (takeBaseName) --filepath

data Tree = 
  Phrase (T.Text, [Tree]) 
  | Word T.Text
  | Err String T.Text 
  deriving (Eq, Generic)

instance Show Tree where
  show tree = T.unpack $ formatSyntaxTree 0 tree

instance A.FromJSON Tree
instance A.ToJSON Tree

data Grammar = CFG | CCG deriving (Show, Read, Eq)

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

formatSyntaxTree ::
  Int ->
  Tree ->
  T.Text
formatSyntaxTree depth (Phrase (label, tree)) =
  T.concat [
    T.pack "\n",
    T.replicate depth (T.pack "\t"),
    T.pack " (",
    label,
    -- T.pack "\n",
    T.concat $ map (formatSyntaxTree (depth + 1)) tree,
    -- T.pack "\n",
    T.pack " )"
    -- T.pack "\n"
  ]
formatSyntaxTree depth (Word word) =
  T.concat [
    T.pack " ",
    word
  ]
formatSyntaxTree depth (Err msg text) =
  T.intercalate (T.pack "\n") [
    T.pack $ "Parse Error: " ++ msg ++ " in ",
    text
  ]

-- 評価のために()を<>に置き換え
replaceBrackets ::
  T.Text ->
  T.Text 
replaceBrackets =
  T.pack . (T.foldr replaceBracket "")
  where
    replaceBracket '(' acc = '<':acc
    replaceBracket ')' acc = '>':acc
    replaceBracket x acc = x:acc


evalFormat ::
  Bool ->
  Tree ->
  T.Text
evalFormat posMode (Phrase (label, tree)) =
  T.concat [
    T.pack " ",
    T.pack "(", 
    replaceBrackets label,
    T.concat $ map (evalFormat posMode) tree,
    T.pack ")"
  ]
evalFormat True (Word word) =
  T.concat [
    T.pack " ",
    word
  ]
evalFormat False (Word word) =
  T.concat [
    T.pack " ",
    T.pack "(WORD ",
    word,
    T.pack ")"
  ]

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

treefiles ::
  Grammar ->
  [FilePath] -> 
  IO [FilePath]
treefiles grammar dirsPath = do
  -- 指定されたディレクトリ以下のファイルを取得
  filePaths <- concat <$> traverse (listFiles grammar) dirsPath
  --readmeは除外
  return $ filter (\f -> takeBaseName f /= "readme") filePaths
