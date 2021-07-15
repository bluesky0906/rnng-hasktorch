{-#  LANGUAGE DeriveGeneric #-}

module PTB where
import GHC.Generics
import qualified Data.Text as T          --text
import qualified Data.Text.IO as T       --text
import qualified Data.Aeson            as A --aeson
import qualified Data.ByteString.Char8 as B --bytestring 
import qualified Data.Yaml             as Y --yaml
import Data.List as L
import Text.Parsec
    ( char, sepBy1, noneOf, many1, (<|>), try, parse )      --parsec
import Text.Parsec.Text --parsec


data CFGdata = NonTerminal (T.Text, [CFGdata]) | Terminal (T.Text, T.Text) | Err String T.Text deriving (Eq, Show, Generic)

instance A.FromJSON CFGdata
instance A.ToJSON CFGdata

data Action = NT T.Text | SHIFT | REDUCE | ERROR deriving (Eq, Show, Generic)
type Stack = [T.Text]

instance A.FromJSON Action
instance A.ToJSON Action

newtype CFGActionData =  CFGActionData (CFGdata, Action) deriving (Eq, Show, Generic)

instance A.FromJSON CFGActionData
instance A.ToJSON CFGActionData


saveCFGData :: FilePath -> [CFGActionData] -> IO()
saveCFGData = Y.encodeFile 

loadCFGData :: FilePath -> IO [CFGActionData]
loadCFGData filepath = do
  -- checkFile filepath
  content <- B.readFile filepath
  let parsedDic = Y.decodeEither' content :: Either Y.ParseException [CFGActionData]
  case parsedDic of
    Left parse_exception -> error $ "Could not parse dic file " ++ filepath ++ ": " ++ (show parse_exception)
    Right dic -> return dic


reduce :: Stack -> T.Text -> (Stack, T.Text)
reduce (top:rest) ans = if top == T.pack "(" 
                          then (T.concat [T.pack "(", ans, T.pack ")"]:rest, ans)
                          else reduce rest $ top `T.append` T.pack " " `T.append` ans

traverseCFG :: (Stack, [Action]) -> CFGdata -> (Stack, [Action])
traverseCFG (stack, actions) (NonTerminal (label, trees)) =
  (fst $ reduce newStack (T.pack ""), REDUCE:newActions)
  where 
    (newStack, newActions) = L.foldl' traverseCFG (label:T.pack "(":stack, NT label:actions)　trees 
traverseCFG (stack, actions) (Terminal (label, word)) = (word:stack, SHIFT:actions)
traverseCFG (stack, actions) (Err message _) = (T.pack message:stack, ERROR:actions)

traverseCFG' :: [(Stack, Action)] -> CFGdata -> [(Stack, Action)]
traverseCFG' [] (NonTerminal (label, trees)) =
  (fst $ reduce newStack (T.pack ""), REDUCE):newHistory
  where
    (newStack, newActions) = head newHistory
    newHistory = L.foldl' traverseCFG' [([label, T.pack "("], NT label)]　trees 
traverseCFG' history (NonTerminal (label, trees)) =
  (fst $ reduce newStack (T.pack ""), REDUCE):newHistory
  where
    (newStack, newActions) = head newHistory
    newHistory = L.foldl' traverseCFG' ((label:T.pack "(":fst (head history), NT label):history)　trees 
traverseCFG' history (Terminal (label, word)) = (word:fst (head history), SHIFT):history
traverseCFG' history (Err message _) = (T.pack message:fst (head history), ERROR):history

printCFGdata :: [CFGdata] -> IO ()
printCFGdata cfgData = T.putStrLn $ T.unlines $ map (formatCFGdata 0) cfgData

formatCFGdata :: Int -> CFGdata -> T.Text
formatCFGdata depth (NonTerminal (label, tree)) =
  T.concat [
    T.replicate depth (T.pack "\t"),
    T.pack " (",
    label,
    T.pack "\n",
    T.intercalate (T.pack "\n") $ map (formatCFGdata (depth + 1)) tree,
    T.pack " )"
  ]
formatCFGdata depth (Terminal (label, word)) =
  T.concat [
    T.replicate depth (T.pack "\t"),
    T.pack " (",
    label,
    T.pack " ",
    word,
    T.pack " )"
  ]
formatCFGdata depth (Err msg text) =
  T.intercalate (T.pack "\n") [
    T.pack $ "Parse Error: " ++ msg ++ " in ",
    text
  ]

cfgParser :: T.Text -> CFGdata
cfgParser text =
  case parse (try terminal <|> nonterminal) "" text of
    Left e -> Err (show e) text
    Right t -> t

sep :: Parser ()
sep = do
  _ <- char ' '
  return ()

literal :: Parser T.Text
literal = T.pack <$> (many1 $ noneOf " ()")

openParen :: Parser T.Text
openParen = T.singleton <$> char '('

closeParen :: Parser T.Text
closeParen = T.singleton <$> char ')'

nonterminal :: Parser CFGdata
nonterminal = do
  openParen
  label <- literal
  sep
  tree <- (try terminal <|> nonterminal) `sepBy1` char ' '
  closeParen
  return $ NonTerminal (label, tree)

terminal :: Parser CFGdata
terminal = do
  openParen
  label <- literal
  sep
  word <- literal
  closeParen
  return $ Terminal (label, word)
