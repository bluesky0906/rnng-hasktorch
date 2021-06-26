module PTB where
import qualified Data.Text as T          --text
import qualified Data.Text.IO as T       --text
import Text.Parsec      --parsec
import Text.Parsec.Text --parsec


data CFGdata = NonTerminal (T.Text, [CFGdata]) | Terminal (T.Text, T.Text) | Err String T.Text deriving (Eq, Show)

printCFGdata :: [CFGdata] -> IO ()
printCFGdata cfgData = do
  T.putStrLn $ T.unlines $ map (formatCFGdata 0) cfgData

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
  case parse ((try terminal) <|> nonterminal) "" text of
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
  tree <- ((try terminal) <|> nonterminal) `sepBy1` (char ' ')
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
