
module Data.CCGRule where
import Data.SyntaxTree
import qualified Data.Text as T          --text
import qualified Data.Text.IO as T       --text
import Text.Parsec
import Text.Parsec.Text --parsec


{-

parse text file to Category

-}

--TODO: GCATEGORY以外についた[]や複数の[]処理
data Category = SLASH (Category, Category) | BSLASH (Category, Category) | GCATEGORY (T.Text, T.Text)

instance Show Category where
  show (SLASH (left, right)) = "(" ++ show left ++ "/" ++ show right ++ ")"
  show (BSLASH (left, right)) = "(" ++ show left ++ "\\" ++ show right ++ ")"
  show (GCATEGORY (cat, subCat)) = T.unpack cat ++ "[" ++ T.unpack subCat ++ "]"

instance Eq Category where
  GCATEGORY (cat1, subCat1) == GCATEGORY (cat2, subCat2) =
    (cat1 == cat2) && ((subCat1 == subCat2) || (subCat1 == T.empty) || (subCat2 == T.empty))
  (SLASH (left1, right1)) == (SLASH (left2, right2)) = 
    (left1 == left2) && (right1 == right2)
  (BSLASH (left1, right1)) == (BSLASH (left2, right2)) = 
    (left1 == left2) && (right1 == right2)
  _ == _ = False
  category1 /= category2 = not (category1 == category2)

subCategory :: Parser T.Text
subCategory = do
  T.singleton <$> char '['
  subCat <- T.pack <$> many1 (noneOf "[]()/\\")
  T.singleton <$> char ']'
  return subCat

groundCategory :: Parser Category
groundCategory = do
  category <- T.pack <$> many1 (noneOf "[]()/\\")
  subCat <- optionMaybe subCategory
  case subCat of
    Just a -> return $ GCATEGORY (category, a)
    Nothing -> return $ GCATEGORY (category, T.empty)

parenCategory :: Parser Category
parenCategory = do
  optional $ T.singleton <$> char '('
  left <- groundCategory <|> parenCategory
  slashOrBslash <- T.singleton <$> oneOf "/\\"
  right <- groundCategory <|> parenCategory
  optional $ T.singleton <$> char '('
  if slashOrBslash == T.pack "/"
    then return $ SLASH (left, right)
    else return $ BSLASH (left, right)

nonParenCategory :: Parser Category
nonParenCategory = do
  left <- groundCategory <|> parenCategory
  slashOrBslash <- T.singleton <$> oneOf "/\\"
  right <- groundCategory <|> parenCategory
  if slashOrBslash == T.pack "/"
    then return $ SLASH (left, right)
    else return $ BSLASH (left, right)

categoryParser :: Parser Category
categoryParser = try nonParenCategory <|> groundCategory

parseCategory :: T.Text -> Category
parseCategory category = 
  case parse categoryParser "" category of
    Left e -> error $ show e
    Right c -> c


{-

CCG Rules

-}


-- CCG combinatory Rules

-- X/Y Y => X
forwardFunctionalAplicationRule ::
  -- | parents
  (Tree, Tree) ->
  -- |　　child
  Tree ->
  Bool
forwardFunctionalAplicationRule (Phrase (cat1, _), Phrase (cat2, _)) (Phrase (childCat, _)) =
  case parseCategory cat1 of
    (SLASH (left, right)) -> (parseCategory cat2 == right) && (parseCategory childCat == left)
    _ -> False
forwardFunctionalAplicationRule _ _ = False

-- Y X\Y => X
backwardFunctionalAplicationRule :: 
  -- | parents
  (Tree, Tree) ->
  -- |　　child
  Tree ->
  Bool
backwardFunctionalAplicationRule (Phrase (cat1, _), Phrase (cat2, _)) (Phrase (childCat, _)) =
  case parseCategory cat2 of
    (BSLASH (left, right)) -> (parseCategory cat1 == right) && (parseCategory childCat == left)
    _ -> False
backwardFunctionalAplicationRule _ _ = False

-- X/Y Y/Z => X/Z
forwardCompositionRule :: 
  -- | parents
  (Tree, Tree) ->
  -- |　　child
  Tree ->
  Bool
forwardCompositionRule (Phrase (cat1, _), Phrase (cat2, _)) (Phrase (childCat, _)) =
  case parseCategory cat1 of
    (SLASH (left1, right1)) -> case parseCategory cat2 of
                                 (SLASH (left2, right2)) -> (right1 == left2) && (parseCategory childCat == SLASH(left1, right2))
                                 _ -> False
    _ -> False
forwardCompositionRule _ _ = False

-- Y\Z X\Y => X\Z
backwardCompositionRule :: 
  -- | parents
  (Tree, Tree) ->
  -- |　　child
  Tree ->
  Bool
backwardCompositionRule (Phrase (cat1, _), Phrase (cat2, _)) (Phrase (childCat, _)) =
  case parseCategory cat1 of
    (BSLASH (left1, right1)) -> case parseCategory cat2 of
                                 (BSLASH (left2, right2)) -> (left1 == right2) && (parseCategory childCat == BSLASH(left2, right1))
                                 _ -> False
    _ -> False
backwardCompositionRule _ _ = False

-- X/Y Y\Z => X\Z
forwardCrossingCompositionRule :: 
  -- | parents
  (Tree, Tree) ->
  -- |　　child
  Tree ->
  Bool
forwardCrossingCompositionRule (Phrase (cat1, _), Phrase (cat2, _)) (Phrase (childCat, _)) =
  case parseCategory cat1 of
    (SLASH (left1, right1)) -> case parseCategory cat2 of
                                 (BSLASH (left2, right2)) -> (right1 == left2) && (parseCategory childCat == BSLASH(left1, right2))
                                 _ -> False
    _ -> False
forwardCrossingCompositionRule _ _ = False

-- Y/Z X\Y => X/Z
backwardCrossingCompositionRule :: 
  -- | parents
  (Tree, Tree) ->
  -- |　　child
  Tree ->
  Bool
backwardCrossingCompositionRule (Phrase (cat1, _), Phrase (cat2, _)) (Phrase (childCat, _)) =
  case parseCategory cat1 of
    (SLASH (left1, right1)) -> case parseCategory cat2 of
                                 (BSLASH (left2, right2)) -> (left1 == right2) && (parseCategory childCat == SLASH(left2, right1))
                                 _ -> False
    _ -> False
backwardCrossingCompositionRule _ _ = False

-- X/Y (Y/Z)/$ => (X/Z)/$ ?
generalizedForwardCompositionRule ::
  -- | parents
  (Tree, Tree) ->
  -- |　　child
  Tree ->
  Bool
generalizedForwardCompositionRule (Phrase (cat1, _), Phrase (cat2, _)) (Phrase (childCat, _)) =
  generalized (parseCategory cat1, parseCategory cat2) (parseCategory childCat) 1
  where
    generalized :: (Category, Category) -> Category -> Int -> Bool
    generalized _ _ 5 = False
    generalized (parent1@(SLASH (left1, right1)), SLASH (left2, right2)) child@(SLASH (childLeft, childRight)) depth =
      ((left1 == childLeft) && (right1 == left2) && (right2 == childRight)) ||
      ((right2 == childRight) && generalized (parent1, left2) childLeft (depth + 1))
    -- X/Y ?\$ ?\$
    generalized (parent1@(SLASH (left1, right1)), BSLASH (left2, right2)) (BSLASH (childLeft, childRight)) depth =
      (right2 == childRight) && generalized (parent1, left2) childLeft (depth + 1)
    generalized _ _ _ = False
generalizedForwardCompositionRule _ _ = False

-- (Y\Z)\$ X\Y => (X\Z)\$
generalizedBackwardCompositionRule ::
  -- | parents
  (Tree, Tree) ->
  -- |　　child
  Tree ->
  Bool
generalizedBackwardCompositionRule (Phrase (cat1, _), Phrase (cat2, _)) (Phrase (childCat, _)) =
  generalized (parseCategory cat1, parseCategory cat2) (parseCategory childCat) 1
  where
    generalized :: (Category, Category) -> Category -> Int -> Bool
    generalized _ _ 5 = False
    generalized (BSLASH (left1, right1), parent2@(BSLASH (left2, right2))) child@(BSLASH (childLeft, childRight)) depth =
      ((left2 == childLeft) && (left1 == right2) && (right1 == childRight)) ||
      ((right1 == childRight) && generalized (left1, parent2) childLeft (depth + 1))
    generalized _ _ _ = False
generalizedBackwardCompositionRule _ _ = False

-- X/Y (Y\Z)$ => (X\Z)$
generalizedForwardCrossingCompositionRule ::
  -- | parents
  (Tree, Tree) ->
  -- |　　child
  Tree ->
  Bool
generalizedForwardCrossingCompositionRule (Phrase (cat1, _), Phrase (cat2, _)) (Phrase (childCat, _)) =
  generalized (parseCategory cat1, parseCategory cat2) (parseCategory childCat) 1
  where
    generalized :: (Category, Category) -> Category -> Int -> Bool
    generalized _ _ 5 = False
    -- X/Y (Y\Z) (X\Z)
    generalized (parent1@(SLASH (left1, right1)), BSLASH (left2, right2)) child@(BSLASH (childLeft, childRight)) depth =
      ((left1 == childLeft) && (right1 == left2) && (right2 == childRight)) ||
      ((right2 == childRight) && generalized (parent1, left2) childLeft (depth + 1))
    -- X/Y ?/$ ?/$
    generalized (parent1@(SLASH (left1, right1)), SLASH (left2, right2)) (SLASH (childLeft, childRight)) depth =
      (right2 == childRight) && generalized (parent1, left2) childLeft (depth + 1)
    generalized _ _ _ = False
generalizedForwardCrossingCompositionRule _ _ = False

-- (Y/Z)$ X\Y => (X/Z)$
generalizedBackwardCrossingCompositionRule :: 
  -- | parents
  (Tree, Tree) ->
  -- |　　child
  Tree ->
  Bool
generalizedBackwardCrossingCompositionRule (Phrase (cat1, _), Phrase (cat2, _)) (Phrase (childCat, _)) =
  generalized (parseCategory cat1, parseCategory cat2) (parseCategory childCat) 1
  where
    generalized :: (Category, Category) -> Category -> Int -> Bool
    generalized _ _ 5 = False
    -- Y/Z X\Y => X/Z
    generalized (SLASH (left1, right1), parent2@(BSLASH (left2, right2))) child@(SLASH (childLeft, childRight)) depth =
      ((left2 == childLeft) && (left1 == right2) && (right1 == childRight)) ||
      ((right1 == childRight) && generalized (left1, parent2) childLeft (depth + 1))
    -- -- ?/$ X\Y => ?/$
    -- generalized (SLASH (left1, right1), parent2@(BSLASH (_, _))) (SLASH (childLeft, childRight)) depth =
    --   (right1 == childRight) && generalized (left1, parent2) childLeft (depth + 1)
    -- ?\$ X\Y => ?\$
    generalized (BSLASH (left1, right1), parent2@(BSLASH (_, _))) (BSLASH (childLeft, childRight)) depth =
      (right1 == childRight) && generalized (left1, parent2) childLeft (depth + 1)
    generalized _ _ _ = False
generalizedBackwardCrossingCompositionRule _ _ = False

-- (Y\Z)$ X\Y => (X\Z)$ ??
generalizedBackwardCrossingCompositionRule2 :: 
  -- | parents
  (Tree, Tree) ->
  -- |　　child
  Tree ->
  Bool
generalizedBackwardCrossingCompositionRule2 (Phrase (cat1, _), Phrase (cat2, _)) (Phrase (childCat, _)) =
  generalized (parseCategory cat1, parseCategory cat2) (parseCategory childCat) 1
  where
    generalized :: (Category, Category) -> Category -> Int -> Bool
    generalized _ _ 5 = False
    -- Y/Z X\Y => X/Z
    generalized (BSLASH (left1, right1), parent2@(BSLASH (left2, right2))) child@(BSLASH (childLeft, childRight)) depth =
      ((left2 == childLeft) && (left1 == right2) && (right1 == childRight)) ||
      ((right1 == childRight) && generalized (left1, parent2) childLeft (depth + 1))
    -- ?/$ X\Y => ?/$
    generalized (SLASH (left1, right1), parent2@(BSLASH (_, _))) (SLASH (childLeft, childRight)) depth =
      (right1 == childRight) && generalized (left1, parent2) childLeft (depth + 1)
    generalized _ _ _ = False
generalizedBackwardCrossingCompositionRule2 _ _ = False

-- (X/Y)/Z Y/Z => X/Z
forwardSubstitutionRule ::
  -- | parents
  (Tree, Tree) ->
  -- |　　child
  Tree ->
  Bool
forwardSubstitutionRule (Phrase (cat1, _), Phrase (cat2, _)) (Phrase (childCat, _)) =
  case parseCategory cat1 of
    (SLASH (SLASH(left_left, left_right), right)) -> 
      (parseCategory cat2 == SLASH(left_right, right)) && (parseCategory childCat == SLASH(left_left, right))
    _ -> False
forwardSubstitutionRule _ _ = False

-- Y\Z (X\Y)\Z  => X\Z
backwardSubstitutionRule :: 
  -- | parents
  (Tree, Tree) ->
  -- |　　child
  Tree ->
  Bool
backwardSubstitutionRule (Phrase (cat1, _), Phrase (cat2, _)) (Phrase (childCat, _)) =
  case parseCategory cat2 of
    (BSLASH (BSLASH(left_left, left_right), right)) -> 
      (parseCategory cat1 == BSLASH(left_right, right)) && (parseCategory childCat == BSLASH(left_left, right))
    _ -> False
backwardSubstitutionRule _ _ = False

-- (X/Y)\Z Y\Z => X\Z
forwardCrossingSubstitutionRule ::
  -- | parents
  (Tree, Tree) ->
  -- |　　child
  Tree ->
  Bool
forwardCrossingSubstitutionRule (Phrase (cat1, _), Phrase (cat2, _)) (Phrase (childCat, _)) =
  case parseCategory cat1 of
    (BSLASH (SLASH(left_left, left_right), right)) -> 
      (parseCategory cat2 == BSLASH(left_right, right)) && (parseCategory childCat == BSLASH(left_left, right))
    _ -> False
forwardCrossingSubstitutionRule _ _ = False

-- Y/Z (X\Y)/Z  => X/Z
backwardCrossingSubstitutionRule :: 
  -- | parents
  (Tree, Tree) ->
  -- |　　child
  Tree ->
  Bool
backwardCrossingSubstitutionRule (Phrase (cat1, _), Phrase (cat2, _)) (Phrase (childCat, _)) =
  case parseCategory cat2 of
    (SLASH (BSLASH(left_left, left_right), right)) -> 
      (parseCategory cat1 == SLASH(left_right, right)) && (parseCategory childCat == SLASH(left_left, right))
    _ -> False
backwardCrossingSubstitutionRule _ _ = False

forwardTypeRaisingRule ::
  -- | parents
  Tree ->
  -- |　　child
  Tree ->
  Bool
forwardTypeRaisingRule (Phrase (cat, _)) (Phrase (childCat, _)) =
  case parseCategory childCat of
    SLASH(left, right) -> right == BSLASH (left, parseCategory cat)
    _ -> False
forwardTypeRaisingRule _ _ = False

backwardTypeRaisingRule ::
  -- | parents
  Tree ->
  -- |　　child
  Tree ->
  Bool
backwardTypeRaisingRule (Phrase (cat, _)) (Phrase (childCat, _)) =
  case parseCategory childCat of
    BSLASH(left, right) -> right == SLASH (left, parseCategory cat)
    _ -> False
backwardTypeRaisingRule _ _ = False

--non-combinatory rules

punctuationRule ::
  -- | parents
  (Tree, Tree) ->
  -- |　　child
  Tree ->
  Bool
punctuationRule (Phrase (cat1, _), Phrase (cat2, _)) (Phrase (childCat, _)) =
  -- X . => X |  X , => X | X ; => X | X : => X
  ((parseCategory cat1 == parseCategory childCat) && ((cat2 == T.pack ".") || (cat2 == T.pack ",") || (cat2 == T.pack ";") || (cat2 == T.pack ":"))) ||
  -- . X => X | , X => X | ; X => X | : X => X
  ((parseCategory cat2 == parseCategory childCat) && ((cat1 == T.pack ".") || (cat1 == T.pack ",") || (cat1 == T.pack ";") || (cat1 == T.pack ":")))
punctuationRule _ _ = False

coordinationRules ::
  -- | parents
  (Tree, Tree) ->
  -- |　　child
  Tree ->
  Bool
coordinationRules (Phrase (cat1, _), Phrase (cat2, _)) (Phrase (childCat, _)) =
  -- conj X => X[conj]
  ((cat1 == T.pack "conj") && (childCat == T.concat [cat2, T.pack "[conj]"])) ||
  -- conj X => X
  ((cat1 == T.pack "conj") && (childCat == cat2)) ||
  -- X conj => X
  ((childCat == cat1) && (cat2 == T.pack "conj")) ||
  -- conj X[conj] => X ??
  ((cat1 == T.pack "conj") && (cat2 == T.concat [childCat, T.pack "[conj]"])) ||
  -- conj Y => X[conj] (manual 3.7.2)
  ((cat1 == T.pack "conj") && T.isSuffixOf (T.pack "[conj]") childCat) ||
  -- X X[conj] => X
  ((T.concat [cat1, T.pack "[conj]"] == cat2) && (cat1 == childCat)) || 
  -- , X => X[conj] || : X => X[conj] || ; X => X[conj] || . X => X[conj]
  ((cat1 `elem` map T.pack [",", ":", ";", "."]) && (childCat == T.concat [cat2, T.pack "[conj]"])) ||
  -- , X[conj] => X || : X[conj] => X || . X[conj] => X
  ((cat1 `elem` map T.pack [",", ":", "."]) && (cat2 == T.concat [childCat, T.pack "[conj]"]))

unaryTypeChangingRules ::
  Tree ->
  Tree ->
  Bool
unaryTypeChangingRules (Phrase (cat, _)) (Phrase (childCat, _)) =
    -- ((S[b]\NP)/NP)/ => (S[b]\NP)/NP
  ((cat == T.pack "((S[b]\\NP)/NP)/") && (childCat == T.pack "(S[b]\\NP)/NP")) ||

  -- | X => Y
  -- N => NP
  ((cat == T.pack "N") && (childCat == T.pack "NP")) ||
  -- S[dcl]\NP => N || S[b]\NP => N || S[ng]\NP => N
  ((cat `elem` map T.pack ["S[dcl]\\NP", "S[ng]\\NP", "S[b]\\NP"]) && (childCat == T.pack "N")) ||
  -- S[dcl] => NP || S[adj] => NP || S[ng] => NP || S[ng]\NP => NP 
  ((cat `elem` map T.pack ["S[dcl]", "S[adj]", "S[ng]", "S[ng]\\NP", "S[to]\\NP"]) && (childCat == T.pack "NP")) ||
  -- S[dcl]\NP => S[dcl]
  ((cat == T.pack "S[dcl]\\NP") && (childCat == T.pack "S[dcl]")) ||
  -- S[to]\NP => S || (S[ng]\NP)/NP => S
  ((cat `elem` map T.pack ["S[dcl]\\NP", "S[to]\\NP", "S[pss]\\NP", "(S[ng]\\NP)/NP"]) && (childCat == T.pack "S")) ||

  -- | bellow rules are of the form Y => X/X or Y => X\X
  -- | S[X]/NP
  -- NP => NP\NP || NP => S/S || NP => S/(S/NP) || NP => (S\NP)/(S\NP) || NP => (S\NP)\(S\NP)
  ((cat == T.pack "NP") && (childCat `elem` map T.pack ["NP\\NP", "S/S", "S/(S/NP)", "(S\\NP)/(S\\NP)", "(S\\NP)\\(S\\NP)"])) ||
  -- S => S\S |　　NP\NP |　(S\NP)/(S\NP) |　(S\NP)\(S\NP)
  (case parseCategory cat of
    GCATEGORY (parsedCat, _) -> (parsedCat == T.pack "S") && (childCat `elem` map T.pack ["S\\S", "NP\\NP", "(S\\NP)/(S\\NP)", "(S\\NP)\\(S\\NP)"])
    _ -> False) ||
  -- S[dcl] => 
  ((cat == T.pack "S[dcl]") && (childCat `elem` map T.pack ["N", "S", "N/N", "N\\N", "NP/NP", "S/S", "(N/N)\\(N/N)", "(NP\\NP)\\(NP\\NP)", "(S/S)\\(S/S)", "((S\\NP)/(S\\NP))\\((S\\NP)/(S\\NP))", "((S\\NP)\\(S\\NP))\\((S\\NP)\\(S\\NP))"])) ||
  -- S[adj] => 
  ((cat == T.pack "S[adj]") && (childCat `elem` map T.pack ["NP/NP", "S/S"])) ||
  -- S[ng] => 
  ((cat == T.pack "S[ng]") && (childCat `elem` map T.pack ["NP\\NP", "S/S"])) ||
  -- S[b] => S/S
  ((cat == T.pack "S[b]") && (childCat == T.pack "S/S")) ||
  -- S[pss] => S/S
  ((cat == T.pack "S[pss]") && (childCat == T.pack "S/S")) ||
  --S[intj] => S/S
  ((cat == T.pack "S[intj]") && (childCat == T.pack "S/S")) ||

  -- | S[X]/NP
  -- S[dcl]/NP => 
  ((cat == T.pack "S[dcl]/NP") && (childCat `elem` map T.pack ["S", "NP\\NP", "S\\S", "(NP\\NP)\\(NP\\NP)", "((S\\NP)\\(S\\NP))\\((S\\NP)\\(S\\NP))"])) ||
  -- S[b]/NP => NP\NP
  ((cat == T.pack "S[b]/NP") && (childCat == T.pack "NP\\NP")) ||

  -- | S[X]\NP
  --S\NP => NP\NP 
  (case parseCategory cat of
    BSLASH(GCATEGORY (parsedCat1, _), GCATEGORY(parsedCat2, _)) -> 
      (parsedCat1 == T.pack "S") && (parsedCat2 == T.pack "NP") && (childCat `elem` map T.pack ["NP\\NP"])
    _ -> False) ||
  -- S[dcl]\NP => S | N/N | N\N | NP\NP | S/S | S\S | (NP\NP)\(NP\NP) | (S/S)\(S/S) | (S\NP)/(S\NP) | (S\NP)\(S\NP) | (S[adj]\NP)\(S[adj]\NP) | ((S\NP)\(S\NP))\((S\NP)\(S\NP))
  ((cat == T.pack "S[dcl]\\NP") && (childCat `elem` map T.pack ["N/N", "N\\N",         "NP\\NP",           "S/S", "S\\S",                                                                                   "(NP\\NP)\\(NP\\NP)", "(S/S)\\(S/S)",                   "(S\\NP)/(S\\NP)", "(S\\NP)\\(S\\NP)", "(S[adj]\\NP)\\(S[adj]\\NP)",                                                                            "((S\\NP)\\(S\\NP))\\((S\\NP)\\(S\\NP))"])) ||
  -- S[adj]\NP => S\S | N\N | PP\PP | S/S | S\S | (N/N)\(N/N) | (NP\NP)\(NP\NP) | (S/S)\(S/S) | (S\NP)/(S\NP) | (S\NP)\(S\NP) | (S[adj]\NP)\(S[adj]\NP) | ((N/N)\(N/N))\((N/N)\(N/N)) | ((S\NP)\(S\NP))\((S\NP)\(S\NP)) | ((S\NP)/(S\NP))\((S\NP)/(S\NP)) | (((S\NP)\(S\NP))\((S\NP)\(S\NP)))\(((S\NP)\(S\NP))\((S\NP)\(S\NP))) 
  ((cat == T.pack "S[adj]\\NP") && (childCat `elem` map T.pack [       "N\\N",                   "PP\\PP", "S/S", "S\\S",                                "(N/N)\\(N/N)",                                    "(NP\\NP)\\(NP\\NP)", "(S/S)\\(S/S)",                   "(S\\NP)/(S\\NP)", "(S\\NP)\\(S\\NP)", "(S[adj]\\NP)\\(S[adj]\\NP)", "((N/N)\\(N/N))\\((N/N)\\(N/N))",                                          "((S\\NP)\\(S\\NP))\\((S\\NP)\\(S\\NP))", "((S\\NP)/(S\\NP))\\((S\\NP)/(S\\NP))", "(((S\\NP)\\(S\\NP))\\((S\\NP)\\(S\\NP)))\\(((S\\NP)\\(S\\NP))\\((S\\NP)\\(S\\NP)))"])) ||
  -- S[pss]\\NP =>"N/N", "N\\N", "NP/NP", "S/S", "S\\S", "(N\\N)\\(N\\N)", "(NP\\NP)\\(NP\\NP)", "(S/S)\\(S/S)", "(S\\NP)/(S\\NP)", "(S\\NP)\\(S\\NP)", "((S\\NP)\\(S\\NP))\\((S\\NP)\\(S\\NP))"
  ((cat == T.pack "S[pss]\\NP") && (childCat `elem` map T.pack ["N/N", "N\\N", "NP/NP",                    "S/S", "S\\S",                                                                 "(N\\N)\\(N\\N)", "(NP\\NP)\\(NP\\NP)", "(S/S)\\(S/S)",                   "(S\\NP)/(S\\NP)", "(S\\NP)\\(S\\NP)",                                                                                                          "((S\\NP)\\(S\\NP))\\((S\\NP)\\(S\\NP))"])) ||
  -- S[b]\\NP => "N/N", "N\\N", "NP/NP", "S/S", "S\\S", "(N\\N)/(N\\N)", "(S\\NP)\\(S\\NP)", "((S\\NP)\\(S\\NP))/((S\\NP)\\(S\\NP))"
  ((cat == T.pack "S[b]\\NP") && (childCat `elem` map T.pack   ["N/N", "N\\N", "NP/NP",                    "S/S", "S\\S",                                                "(N\\N)/(N\\N)",                                                                                              "(S\\NP)\\(S\\NP)",                                                                 "((S\\NP)\\(S\\NP))/((S\\NP)\\(S\\NP))"])) ||
  -- S[to]\\NP => 
  ((cat == T.pack "S[to]\\NP") && (childCat `elem` map T.pack  [       "N\\N",                             "S/S", "S\\S", "S[wq]\\NP", "S[adj]\\S[adj]",                                                    "(NP\\NP)\\(NP\\NP)",                                   "(S\\NP)/(S\\NP)", "(S\\NP)\\(S\\NP)", "(S[adj]\\NP)\\(S[adj]\\NP)",                                                                            "((S\\NP)\\(S\\NP))\\((S\\NP)\\(S\\NP))", "(((S\\NP)\\(S\\NP))\\PP)\\(((S\\NP)\\(S\\NP))\\PP)"])) ||
  -- S[ng]\NP => 
  ((cat == T.pack "S[ng]\\NP") && (childCat `elem` map T.pack  ["N/N",         "NP/NP",         "PP\\PP",  "S/S", "S\\S",                                                                                   "(NP\\NP)\\(NP\\NP)",                 "(S\\S)\\(S\\S)", "(S\\NP)/(S\\NP)", "(S\\NP)\\(S\\NP)", "(S[adj]\\NP)\\(S[adj]\\NP)",                                                                            "((S\\NP)\\(S\\NP))\\((S\\NP)\\(S\\NP))"])) ||  
  -- S[pt]\\NP => (S\NP)\(S\NP)
  ((cat == T.pack "S[pt]\\NP") && (childCat == T.pack "(S\\NP)\\(S\\NP)")) ||

  -- | S[X]/S[X]
  -- S[dcl]/S[dcl] =>
  ((cat == T.pack "S[dcl]/S[dcl]") && (childCat `elem` map T.pack ["NP\\NP", "S/S", "S\\S", "(S\\NP)/(S\\NP)", "(S\\NP)\\(S\\NP)"])) ||

  -- | S[X]\S[X]
  -- S[dcl]\S[dcl] => 
  ((cat == T.pack "S[dcl]\\S[dcl]") && (childCat == T.pack "(S\\NP)\\(S\\NP)")) ||
  -- S[dcl]\S[dcl] => S\S
  ((cat == T.pack "S[dcl]\\S[dcl]") && (childCat == T.pack "S\\S")) ||
  -- S[dcl]\S[dcl] => (S\NP)/(S\NP)
  ((cat == T.pack "S[dcl]\\S[dcl]") && (childCat == T.pack "(S\\NP)/(S\\NP)")) ||
  -- S[dcl]\S[dcl] => S/S
  ((cat == T.pack "S[dcl]\\S[dcl]") && (childCat == T.pack "S/S")) ||

  -- | (S[X]\NP)/NP
  -- (S[to]\\NP)/NP => 
  ((cat == T.pack "(S[to]\\NP)/NP") && (childCat `elem` map T.pack ["N\\N", "NP\\NP", "(S\\NP)\\(S\\NP)", "((NP\\NP)\\(NP\\NP))\\((NP\\NP)\\(NP\\NP))"])) ||
  -- (S[adj]\\NP)/NP => NP\NP
  ((cat == T.pack "(S[adj]\\NP)/NP") && (childCat `elem` map T.pack ["(NP\\NP)\\(NP\\NP)"])) ||
    -- (S[dcl]\NP)/NP => NP\\NP
  ((cat == T.pack "(S[dcl]\\NP)/NP") && (childCat == T.pack "NP\\NP")) ||
  -- (S[ng]\NP)/NP => NP\\NP
  ((cat == T.pack "(S[ng]\\NP)/NP") && (childCat == T.pack "NP\\NP")) ||

  -- | (S[X]\NP)/PP
  -- (S[pss]\NP)/PP => S[pss]\NP
  ((cat == T.pack "(S[pss]\\NP)/PP") && (childCat == T.pack "S[pss]\\NP")) ||

  -- | ((S[X]\NP)/(S[Y]\NP))/NP
  -- ((S[pss]\NP)/(S[adj]\NP))/NP => (S[pss]\\NP)/(S[adj]\\NP)
  ((cat == T.pack "((S[pss]\\NP)/(S[adj]\\NP))/NP") && (childCat == T.pack "(S[pss]\\NP)/(S[adj]\\NP)"))
unaryTypeChangingRules _ _ = False


binaryTypeChangingRules ::
  (Tree, Tree) ->
  Tree ->
  Bool
binaryTypeChangingRules (Phrase (cat1, _), Phrase (cat2, _)) (Phrase (childCat, _)) =
  -- NP N\N => NP
  ((cat1 == T.pack "NP") && (cat2 == T.pack "N\\N") && (childCat == T.pack "NP")) ||

  -- | , X => Y
  -- , N => NP
  ((cat1 == T.pack ",") && (cat2 == T.pack "N") && (childCat == T.pack "NP")) ||
  -- , NP => 
  ((cat1 == T.pack ",") && (cat2 == T.pack "NP") && (childCat `elem` map T.pack ["S\\S", "(S\\NP)/(S\\NP)", "(S\\NP)\\(S\\NP)"])) ||
  -- ; NP => (S\NP)\\(S\NP)
  ((cat1 == T.pack ";") && (cat2 == T.pack "NP") && (childCat == T.pack "(S\\NP)\\(S\\NP)")) ||

  -- , S[em] => S[dcl][conj]
  ((cat1 == T.pack ",") && (cat2 == T.pack "S[em]") && (childCat == T.pack "S[dcl][conj]")) ||
  -- , S[ng] => NP\NP
  ((cat1 == T.pack ",") && (cat2 == T.pack "S[ng]") && (childCat == T.pack "NP\\NP")) ||
  -- , S[dcl] =>
  ((cat1 == T.pack ",") && (cat2 == T.pack "S[dcl]") && (childCat `elem` map T.pack ["NP\\NP", "S/S", "S\\S", "(S\\NP)\\(S\\NP)", "(S/S)\\(S/S)"])) ||

  -- , NP\\NP => NP[conj]
  ((cat1 == T.pack ",") && (cat2 == T.pack "NP\\NP") && (childCat == T.pack "NP[conj]")) ||

  -- , S[ng]\\NP => NP\NP
  ((cat1 == T.pack ",") && (cat2 == T.pack "S[ng]\\NP") && (childCat == T.pack "NP\\NP")) ||
  -- , S[to]\\NP => (S\NP)\(S\NP)
  ((cat1 == T.pack ",") && (cat2 == T.pack "S[to]\\NP") && (childCat == T.pack "(S\\NP)\\(S\\NP)")) ||
  -- , S[b]\\NP => (NP\NP)\(NP\NP)
  ((cat1 == T.pack ",") && (cat2 == T.pack "S[b]\\NP") && (childCat == T.pack "(NP\\NP)\\(NP\\NP)")) ||

  -- , S[dcl] =>
  ((cat1 == T.pack ",") && (cat2 == T.pack "S[dcl]/S[dcl]") && (childCat `elem` map T.pack ["NP\\NP", "S\\S", "(S\\NP)/(S\\NP)", "(S\\NP)\\(S\\NP)"])) ||

  -- , S[dcl]\S[dcl] => (S\NP)/(S\NP)
  ((cat1 == T.pack ",") && (cat2 == T.pack "S[dcl]\\S[dcl]") && (childCat == T.pack "(S\\NP)/(S\\NP)")) ||
  -- , S[dcl]\S[dcl] => S/S
  ((cat1 == T.pack ",") && (cat2 == T.pack "S[dcl]\\S[dcl]") && (childCat == T.pack "S/S")) ||

  -- | X , => Y
  -- NP , => S/S
  ((cat1 == T.pack "NP") && (cat2 == T.pack ",") && (childCat `elem` map T.pack ["S/S", "(S\\NP)/(S\\NP)"])) ||
  -- S[dcl] , => (S\NP)\(S\NP)
  ((cat1 == T.pack "S[dcl]") && (cat2 == T.pack ",") && (childCat `elem` map T.pack ["NP/NP", "NP\\NP", "S/S", "S\\S", "(N/N)\\(N/N)","(NP\\NP)\\(NP\\NP)", "(S\\NP)/(S\\NP)", "(S\\NP)\\(S\\NP)", "(S[adj]\\NP)\\(S[adj]\\NP)"])) ||  
  -- S[b] , => (S\NP)/(S\NP)
  ((cat1 == T.pack "S[b]") && (cat2 == T.pack ",") && (childCat == T.pack "(S\\NP)/(S\\NP)")) ||
  -- S[ng] , => (S\NP)/(S\NP)
  ((cat1 == T.pack "S[ng]") && (cat2 == T.pack ",") && (childCat == T.pack "(S\\NP)/(S\\NP)")) ||
  -- S[to] , => NP\NP
  ((cat1 == T.pack "S[to]\\NP") && (cat2 == T.pack ",") && (childCat == T.pack "NP\\NP")) ||

  -- S[dcl]/NP , => NP\NP
  ((cat1 == T.pack "S[dcl]/NP") && (cat2 == T.pack ",") && (childCat == T.pack "NP\\NP")) ||
  -- S[to]\NP , => (S\NP)/(S\NP)
  ((cat1 == T.pack "S[to]\\NP") && (cat2 == T.pack ",") && (childCat `elem` map T.pack ["S\\S", "(S\\NP)/(S\\NP)"])) ||
  -- S[b]\NP , => (S\NP)/(S\NP)
  ((cat1 == T.pack "S[b]\\NP") && (cat2 == T.pack ",") && (childCat `elem` map T.pack ["S/S", "(S\\NP)/(S\\NP)"])) ||
  -- S[pss]|NP , => (S\NP)\(S\NP)
  ((cat1 == T.pack "S[pss]\\NP") && (cat2 == T.pack ",") && (childCat == T.pack "(S\\NP)\\(S\\NP)")) ||

  -- S[dcl]/S[dcl] , => 
  ((cat1 == T.pack "S[dcl]/S[dcl]") && (cat2 == T.pack ",") && (childCat `elem` map T.pack ["NP\\NP", "S\\S", "(NP\\NP)\\(NP\\NP)", "(S\\NP)/(S\\NP)", "(S\\NP)\\(S\\NP)", "(S[adj]\\NP)\\(S[adj]\\NP)", "((S\\NP)\\(S\\NP))\\((S\\NP)\\(S\\NP))"])) ||
  -- S[b]/S[dcl] , => (S\NP)/(S\NP)
  ((cat1 == T.pack "S[b]/S[dcl]") && (cat2 == T.pack ",") && (childCat == T.pack "(S\\NP)/(S\\NP)")) ||

  -- S[dcl]\S[dcl] , => (S\NP)/(S\NP)
  ((cat1 == T.pack "S[dcl]\\S[dcl]") && (cat2 == T.pack ",") && (childCat `elem` map T.pack ["NP\\NP", "S/S", "(S\\NP)/(S\\NP)", "(S\\NP)\\(S\\NP)", "(S[adj]\\NP)\\(S[adj]\\NP)"])) ||

  -- X RRB => X
  ((cat2 == T.pack "RRB") && (childCat == cat1)) ||
  -- RRB X => X
  ((cat1 == T.pack "RRB") && (childCat == cat2)) ||
  -- LRB X => X
  ((cat1 == T.pack "LRB") && (childCat == cat2)) ||
  -- LRB X => X[conj]
  ((cat1 == T.pack "LRB") && (childCat == T.concat [cat2, T.pack "[conj]"])) ||
  -- LRB X[conj] => X
  ((cat1 == T.pack "LRB") && (cat2 == T.concat [childCat, T.pack "[conj]"])) ||
  -- X LRB => X
  ((cat2 == T.pack "LRB") && (childCat == cat1))
binaryTypeChangingRules _ _ = False

