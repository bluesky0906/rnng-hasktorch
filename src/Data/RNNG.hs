{-# LANGUAGE GADTs #-}

module Data.RNNG where
import Data.RNNGSentence

import Torch

import qualified Data.Text as T          --text
import qualified Data.Text.IO as T       --text



data ParsingMode = Point | All deriving (Show)


data Mode where
  Mode ::
    {
      device :: Device,
      dropoutProb :: Maybe Double,
      parsingMode :: ParsingMode,
      posMode :: Bool
    } ->
    Mode

data IndexData = IndexData {
    wordIndexFor :: T.Text -> Int,
    indexWordFor :: Int -> T.Text,
    actionIndexFor :: Action -> Int,
    indexActionFor :: Int -> Action,
    ntIndexFor :: T.Text -> Int,
    indexNTFor :: Int -> T.Text
  }