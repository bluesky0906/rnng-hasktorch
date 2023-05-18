{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}

module Model.Device where
import Torch
import Data.Binary
import GHC.Generics

deriving instance Generic DeviceType
deriving instance Binary DeviceType
deriving instance Generic Device
deriving instance Binary Device

