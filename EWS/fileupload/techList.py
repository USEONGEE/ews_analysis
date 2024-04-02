
from enum import Enum
class AnalysisTech_forServer(Enum):
  LinearRegression = "lr"
  RidgeRegression = "ridge"
  ExtremeGradientBoosting = 'xgboost'
  LightGBM = 'lightgbm'
  
class AnalysisTech_forClient(Enum):
  LinearRegression = "LinearRegression"
  RidgeRegression = "RidgeRegression"
  ExtremeGradientBoosting = 'ExtremeGradientBoosting'
  LightGBM = 'LightGBM'
