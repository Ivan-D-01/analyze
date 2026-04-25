Target distribution: MSZoning
RL: count=1151, share=0.7884
RM: count=218, share=0.1493
FV: count=65, share=0.0445
RH: count=16, share=0.0110
C (all): count=10, share=0.0068

1. Binary logistic regression
C:\Users\jds10\Documents\VSCodeProjects\vuz\analyze\.venv\Lib\site-packages\statsmodels\genmod\families\links.py:198: RuntimeWarning: overflow encountered in exp
  t = np.exp(-z)
C:\Users\jds10\Documents\VSCodeProjects\vuz\analyze\.venv\Lib\site-packages\statsmodels\genmod\generalized_linear_model.py:1342: PerfectSeparationWarning: Perfect separation or prediction detected, parameter may not be identified
  warnings.warn(msg, category=PerfectSeparationWarning)
C:\Users\jds10\Documents\VSCodeProjects\vuz\analyze\.venv\Lib\site-packages\statsmodels\genmod\generalized_linear_model.py:1342: PerfectSeparationWarning: Perfect separation or prediction detected, parameter may not be identified
  warnings.warn(msg, category=PerfectSeparationWarning)
C:\Users\jds10\Documents\VSCodeProjects\vuz\analyze\.venv\Lib\site-packages\statsmodels\genmod\generalized_linear_model.py:1342: PerfectSeparationWarning: Perfect separation or prediction detected, parameter may not be identified
  warnings.warn(msg, category=PerfectSeparationWarning)
C:\Users\jds10\Documents\VSCodeProjects\vuz\analyze\.venv\Lib\site-packages\statsmodels\genmod\generalized_linear_model.py:1342: PerfectSeparationWarning: Perfect separation or prediction detected, parameter may not be identified
  warnings.warn(msg, category=PerfectSeparationWarning)

Level C (all)
Accuracy: 0.9760
Precision: 0.2222
Recall: 1.0000
F1: 0.3636
Top 3 coefficients: {'SalePrice': -292.4086, 'const': -265.0786, 'Neighborhood_Mitchel': -222.6686}
Bottom 3 coefficients: {'Neighborhood_BrkSide': 3.8717, 'LotShape_Reg': -5.0517, 'GarageArea': 9.2384}
C:\Users\jds10\Documents\VSCodeProjects\vuz\analyze\.venv\Lib\site-packages\statsmodels\genmod\families\links.py:198: RuntimeWarning: overflow encountered in exp
  t = np.exp(-z)

Level FV
Accuracy: 0.9829
Precision: 0.7500
Recall: 0.9231
F1: 0.8276
Top 3 coefficients: {'Utilities_NoSeWa': 3621059856584900.5, 'const': -3568016828568647.0, 'Neighborhood_CollgCr': -2919383761661167.0}
Bottom 3 coefficients: {'LotConfig_FR2': -30992475302781.496, 'TotalBsmtSF': -38872156523360.32, 'GarageArea': -62621153911633.625}

Level RH
Accuracy: 0.9897
Precision: 0.0000
Recall: 0.0000
F1: 0.0000
Top 3 coefficients: {'Utilities_NoSeWa': 68.1108, 'LotConfig_CulDSac': -46.0104, 'const': -43.7546}
Bottom 3 coefficients: {'TotalBsmtSF': -0.0856, 'Alley_Pave': -0.1712, '1stFlrSF': 0.1912}

Level RL
Accuracy: 0.9212
Precision: 0.9295
Recall: 0.9739
F1: 0.9512
Top 3 coefficients: {'Neighborhood_IDOTRR': -99.7082, 'LotConfig_FR3': -54.5985, 'Neighborhood_MeadowV': -54.2288}
Bottom 3 coefficients: {'YearBuilt': 0.0006, 'YearRemodAdd': -0.019, 'LotConfig_Inside': 0.022}

Level RM
Accuracy: 0.9418
Precision: 0.8649
Recall: 0.7273
F1: 0.7901
Top 3 coefficients: {'Neighborhood_MeadowV': 56.1203, 'Neighborhood_BrDale': 55.6438, 'Neighborhood_Blueste': 54.8176}
Bottom 3 coefficients: {'YearRemodAdd': 0.0881, 'LotFrontage': -0.1908, 'YearBuilt': 0.2482}

2. Binary probit regression
C:\Users\jds10\Documents\VSCodeProjects\vuz\analyze\.venv\Lib\site-packages\statsmodels\genmod\generalized_linear_model.py:1342: PerfectSeparationWarning: Perfect separation or prediction detected, parameter may not be identified
  warnings.warn(msg, category=PerfectSeparationWarning)
C:\Users\jds10\Documents\VSCodeProjects\vuz\analyze\.venv\Lib\site-packages\statsmodels\genmod\generalized_linear_model.py:1342: PerfectSeparationWarning: Perfect separation or prediction detected, parameter may not be identified
  warnings.warn(msg, category=PerfectSeparationWarning)
C:\Users\jds10\Documents\VSCodeProjects\vuz\analyze\.venv\Lib\site-packages\statsmodels\genmod\generalized_linear_model.py:1342: PerfectSeparationWarning: Perfect separation or prediction detected, parameter may not be identified
  warnings.warn(msg, category=PerfectSeparationWarning)
C:\Users\jds10\Documents\VSCodeProjects\vuz\analyze\.venv\Lib\site-packages\statsmodels\genmod\generalized_linear_model.py:1342: PerfectSeparationWarning: Perfect separation or prediction detected, parameter may not be identified
  warnings.warn(msg, category=PerfectSeparationWarning)

Level C (all)
Accuracy: 0.9795
Precision: 0.2500
Recall: 1.0000
F1: 0.4000
Top 3 coefficients: {'SalePrice': -79.3928, 'const': -69.8662, 'Utilities_NoSeWa': 56.9718}
Bottom 3 coefficients: {'LotConfig_CulDSac': -1.0339, 'Neighborhood_SWISU': 1.5368, '2ndFlrSF': 1.6017}

Level FV
Accuracy: 0.9863
Precision: 0.8462
Recall: 0.8462
F1: 0.8462
Top 3 coefficients: {'Utilities_NoSeWa': 605118673140112.8, 'const': -529467105637437.6, 'LandContour_Low': 363168481427623.25}
Bottom 3 coefficients: {'Street_Pave': -447078357583.914, 'LotArea': -1185151111884.6328, 'LotFrontage': 1719507358562.0742}

Level RH
Accuracy: 0.9897
Precision: 0.0000
Recall: 0.0000
F1: 0.0000
Top 3 coefficients: {'Utilities_NoSeWa': 21.0544, 'LotConfig_CulDSac': -12.9623, 'LandSlope_Mod': -10.5674}
Bottom 3 coefficients: {'TotalBsmtSF': -0.0405, '1stFlrSF': -0.0725, 'Alley_Pave': 0.0751}

Level RL
Accuracy: 0.9212
Precision: 0.9295
Recall: 0.9739
F1: 0.9512
Top 3 coefficients: {'Neighborhood_IDOTRR': -26.9709, 'LotConfig_FR3': -15.7496, 'Neighborhood_MeadowV': -13.865}
Bottom 3 coefficients: {'YearBuilt': 0.0243, 'LandContour_Lvl': -0.0281, 'YearRemodAdd': -0.0283}

Level RM
Accuracy: 0.9418
Precision: 0.8649
Recall: 0.7273
F1: 0.7901
Top 3 coefficients: {'LotConfig_FR3': 14.8918, 'Neighborhood_MeadowV': 14.8874, 'Neighborhood_BrDale': 14.5284}
Bottom 3 coefficients: {'YearRemodAdd': 0.0631, 'YearBuilt': 0.0972, 'LotFrontage': -0.1168}