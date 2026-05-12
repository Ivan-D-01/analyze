Target distribution: MSZoning
RL: count=1151, share=0.7884
RM: count=218, share=0.1493
FV: count=65, share=0.0445
RH: count=16, share=0.0110
C (all): count=10, share=0.0068


#########################

1. Binary logistic regression

Target split distribution
RL: full=1151 (0.7884), train=921 (0.7885), test=230 (0.7877)
RM: full=218 (0.1493), train=174 (0.1490), test=44 (0.1507)
FV: full=65 (0.0445), train=52 (0.0445), test=13 (0.0445)
RH: full=16 (0.0110), train=13 (0.0111), test=3 (0.0103)
C (all): full=10 (0.0068), train=8 (0.0068), test=2 (0.0068)
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
Test log-likelihood: -120.2390
Top 3 coefficients: {'SalePrice': -292.4086, 'const': -265.0786, 'Neighborhood_Mitchel': -222.6686}
Bottom 3 coefficients: {'Neighborhood_BrkSide': 3.8717, 'LotShape_Reg': -5.0517, 'GarageArea': 9.2384}
C:\Users\jds10\Documents\VSCodeProjects\vuz\analyze\.venv\Lib\site-packages\statsmodels\genmod\families\links.py:198: RuntimeWarning: overflow encountered in exp
  t = np.exp(-z)

Level FV
Accuracy: 0.9829
Precision: 0.7500
Recall: 0.9231
F1: 0.8276
Test log-likelihood: -138.1552
Top 3 coefficients: {'Utilities_NoSeWa': 3621059856584900.5, 'const': -3568016828568647.0, 'Neighborhood_CollgCr': -2919383761661167.0}
Bottom 3 coefficients: {'LotConfig_FR2': -30992475302781.496, 'TotalBsmtSF': -38872156523360.32, 'GarageArea': -62621153911633.625}

Level RH
Accuracy: 0.9897
Precision: 0.0000
Recall: 0.0000
F1: 0.0000
Test log-likelihood: -40.8966
Top 3 coefficients: {'Utilities_NoSeWa': 68.1108, 'LotConfig_CulDSac': -46.0104, 'const': -43.7546}
Bottom 3 coefficients: {'TotalBsmtSF': -0.0856, 'Alley_Pave': -0.1712, '1stFlrSF': 0.1912}

Level RL
Accuracy: 0.9212
Precision: 0.9295
Recall: 0.9739
F1: 0.9512
Test log-likelihood: -118.1500
Top 3 coefficients: {'Neighborhood_IDOTRR': -99.7082, 'LotConfig_FR3': -54.5985, 'Neighborhood_MeadowV': -54.2288}
Bottom 3 coefficients: {'YearBuilt': 0.0006, 'YearRemodAdd': -0.019, 'LotConfig_Inside': 0.022}

Level RM
Accuracy: 0.9418
Precision: 0.8649
Recall: 0.7273
F1: 0.7901
Test log-likelihood: -84.5143
Top 3 coefficients: {'Neighborhood_MeadowV': 56.1203, 'Neighborhood_BrDale': 55.6438, 'Neighborhood_Blueste': 54.8176}
Bottom 3 coefficients: {'YearRemodAdd': 0.0881, 'LotFrontage': -0.1908, 'YearBuilt': 0.2482}


#########################

2. Binary probit regression

Target split distribution
RL: full=1151 (0.7884), train=921 (0.7885), test=230 (0.7877)
RM: full=218 (0.1493), train=174 (0.1490), test=44 (0.1507)
FV: full=65 (0.0445), train=52 (0.0445), test=13 (0.0445)
RH: full=16 (0.0110), train=13 (0.0111), test=3 (0.0103)
C (all): full=10 (0.0068), train=8 (0.0068), test=2 (0.0068)
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
Test log-likelihood: -123.9926
Top 3 coefficients: {'SalePrice': -79.3928, 'const': -69.8662, 'Utilities_NoSeWa': 56.9718}
Bottom 3 coefficients: {'LotConfig_CulDSac': -1.0339, 'Neighborhood_SWISU': 1.5368, '2ndFlrSF': 1.6017}

Level FV
Accuracy: 0.9863
Precision: 0.8462
Recall: 0.8462
F1: 0.8462
Test log-likelihood: -110.5241
Top 3 coefficients: {'Utilities_NoSeWa': 605118673140112.8, 'const': -529467105637437.6, 'LandContour_Low': 363168481427623.25}
Bottom 3 coefficients: {'Street_Pave': -447078357583.914, 'LotArea': -1185151111884.6328, 'LotFrontage': 1719507358562.0742}

Level RH
Accuracy: 0.9897
Precision: 0.0000
Recall: 0.0000
F1: 0.0000
Test log-likelihood: -44.1765
Top 3 coefficients: {'Utilities_NoSeWa': 21.0544, 'LotConfig_CulDSac': -12.9623, 'LandSlope_Mod': -10.5674}
Bottom 3 coefficients: {'TotalBsmtSF': -0.0405, '1stFlrSF': -0.0725, 'Alley_Pave': 0.0751}

Level RL
Accuracy: 0.9212
Precision: 0.9295
Recall: 0.9739
F1: 0.9512
Test log-likelihood: -103.2857
Top 3 coefficients: {'Neighborhood_IDOTRR': -26.9709, 'LotConfig_FR3': -15.7496, 'Neighborhood_MeadowV': -13.865}
Bottom 3 coefficients: {'YearBuilt': 0.0243, 'LandContour_Lvl': -0.0281, 'YearRemodAdd': -0.0283}

Level RM
Accuracy: 0.9418
Precision: 0.8649
Recall: 0.7273
F1: 0.7901
Test log-likelihood: -79.4716
Top 3 coefficients: {'LotConfig_FR3': 14.8918, 'Neighborhood_MeadowV': 14.8874, 'Neighborhood_BrDale': 14.5284}
Bottom 3 coefficients: {'YearRemodAdd': 0.0631, 'YearBuilt': 0.0972, 'LotFrontage': -0.1168}

#########################

3. Log-likelihood summary
model       logit    probit  winner
level                              
C (all) -120.2390 -123.9926   logit
FV      -138.1552 -110.5241  probit
RH       -40.8966  -44.1765   logit
RL      -118.1500 -103.2857  probit
RM       -84.5143  -79.4716  probit
C:\Users\jds10\Documents\VSCodeProjects\vuz\analyze\.venv\Lib\site-packages\sklearn\linear_model\_logistic.py:1247: FutureWarning: 'multi_class' was deprecated in version 1.5 and will be removed in 1.7. From then on, it will always use 'multinomial'. Leave it to its default value to avoid this warning.
  warnings.warn(

#########################

4. Multinomial logistic regression
Accuracy: 0.9075
Macro F1: 0.5225
C (all): top 3 coefficients = {'SalePrice': -33.566, 'Neighborhood_IDOTRR': 20.1458, 'const': -13.1206}
FV: top 3 coefficients = {'Neighborhood_Somerst': 31.634, 'SalePrice': 10.2892, 'YearRemodAdd': -9.3976}
RH: top 3 coefficients = {'LandSlope_Mod': -19.3304, 'Neighborhood_SawyerW': 17.4157, 'Neighborhood_NAmes': 14.0327}
RL: top 3 coefficients = {'Neighborhood_IDOTRR': -27.4029, 'const': 20.2962, 'LotConfig_FR3': -19.9828}
RM: top 3 coefficients = {'LotConfig_FR3': 19.1644, 'Neighborhood_Somerst': -17.7815, 'Neighborhood_MeadowV': 16.879}

#########################

5. Gamma vs Gaussian regression
Gamma test log-likelihood: -3386.4970
Gaussian test log-likelihood: -3496.8648
Gamma regression is better than Gaussian on the test set by log-likelihood.