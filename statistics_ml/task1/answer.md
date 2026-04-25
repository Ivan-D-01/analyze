Эксперимент 1. Базовая модель

Аналитическое решение
Матрица невырождена
R2 на обучающей выборке: 0.8266
R2 на тестовой выборке: 0.8351
Коэффициенты (первые 10):
LotFrontage: 9.58
LotArea: 0.51
OverallQual: 14754.47
OverallCond: 5663.48
YearBuilt: 364.50
YearRemodAdd: 142.01
1stFlrSF: 8.99
2ndFlrSF: -1.24
GrLivArea: 45.48
TotalBsmtSF: 13.65

МНК через scikit-learn
R2 на обучающей выборке: 0.8266
R2 на тестовой выборке: 0.8351
Коэффициенты (первые 10):
LotFrontage: 9.58
LotArea: 0.51
OverallQual: 14754.47
OverallCond: 5663.48
YearBuilt: 364.50
YearRemodAdd: 142.01
1stFlrSF: 8.99
2ndFlrSF: -1.24
GrLivArea: 45.48
TotalBsmtSF: 13.65
interaction columns: 24

Эксперимент 2. Полные взаимодействия OverallQual x Neighborhood

Аналитическое решение
Матрица вырождена: ранг 74 < 77
R2 на обучающей выборке: 0.8746
R2 на тестовой выборке: 0.8806
Коэффициенты (первые 10):
LotFrontage: -0.76
LotArea: 0.67
OverallQual: 39325.02
OverallCond: 8113.06
YearBuilt: 520.50
YearRemodAdd: 128.77
1stFlrSF: -8.03
2ndFlrSF: -8.69
GrLivArea: 56.99
TotalBsmtSF: 18.44

МНК через scikit-learn
R2 на обучающей выборке: 0.8746
R2 на тестовой выборке: 0.8806
Коэффициенты (первые 10):
LotFrontage: -0.76
LotArea: 0.67
OverallQual: 39325.02
OverallCond: 8113.06
YearBuilt: 520.50
YearRemodAdd: 128.77
1stFlrSF: -8.03
2ndFlrSF: -8.69
GrLivArea: 56.99
TotalBsmtSF: 18.44
interaction columns: 5

Эксперимент 3. Частичные взаимодействия OverallQual x Neighborhood

Аналитическое решение
Матрица невырождена
R2 на обучающей выборке: 0.8502
R2 на тестовой выборке: 0.8482
Коэффициенты (первые 10):
LotFrontage: 36.80
LotArea: 0.73
OverallQual: 21569.96
OverallCond: 7339.16
YearBuilt: 522.99
YearRemodAdd: 130.12
1stFlrSF: -0.32
2ndFlrSF: -4.62
GrLivArea: 53.07
TotalBsmtSF: 19.92

МНК через scikit-learn
R2 на обучающей выборке: 0.8502
R2 на тестовой выборке: 0.8482
Коэффициенты (первые 10):
LotFrontage: 36.80
LotArea: 0.73
OverallQual: 21569.96
OverallCond: 7339.16
YearBuilt: 522.99
YearRemodAdd: 130.12
1stFlrSF: -0.32
2ndFlrSF: -4.62
GrLivArea: 53.07
TotalBsmtSF: 19.92