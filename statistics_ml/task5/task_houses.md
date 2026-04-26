#########################

Задание 5. Данные house_prices.csv
Прогноз непрерывной переменной SalePrice
Целевая переменная: SalePrice
Используемые признаки: GrLivArea, GarageArea
Примечание: для устойчивого непараметрического сглаживания на реальных данных используется сокращённый набор непрерывных признаков.
Размер обучающей выборки: 1168
Размер тестовой выборки: 292

Ядерное сглаживание: все конфигурации
gaussian, local_constant: bandwidth = [0.3000, 0.3000], CV MSE = 1785905314.9270, test MSE = 1995628372.2956, test R^2 = 0.7398
gaussian, local_constant: bandwidth = [0.6000, 0.6000], CV MSE = 2022777361.4311, test MSE = 2425284367.5899, test R^2 = 0.6838
gaussian, local_constant: bandwidth = [0.9000, 0.9000], CV MSE = 2569028446.4566, test MSE = 3183046810.3890, test R^2 = 0.5850
gaussian, local_constant: bandwidth = [1.2000, 1.2000], CV MSE = 3239168312.6251, test MSE = 3978586514.2480, test R^2 = 0.4813
gaussian, local_constant: bandwidth = [1.8000, 1.8000], CV MSE = 4379317965.4108, test MSE = 5292965994.8492, test R^2 = 0.3099
gaussian, local_linear: bandwidth = [0.3000, 0.3000], CV MSE = 1576601176.9277, test MSE = 2114609145.7363, test R^2 = 0.7243
gaussian, local_linear: bandwidth = [0.6000, 0.6000], CV MSE = 1704625927.9778, test MSE = 2019823659.1287, test R^2 = 0.7367
gaussian, local_linear: bandwidth = [0.9000, 0.9000], CV MSE = 1760898423.8588, test MSE = 2161116454.7279, test R^2 = 0.7182
gaussian, local_linear: bandwidth = [1.2000, 1.2000], CV MSE = 1799076641.0015, test MSE = 2264881922.3569, test R^2 = 0.7047
gaussian, local_linear: bandwidth = [1.8000, 1.8000], CV MSE = 1854115708.6810, test MSE = 2420321382.3206, test R^2 = 0.6845
tricube, local_constant: bandwidth = [0.3000, 0.3000], CV MSE = 6906881603.1512, test MSE = 8182821746.0168, test R^2 = -0.0668
tricube, local_constant: bandwidth = [0.6000, 0.6000], CV MSE = 7193521802.2971, test MSE = 8521868968.9123, test R^2 = -0.1110
tricube, local_constant: bandwidth = [0.9000, 0.9000], CV MSE = 7362786843.2613, test MSE = 8631680020.0221, test R^2 = -0.1253
tricube, local_constant: bandwidth = [1.2000, 1.2000], CV MSE = 7286982261.8099, test MSE = 8532576692.9486, test R^2 = -0.1124
tricube, local_constant: bandwidth = [1.8000, 1.8000], CV MSE = 6589945119.5231, test MSE = 7783513400.8657, test R^2 = -0.0148
tricube, local_linear: bandwidth = [0.3000, 0.3000], CV MSE = 2096197789.3238, test MSE = 2739398754.5844, test R^2 = 0.6429
tricube, local_linear: bandwidth = [0.6000, 0.6000], CV MSE = 2125211594.3550, test MSE = 2815533531.7599, test R^2 = 0.6329
tricube, local_linear: bandwidth = [0.9000, 0.9000], CV MSE = 2142684468.7953, test MSE = 2808586742.3161, test R^2 = 0.6338
tricube, local_linear: bandwidth = [1.2000, 1.2000], CV MSE = 2111889140.6059, test MSE = 2757684168.0812, test R^2 = 0.6405
tricube, local_linear: bandwidth = [1.8000, 1.8000], CV MSE = 2087807677.1526, test MSE = 2674739531.4459, test R^2 = 0.6513

Лучшая модель ядерного сглаживания
Ядро: gaussian
Локальная аппроксимация: local_linear
Ширины окна: [0.3000, 0.3000]
CV MSE: 1576601176.9277
Train MSE: 1636863333.4535
Test MSE: 2114609145.7363
Train R^2: 0.7256
Test R^2: 0.7243

Гауссовский процесс: все конфигурации
linear, kernel = DotProduct(sigma_0=0.5) + WhiteKernel(noise_level=0.01): CV MSE = 2362518793.1651, test MSE = 2625171654.9962, test R^2 = 0.6577
linear, kernel = DotProduct(sigma_0=0.5) + WhiteKernel(noise_level=0.05): CV MSE = 2362722826.9263, test MSE = 2625206540.1384, test R^2 = 0.6577
linear, kernel = DotProduct(sigma_0=1) + WhiteKernel(noise_level=0.01): CV MSE = 2362518793.1639, test MSE = 2625171655.0001, test R^2 = 0.6577
linear, kernel = DotProduct(sigma_0=1) + WhiteKernel(noise_level=0.05): CV MSE = 2362722826.9262, test MSE = 2625206540.1395, test R^2 = 0.6577
gaussian, kernel = RBF(length_scale=0.4) + WhiteKernel(noise_level=0.01): CV MSE = 4130805344.7121, test MSE = 2214701167.7320, test R^2 = 0.7113
gaussian, kernel = RBF(length_scale=0.4) + WhiteKernel(noise_level=0.05): CV MSE = 3263087787.4338, test MSE = 2202462076.1147, test R^2 = 0.7129
gaussian, kernel = RBF(length_scale=0.8) + WhiteKernel(noise_level=0.01): CV MSE = 2770794452.4192, test MSE = 2206673060.6901, test R^2 = 0.7123
gaussian, kernel = RBF(length_scale=0.8) + WhiteKernel(noise_level=0.05): CV MSE = 2365144452.4002, test MSE = 2039552373.2740, test R^2 = 0.7341
gaussian, kernel = RBF(length_scale=1.2) + WhiteKernel(noise_level=0.01): CV MSE = 2033634276.8052, test MSE = 2016713680.6030, test R^2 = 0.7371
gaussian, kernel = RBF(length_scale=1.2) + WhiteKernel(noise_level=0.05): CV MSE = 1922769165.8154, test MSE = 1961564914.9074, test R^2 = 0.7443
exponential, kernel = Matern(length_scale=0.4, nu=0.5) + WhiteKernel(noise_level=0.01): CV MSE = 3351097436.4911, test MSE = 2707971536.8548, test R^2 = 0.6470
exponential, kernel = Matern(length_scale=0.4, nu=0.5) + WhiteKernel(noise_level=0.05): CV MSE = 3260352784.3151, test MSE = 2598062941.9043, test R^2 = 0.6613
exponential, kernel = Matern(length_scale=0.8, nu=0.5) + WhiteKernel(noise_level=0.01): CV MSE = 2944254388.8203, test MSE = 2399833172.6819, test R^2 = 0.6871
exponential, kernel = Matern(length_scale=0.8, nu=0.5) + WhiteKernel(noise_level=0.05): CV MSE = 2786808752.4366, test MSE = 2255784956.0934, test R^2 = 0.7059
exponential, kernel = Matern(length_scale=1.2, nu=0.5) + WhiteKernel(noise_level=0.01): CV MSE = 2809450501.3249, test MSE = 2310136276.2134, test R^2 = 0.6988
exponential, kernel = Matern(length_scale=1.2, nu=0.5) + WhiteKernel(noise_level=0.05): CV MSE = 2615310613.3597, test MSE = 2155725466.2858, test R^2 = 0.7190

Лучшая модель гауссовского процесса
Ядро: gaussian
Параметры: RBF(length_scale=1.2) + WhiteKernel(noise_level=0.05)
CV MSE: 1922769165.8154
Train MSE: 1767261169.1201
Test MSE: 1961564914.9074
Train R^2: 0.7037
Test R^2: 0.7443
Среднее стандартное отклонение на тесте: 17789.8841

Сравнение лучших моделей
Лучшая модель по CV: kernel_smoothing (CV MSE = 1576601176.9277)
График сохранён в: C:\Users\jds10\Documents\VSCodeProjects\vuz\analyze\statistics_ml\task5\house_prices_methods_plot.png