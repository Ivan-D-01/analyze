#########################

Задание 5. Пункт 1
Оценка функции методом ядерного сглаживания
Область определения: (-2.0000, 4.0000)
Сигма: 0.2000
Размер выборки: 50

Все конфигурации
gaussian, local_constant: bandwidth = 0.2042, LOOCV MSE = 0.0416, train MSE = 0.0258
gaussian, local_linear: bandwidth = 0.1849, LOOCV MSE = 0.0389, train MSE = 0.0227
tricube, local_constant: bandwidth = 0.8880, LOOCV MSE = 0.3306, train MSE = 0.3156
tricube, local_linear: bandwidth = 0.8374, LOOCV MSE = 0.1611, train MSE = 0.1472

Лучшая конфигурация
Ядро: gaussian
Локальная аппроксимация: local_linear
Ширина окна: 0.1849
Средняя квадратичная ошибка Leave-One-Out Cross-Validation: 0.0389
Средняя квадратичная ошибка на обучении: 0.0227
График сохранён в: C:\Users\jds10\Documents\VSCodeProjects\vuz\analyze\statistics_ml\task5\kernel_smoothing_plot.png

#########################

Задание 5. Пункт 2
Оценка функции с помощью регрессии гауссовским процессом

Топ-3 конфигурации для каждого ядра

linear
top-1: kernel = DotProduct(sigma_0=0.5) + WhiteKernel(noise_level=0.09), CV MSE = 0.1492, train MSE = 0.1311, mean train std = 0.1564
top-2: kernel = DotProduct(sigma_0=0.5) + WhiteKernel(noise_level=0.04), CV MSE = 0.1493, train MSE = 0.1311, mean train std = 0.1043
top-3: kernel = DotProduct(sigma_0=1) + WhiteKernel(noise_level=0.09), CV MSE = 0.1493, train MSE = 0.1311, mean train std = 0.1564

gaussian
top-1: kernel = RBF(length_scale=0.4) + WhiteKernel(noise_level=0.09), CV MSE = 0.0380, train MSE = 0.0242, mean train std = 0.1734
top-2: kernel = RBF(length_scale=0.4) + WhiteKernel(noise_level=0.04), CV MSE = 0.0383, train MSE = 0.0233, mean train std = 0.1168
top-3: kernel = RBF(length_scale=0.8) + WhiteKernel(noise_level=0.01), CV MSE = 0.0398, train MSE = 0.0273, mean train std = 0.0561

exponential
top-1: kernel = Matern(length_scale=1.2, nu=0.5) + WhiteKernel(noise_level=0.09), CV MSE = 0.0384, train MSE = 0.0134, mean train std = 0.1886
top-2: kernel = Matern(length_scale=0.8, nu=0.5) + WhiteKernel(noise_level=0.09), CV MSE = 0.0393, train MSE = 0.0116, mean train std = 0.1924
top-3: kernel = Matern(length_scale=1.2, nu=0.5) + WhiteKernel(noise_level=0.04), CV MSE = 0.0408, train MSE = 0.0098, mean train std = 0.1308

Лучшая конфигурация
Ядро: gaussian
Параметры: RBF(length_scale=0.4) + WhiteKernel(noise_level=0.09)
Средняя квадратичная ошибка CV: 0.0380
Средняя квадратичная ошибка на обучении: 0.0242
Среднее стандартное отклонение на обучении: 0.1734
График сохранён в: C:\Users\jds10\Documents\VSCodeProjects\vuz\analyze\statistics_ml\task5\gaussian_process_plot.png