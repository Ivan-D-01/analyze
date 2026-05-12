from sklearn.datasets import load_iris


iris = load_iris(as_frame=True)
iris_features = iris.data


import numpy as np


class EMAlgorithm:
    def __init__(self, n_components, max_iter=100, tol=1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

        # параметры модели
        self.params = None

        # распределение скрытых переменных
        self.responsibilities = None

    def fit(self, X):
        """
        Главный цикл EM-алгоритма.
        """
        self.params = self.initialize_parameters(X)

        prev_score = None

        for step in range(self.max_iter):
            # E-step:
            # оцениваем скрытые переменные при текущих параметрах
            self.responsibilities = self.e_step(X, self.params)

            # M-step:
            # обновляем параметры по оценённым скрытым переменным
            self.params = self.m_step(X, self.responsibilities)

            # проверяем сходимость
            score = self.compute_score(X, self.params)

            if self.has_converged(prev_score, score):
                break

            prev_score = score

        return self

    def initialize_parameters(self, X):
        """
        Задать начальные параметры модели.

        Например для GMM:
        - веса компонент
        - средние
        - ковариации
        """
        raise NotImplementedError

    def e_step(self, X, params):
        """
        E-step.

        Посчитать responsibilities:

            q(z_i = k) = p(z_i = k | x_i, params)

        То есть для каждой точки понять,
        насколько она относится к каждой скрытой компоненте.
        """
        raise NotImplementedError

    def m_step(self, X, responsibilities):
        """
        M-step.

        Обновить параметры модели,
        используя текущие responsibilities.

        Например для GMM:
        - пересчитать веса компонент
        - пересчитать средние
        - пересчитать ковариации
        """
        raise NotImplementedError

    def compute_score(self, X, params):
        """
        Посчитать качество текущих параметров.

        Обычно это log-likelihood:

            log p(X | params)
        """
        raise NotImplementedError

    def has_converged(self, prev_score, score):
        """
        Проверить условие остановки.
        """
        if prev_score is None:
            return False

        return abs(score - prev_score) < self.tol