import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

iris = load_iris()

X = iris.data         
y = iris.target       
target_names = iris.target_names


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


gmm = GaussianMixture(
    n_components=3,
    covariance_type="full",
    max_iter=100,
    tol=1e-4,
    random_state=42
)

gmm.fit(X_scaled)


predicted_labels = gmm.predict(X_scaled)

responsibilities = gmm.predict_proba(X_scaled)


pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)


print("Сошёлся ли EM:", gmm.converged_)
print("Число итераций:", gmm.n_iter_)

print("\nВеса компонент:")
print(gmm.weights_)

print("\nСредние компонент:")
print(gmm.means_)

print("\nПервые 5 responsibilities:")
print(responsibilities[:5])

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=predicted_labels,
    s=50,
    alpha=0.8
)

plt.title("Iris: кластеры, найденные GMM через EM")
plt.xlabel("PCA component 1")
plt.ylabel("PCA component 2")
plt.grid(True)

plt.show()