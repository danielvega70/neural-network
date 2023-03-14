import numpy as np
import matplotlib.pyplot as plt


class KohonenSOM:
    def __init__(self, map_shape, sigma=1.0, learning_rate=0.5, n_iterations=1000):
        self.map_shape = map_shape
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = np.random.normal(size=(map_shape[0], map_shape[1], map_shape[2]))

    def _get_bmu(self, x):
        # Calculate the distance between the input and each weight
        dist = np.linalg.norm(self.weights - x, axis=2)

        # Find the index of the best matching unit (BMU)
        bmu_idx = np.unravel_index(np.argmin(dist), self.map_shape[:2])

        return bmu_idx

    def fit(self, X):
        for iteration in range(self.n_iterations):
            # Randomly sample an input vector
            x = X[np.random.randint(X.shape[0])]

            # Find the best matching unit
            bmu_idx = self._get_bmu(x)

            # Update the weights of the BMU and its neighbors
            for i in range(self.map_shape[0]):
                for j in range(self.map_shape[1]):
                    dist = np.linalg.norm(np.array([i, j]) - np.array(bmu_idx))
                    if dist <= self.sigma:
                        self.weights[i, j] += self.learning_rate * (x - self.weights[i, j])

            # Reduce the learning rate and neighborhood size over time
            self.learning_rate = 0.5 * (1 - iteration / self.n_iterations)
            self.sigma = self.sigma * (1 - iteration / self.n_iterations)

    def predict(self, X):
        # Find the best matching unit for each input
        bmu_idxs = np.zeros(X.shape[0], dtype=np.int)
        for i, x in enumerate(X):
            bmu_idxs[i] = self._get_bmu(x)

        return bmu_idxs


# Example usage
# Create a toy dataset with 3 clusters
X = np.vstack([
    np.random.normal(loc=[0, 0], scale=0.5, size=(100, 2)),
    np.random.normal(loc=[2, 2], scale=0.5, size=(100, 2)),
    np.random.normal(loc=[-2, 2], scale=0.5, size=(100, 2))
])
y = np.hstack([np.zeros(100), np.ones(100), 2 * np.ones(100)])

# Normalize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train the SOM on the data
som = KohonenSOM(map_shape=(10, 10, 2), sigma=5.0, learning_rate=0.5, n_iterations=1000)
som.fit(X)

# Plot the SOM and the cluster assignments
fig, ax = plt.subplots(figsize=(10, 10))
for i, x in enumerate(X):
    bmu_idx = som._get_bmu(x)
    ax.scatter(bmu_idx[1], bmu_idx[0], color=plt.cm.Set1(y[i] / 3.), alpha=0.5)
ax.imshow(np.sum(som.weights, axis=2), cmap='gray')
ax.set_xticks([])
ax.set_yticks([])
plt.show()
