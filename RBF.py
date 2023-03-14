from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np


class RBFNet:
    def __init__(self, n_centers, sigma=1.0):
        self.n_centers = n_centers
        self.sigma = sigma
        self.centers = None
        self.weights = None
    
    def fit(self, X, y):
        # Use k-means to find the centers
        kmeans = KMeans(n_clusters=self.n_centers, random_state=0).fit(X)
        self.centers = kmeans.cluster_centers_

        # Calculate the distance between each data point and the centers
        dist = np.zeros((X.shape[0], self.n_centers))
        for i in range(self.n_centers):
            dist[:, i] = np.linalg.norm(X - self.centers[i], axis=1)

        # Apply the radial basis function to the distance matrix
        phi = np.exp(-1.0 / (2 * self.sigma**2) * dist**2)

        # Solve for the weights using linear regression
        self.weights = np.linalg.lstsq(phi, y, rcond=None)[0]
    
    def predict(self, X):
        # Calculate the distance between each data point and the centers
        dist = np.zeros((X.shape[0], self.n_centers))
        for i in range(self.n_centers):
            dist[:, i] = np.linalg.norm(X - self.centers[i], axis=1)

        # Apply the radial basis function to the distance matrix
        phi = np.exp(-1.0 / (2 * self.sigma**2) * dist**2)

        # Make predictions using the weights
        y_pred = phi.dot(self.weights)

        return y_pred


# Example usage
X_train = np.array([[0], [1], [2], [3], [4], [5]])
y_train = np.array([0, 0, 1, 1, 0, 0])

rbf = RBFNet(n_centers=2, sigma=1.0)
rbf.fit(X_train, y_train)

X_test = np.array([[1.5], [3.5]])
y_pred = rbf.predict(X_test)
print(y_pred)
