import numpy as np

class FeedForwardNN:
    def __init__(self, layers):
        self.layers = layers
        self.weights = [np.random.randn(layers[i], layers[i-1]) for i in range(1, len(layers))]
        self.biases = [np.random.randn(layers[i], 1) for i in range(1, len(layers))]
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward_propagation(self, x):
        a = x
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = self.sigmoid(z)
        return a
    
    def train(self, X, y, epochs, learning_rate):
        for i in range(epochs):
            for x, t in zip(X, y):
                x = np.reshape(x, (len(x), 1))
                t = np.reshape(t, (len(t), 1))
                a = x
                activations = [a]
                zs = []
                for w, b in zip(self.weights, self.biases):
                    z = np.dot(w, a) + b
                    zs.append(z)
                    a = self.sigmoid(z)
                    activations.append(a)
                delta = (activations[-1] - t) * activations[-1] * (1 - activations[-1])
                deltas = [delta]
                for j in range(len(self.layers)-2, 0, -1):
                    delta = np.dot(self.weights[j].transpose(), delta) * activations[j] * (1 - activations[j])
                    deltas.append(delta)
                deltas.reverse()
                nabla_w = [np.dot(d, a.transpose()) for d, a in zip(deltas, activations[:-1])]
                nabla_b = deltas
                self.weights = [w - learning_rate * nw for w, nw in zip(self.weights, nabla_w)]
                self.biases = [b - learning_rate * nb for b, nb in zip(self.biases, nabla_b)]
                
    def predict(self, X):
        Y_pred = []
        for x in X:
            y = self.forward_propagation(x)
            Y_pred.append(y)
        return Y_pred
