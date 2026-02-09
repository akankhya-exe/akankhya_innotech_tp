import numpy as np
import matplotlib.pyplot as plt
import time

lr = 0.01
iterations = 1000

def gen_data(n_samples=100, n_features=1, noise=5.0, random_seed=42):
    """
    Generates synthetic data for testing purposes
    
    :param n_samples: Number of data entries (rows)
    :param n_features: Number of features (columns)
    :param noise: Noise added to the output for the model to decipher
    :param random_seed: Reproduce results

    Returns:
    X: Input features with dimension (n_samples, n_features)
    y: True labels of (n_samples)
    true_w: Actual weight (verification)
    true_b: Actual bias (verification)
    """
    np.random.seed(random_seed)
    X = 2.0 * np.random.randn(n_samples, n_features)
    true_w = np.random.randn(n_features) * 10
    true_b = np.random.randn() * 5 
    y = np.dot(X, true_w) + true_b
    y += np.random.randn(n_samples) * noise
    return X, y, true_w, true_b

def train_test_split(X, y, test_size=0.2, random_seed=42):
    np.random.seed(random_seed)
    indices = np.random.permutation(X.shape[0])
    split_row = int(X.shape[0] * (1 - test_size))
    train_idx, test_idx = indices[:split_row], indices[split_row:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

class LinearRegression:
    def __init__(self, lr=lr, iterations=iterations):
        self.lr = lr
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.loss_history = [] 

    def fit(self, X, y):
        n_samples, n_features = X.shape[0], X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.iterations):
            dw = np.zeros(n_features)
            db = 0
            for i in range(n_samples):
                y_pred = 0
                for j in range(n_features):
                    y_pred += self.weights[j] * X[i][j]
                y_pred += self.bias
                for j in range(n_features):
                    dw[j] += (2/n_samples) * (y_pred - y[i]) * X[i][j]
                db += (2/n_samples) * (y_pred - y[i])
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db
            current_loss = self.cost_function(X, y)
            self.loss_history.append(current_loss)

    def predict(self, X):
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples)
        for i in range(n_samples):
            prediction = 0
            for j in range(len(self.weights)):
                prediction += self.weights[j] * X[i][j]
            y_pred[i] = prediction + self.bias
        return y_pred

    def cost_function(self, X, y):
        total_loss = 0
        n_samples = X.shape[0]
        for i in range(n_samples):
            y_pred = 0
            for j in range(len(self.weights)):
                y_pred += self.weights[j] * X[i][j]
            y_pred += self.bias
            error = (y_pred - y[i]) ** 2
            total_loss += error
        return total_loss / n_samples

class LinearRegressionVectorized:
    def __init__(self, lr=lr, iterations=iterations):
        self.lr = lr
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.loss_history = [] 

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (2 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (2 / n_samples) * np.sum(y_pred - y)
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db
            loss = np.mean((y_pred - y)**2)
            self.loss_history.append(loss)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

if __name__ == "__main__":
    X, y, _, _ = gen_data(n_samples=10000, n_features=20)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    print(f"Dataset split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")

    print("Method 1: Without vectorization")
    model1 = LinearRegression(lr, iterations=50)
    start = time.time()
    model1.fit(X_train, y_train)
    loop_time = time.time() - start
    print(f"Loop training time: {loop_time:.4f}")

    print("Method 2: With vectorization")
    model2 = LinearRegressionVectorized(lr, iterations=50)
    start = time.time()
    model2.fit(X_train, y_train)
    vec_time = time.time() - start
    print(f"Vectorized training time: {vec_time:.4f}")

    test_preds = model2.predict(X_test)
    test_mse = np.mean((test_preds - y_test)**2)
    print(f"Test Set MSE: {test_mse:.4f}")

    X_small, y_small, true_w, true_b = gen_data(n_samples=100, n_features=1)
    Xs_train, Xs_test, ys_train, ys_test = train_test_split(X_small, y_small, test_size=0.2)
    
    vis_model = LinearRegressionVectorized(lr=0.05, iterations=50)
    vis_model.fit(Xs_train, ys_train)
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(vis_model.loss_history)
    plt.title("Training Loss Convergence")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.scatter(Xs_test, ys_test, color='blue', label='Test Data')
    plt.plot(Xs_test, vis_model.predict(Xs_test), color='red', label='Test Prediction')
    plt.title("Model Performance on Test Set")
    plt.legend()
    plt.tight_layout()
    plt.show()

