import numpy as np


class OS_ELM:
    def __init__(self, input_size, output_size, hidden_layer_size, random_state=None):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_size = hidden_layer_size
        self.input_weights = np.random.uniform(-1, 1, size=[input_size, hidden_layer_size])
        self.bias = np.random.uniform(-1, 1, size=[hidden_layer_size])
        self.output_weights = np.random.uniform(-1, 1, size=[hidden_layer_size + 1, output_size])
        self.random_state = random_state
        self.H = None
        self.beta = None

    def relu(self, x):
        return np.maximum(0, x)

    def fit(self, X, y, regularization=0, batch_size=None):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        if batch_size is None:
            H = self.relu(np.dot(X, self.input_weights) + self.bias)
            H = np.hstack([H, np.ones((H.shape[0], 1))])  # add bias term
            self.beta = np.dot(np.linalg.pinv(H.T.dot(H) + regularization * np.eye(H.shape[1])), H.T.dot(y))
        else:
            n_samples = X.shape[0]
            n_batches = int(np.ceil(n_samples / batch_size))
            H_batches = []
            y_batches = []
            for i in range(n_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, n_samples)
                H_batch = self.relu(np.dot(X[start:end], self.input_weights) + self.bias)
                H_batch = np.hstack([H_batch, np.ones((H_batch.shape[0], 1))])  # add bias term
                H_batches.append(H_batch)
                y_batches.append(y[start:end])
            H = np.vstack(H_batches)
            y = np.vstack(y_batches)
            self.beta = np.dot(np.linalg.pinv(H.T.dot(H) + regularization * np.eye(H.shape[1])), H.T.dot(y))

        self.H = H

    def predict(self, X):
        H = self.relu(np.dot(X, self.input_weights) + self.bias)
        H = np.hstack([H, np.ones((H.shape[0], 1))])  # add bias term
        y_pred = np.dot(H, self.beta)
        return y_pred
