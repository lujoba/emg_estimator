import tensorflow as tf
import numpy as np


# class NMF:
#     """
#     Compute Non-negative Matrix Factorization (NMF)
#     Based on implementation by: Eesung Kim <eesungk@gmail.com>
#     """
#     def __init__(self, max_iter=200, learning_rate=0.01, display_step=10, optimizer='mu', initW=False):
#         self.max_iter = max_iter
#         self.learning_rate = learning_rate
#         self.display_step = display_step
#         self.optimizer = optimizer
# 
#     def NMF(self, X, r_components, learning_rate, max_iter, display_step, optimizer, initW, givenW):
#         m, n = np.shape(X[0])
# 
#         V = tf.keras.Input(shape=(n,), dtype=tf.float32)
# 
#         initializer = tf.keras.initializers.RandomUniform(minval=0.0, maxval=1.0)
#         if not initW:
#             W = tf.keras.layers.Dense(r_components, activation='relu', name="W", kernel_initializer=initializer)(V)
#             H = tf.keras.layers.Dense(n, activation='relu', name="H", kernel_initializer=initializer)(W)
#         else:
#             W = tf.keras.layers.Dense(r_components, activation='relu', name="W", use_bias=False, kernel_initializer=tf.constant_initializer(givenW))(V)
#             H = tf.keras.layers.Dense(n, activation='relu', name="H", kernel_initializer=initializer)(W)
# 
#         WH = tf.matmul(W, H)
#         cost = tf.reduce_sum(tf.square(V - WH))
# 
#         for _ in range(max_iter):
#             if optimizer == 'mu':
#                 """Compute Non-negative Matrix Factorization with Multiplicative Update"""
#                 Wt = tf.transpose(W)
#                 H_new = H * tf.matmul(Wt, V) / tf.matmul(tf.matmul(Wt, W), H)
#                 H = H_new
# 
#                 if not initW:
#                     Ht = tf.transpose(H)
#                     W_new = W * tf.matmul(V, Ht) / tf.matmul(W, tf.matmul(H, Ht))
#                     W = W_new
# 
#             elif optimizer == 'pg':
#                 """Optimization; Projected Gradient method"""
#                 with tf.GradientTape() as tape:
#                     tape.watch([W, H])
#                     loss = cost
#                 gradients = tape.gradient(loss, [W, H])
#                 W, H = self.update_parameters(W, H, gradients, learning_rate)
# 
#         return W, H
# 
#     def update_parameters(self, W, H, gradients, learning_rate):
#         dW, dH = gradients
#         W -= learning_rate * dW
#         H -= learning_rate * dH
#         W = tf.where(tf.less(W, 0.0), tf.zeros_like(W), W)
#         H = tf.where(tf.less(H, 0.0), tf.zeros_like(H), H)
#         return W, H
# 
#     def fit_transform(self, X, r_components, initW, givenW):
#         """Transform input data to W, H matrices which are the non-negative matrices."""
#         W, H = self.NMF(X, r_components, self.learning_rate, self.max_iter, self.display_step, self.optimizer, initW, givenW)
#         return W, H
# 
#     def inverse_transform(self, W, H):
#         """Transform data back to its original space."""
#         return np.matmul(W, H)

class NMF(tf.Module):
    def __init__(self, num_components, max_iter=100, tol=1e-5):
        super(NMF, self).__init__()
        self.num_components = num_components
        self.max_iter = max_iter
        self.tol = tol

    def initialize_factors(self, shape):
        return tf.abs(tf.random.normal(shape))

    def fit(self, V):
        V = tf.convert_to_tensor(V, dtype=tf.float32)
        num_samples, num_features = V.shape

        W = self.initialize_factors([num_samples, self.num_components])
        H = self.initialize_factors([self.num_components, num_features])

        for _ in range(self.max_iter):
            # Update H
            WH = tf.matmul(W, H)
            WH_eps = tf.maximum(WH, tf.keras.backend.epsilon())
            ratio = tf.divide(V, WH_eps)
            update_H = tf.matmul(tf.transpose(W), ratio)
            H *= update_H

            # Update W
            WH = tf.matmul(W, H)
            WH_eps = tf.maximum(WH, tf.keras.backend.epsilon())
            ratio = tf.divide(V, WH_eps)
            update_W = tf.matmul(ratio, tf.transpose(H))
            W *= update_W

            # Compute Frobenius norm
            frobenius_norm = tf.norm(V - WH, ord='fro', axis=[0, 1])

            # Check convergence
            if tf.reduce_mean(frobenius_norm) < self.tol:
                break

        return W, H


def build_model(input_shape, num_components, output_shape):
    """
    Builds a sequential model with a single NMF layer.

    Args:
    input_shape: The shape of the input data.
    num_components: The number of components for the NMF layer.
    output_shape: The shape of the output data.
    """
    # Definindo modelo de pré-processamento
    preprocessing_model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.ReLU()
    ])
    
    # Definindo modelo completo
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = preprocessing_model(inputs)  # Aplicando pré-processamento
    nmf = NMF(num_components=num_components)
    W, _ = nmf.fit(x)  # Executando a função fit da NMF
    outputs = tf.keras.layers.Dense(output_shape, activation="linear")(W)
    
    model = tf.keras.Model(inputs, outputs)
    return model
