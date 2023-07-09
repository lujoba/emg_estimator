import tensorflow as tf


# Definir uma função que recebe uma matriz 3D como entrada
def my_function(matrix):
    return tf.reduce_sum(tf.square(matrix))

def compute_jacobian(matrix):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(matrix)
        output = my_function(matrix)  # Sua função personalizada

    jacobian = tape.gradient(output, matrix)
    return jacobian


# train_jacobian = train_dataset_f1.map(lambda x, y: (compute_jacobian(x), y))
# val_jacobian = val_dataset_f1.map(lambda x, y: (compute_jacobian(x), y))
