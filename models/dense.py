import tensorflow as tf

def build_model(input_shape, output_shape, num_layers):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))
    
    # Adiciona uma camada Flatten para transformar o tensor 2D em 1D
    model.add(tf.keras.layers.Flatten())
    
    for _ in range(num_layers):
        model.add(tf.keras.layers.Dense(512, activation='relu'))  # 512 é o número de unidades em cada camada oculta
    
    model.add(tf.keras.layers.Dense(output_shape, activation='linear')) # Linear para regrassão
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
