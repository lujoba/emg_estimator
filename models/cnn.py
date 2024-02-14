from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape

def build_model(input_shape, num_outputs, num_layers):
    """
    # Example usage:
    input_shape = (100, 256)  # Input shape: [batch, height, width]
    num_layers = 5  # Number of convolutional layers
    num_outputs = 5  # Assuming five (each finger) continuous output
    """
    model = Sequential()
    
    model.add(Reshape(input_shape + (1,), input_shape=input_shape))
    
    # Add the first convolutional layer
    model.add(Conv2D(32, kernel_size=(5, 5), activation='relu'))
    
    # Add additional convolutional layers
    for _ in range(num_layers - 1):
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Flatten the feature maps
    model.add(Flatten())
    
    # Add a fully connected layer
    model.add(Dense(128, activation='relu'))
    
    # Add the output layer
    model.add(Dense(num_outputs, activation='linear'))
    
    return model
