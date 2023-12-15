import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

def ResNetSimCLR(input_dim, output_dim):
    # Input layer
    inputs = Input(shape=(input_dim))

    # Base Encoder: ResNet50 without the top layer and pretrained weights
    base_encoder = ResNet50(include_top=False, weights=None, pooling='avg', input_tensor=inputs)
    
    # Pass inputs through the encoder
    h = base_encoder(inputs)

    # Projection Head: Two dense layers on top of the base encoder
    x = Dense(units=output_dim, activation='relu')(h)  # First layer with ReLU activation
    x = Dense(units=output_dim)(x)  # Second layer

    # Model outputs: raw features and projection head output
    return Model(inputs=inputs, outputs=[h, x])
