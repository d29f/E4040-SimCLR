import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

def ResNetSimCLR(input_dim, output_dim):
    # Input layer
    inputs = Input(shape=input_dim)

    # Base Encoder: ResNet50 without the top layer and pretrained weights
    base_encoder = ResNet50(include_top=False, weights='imagenet', pooling='avg', input_tensor=inputs)
    base_encoder.trainable = True
    
    # Pass inputs through the encoder
    h = base_encoder(inputs,training=True)

    # Projection Head: Three dense layers on top of the base encoder
    x = Dense(units=256, activation='relu')(h)
    x = Dense(units=128, activation='relu')(h)
    x = Dense(units=output_dim)(x)

    # Model outputs: raw features and projection head output
    return Model(inputs=inputs, outputs=[h, x])
