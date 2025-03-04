import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

def build_cnn(input_shape=(128, 128, 3), num_classes=2):
    """Builds a Transfer Learning model using ResNet50."""
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    
    for layer in base_model.layers:
        layer.trainable = False  # Freeze pre-trained layers

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    return model