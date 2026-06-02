"""Modèle CNN simple pour la classification fresh / rotten."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@keras.utils.register_keras_serializable(package="FreshOrRotten")
class GradientReversal(layers.Layer):
    """Inverse le gradient pour réduire l'information liée au product_type."""

    def __init__(self, strength=0.1, **kwargs):
        super().__init__(**kwargs)
        self.strength = float(strength)

    def call(self, inputs):
        strength = self.strength

        @tf.custom_gradient
        def reverse_gradient(values):
            def grad(upstream_gradient):
                # Le backbone reçoit un gradient inversé pour rendre le product_type moins prédictible.
                return -strength * upstream_gradient

            return values, grad

        return reverse_gradient(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({"strength": self.strength})
        return config


def build_baseline_model(image_size, learning_rate=0.001, dropout_rate=0.3):
    """Construit et compile le baseline_model."""
    image_height, image_width = image_size
    input_shape = (image_height, image_width, 3)

    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, kernel_size=3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, kernel_size=3, activation="relu", padding="same"),
            layers.GlobalAveragePooling2D(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(dropout_rate),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="baseline_model",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )

    return model


def build_product_type_invariant_model(
    image_size,
    product_type_count,
    learning_rate=0.001,
    dropout_rate=0.3,
    adversarial_weight=0.1,
):
    """Construit un CNN qui limite la dépendance des features au product_type."""
    image_height, image_width = image_size
    input_shape = (image_height, image_width, 3)

    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, kernel_size=3, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling2D(name="feature_pooling")(x)
    features = layers.Dense(64, activation="relu", name="freshness_embedding")(x)
    x = layers.Dropout(dropout_rate)(features)

    freshness_output = layers.Dense(1, activation="sigmoid", name="freshness")(x)

    reversed_features = GradientReversal(strength=adversarial_weight, name="gradient_reversal")(features)
    product_type_hidden = layers.Dense(64, activation="relu")(reversed_features)
    product_type_output = layers.Dense(
        product_type_count,
        activation="softmax",
        name="product_type",
    )(product_type_hidden)

    model = keras.Model(
        inputs=inputs,
        outputs={
            "freshness": freshness_output,
            "product_type": product_type_output,
        },
        name="product_type_invariant_model",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "freshness": "binary_crossentropy",
            "product_type": "sparse_categorical_crossentropy",
        },
        metrics={
            "freshness": [
                keras.metrics.BinaryAccuracy(name="accuracy"),
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
            ],
            "product_type": [
                keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            ],
        },
    )

    return model
