import keras as ks
from keras import Sequential


def add_layers(model: Sequential, *args) -> Sequential:
    """Add layers to the model.

    Args:
        model (Sequential): The model to add layers to.
        *args: The arguments to pass to the model.

    Returns:
        Sequential: The model with the added layers.
    """
    # Add an input layer
    model.add(ks.layers.InputLayer(shape=(28, 28, 1)))

    # Add a Convolutional layer with 32 filters, a 3x3 kernel, and ReLU activation
    model.add(ks.layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(ks.layers.MaxPooling2D((2, 2)))  # Add a Max Pooling layer

    # Add another convolutional layer and pooling layer
    model.add(ks.layers.Conv2D(64, (3, 3), activation="relu", name="conv2d_last"))
    model.add(ks.layers.MaxPooling2D((2, 2)))

    # Flatten the results and add a fully connected layer
    model.add(ks.layers.Flatten())
    model.add(ks.layers.Dense(64, activation="relu"))

    # Dropout layer to prevent overfitting
    model.add(ks.layers.Dropout(args[0]))

    # Output layer with 10 units for the 10 digit classes, with softmax activation
    model.add(ks.layers.Dense(args[1], activation="softmax"))

    return model


def main() -> None:
    """Train a model to recognize handwritten digits."""
    # Get the data
    epochs_it: int = int(input("Enter the number of epochs: "))
    dropout_value: float = float(input("Enter the dropout value (0.0 to 1.0): "))
    dense_units: int = int(input("Enter the number of units in the dense layer: "))
    learning_rate: float = float(input("Enter the learning rate (0.0... to 1.0): "))

    (train_images, train_labels), (test_images, test_labels) = (
        ks.datasets.mnist.load_data()
    )

    # Normalize pixel values to be between 0 and 1
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Reshape data to include a single color channel
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

    # Build the model
    model = ks.models.Sequential()

    model = add_layers(model, dropout_value, dense_units)

    model.compile(
        optimizer=ks.optimizers.Adam(learning_rate=learning_rate),  # pyright: ignore
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    # Add EarlyStopping callback to stop training when validation accuracy stops improving
    early_stopping = ks.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=3, restore_best_weights=True, mode="max"
    )

    # Add TensorBoard callback to visualize training
    tensorboard = ks.callbacks.TensorBoard(log_dir="./logs")

    model.fit(
        train_images,
        train_labels,
        epochs=epochs_it,
        validation_data=(test_images, test_labels),
        callbacks=[early_stopping, tensorboard],
    )

    model.save("model.keras")


if __name__ == "__main__":
    main()
