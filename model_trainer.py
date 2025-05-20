import keras as ks
from keras import Sequential

# INPUT_LAYER_SHAPE: tuple[int, int, int] = (28, 28, 1)

CONV_LAYER_SIZE: int = 32
DENSE_LAYER_SIZE: int = 512

OUTPUT_LAYER_SIZE: int = 10


def add_conv_layers(model: Sequential) -> Sequential:
    """Add convolutional layers to a model.

    Args:
        model (Sequential): The model to add layers to.

    Returns:
        Sequential: The model with the added layers.
    """
    # Add a Convolutional layer with 32 filters, a 3x3 kernel, and ReLU activation + pooling layer
    model.add(ks.layers.Conv2D(CONV_LAYER_SIZE, (3, 3), activation="relu"))
    model.add(ks.layers.MaxPooling2D((2, 2)))

    # Add another convolutional layer and pooling layer
    model.add(
        ks.layers.Conv2D(
            CONV_LAYER_SIZE * 2, (3, 3), activation="relu", name="conv2d_last"
        )
    )
    model.add(ks.layers.MaxPooling2D((2, 2)))

    # Flatten the results and add a fully connected layer
    model.add(ks.layers.Flatten())
    model.add(ks.layers.Dense(64, activation="relu"))

    return model


def add_dense_layers(model: Sequential) -> Sequential:
    """Add dense layers to a model.

    Args:
        model (Sequential): The model to add layers to.

    Returns:
        Sequential: The model with the added layers.
    """
    # model.add(ks.layers.Flatten())
    model.add(ks.layers.Dense(DENSE_LAYER_SIZE, activation="relu"))

    model.add(ks.layers.Dropout(0.1))

    model.add(ks.layers.Dense(DENSE_LAYER_SIZE, activation="relu"))

    return model


def main() -> None:
    """Train a model to recognize handwritten digits."""
    # --- Data Preparation ---

    # Model type
    model_type: str = input("Enter the model type (conv or dense): ")

    if model_type is None:
        model_type = "conv"

    # Get user input
    epochs_it: int = int(input("Enter the number of epochs: "))
    learning_rate: float = float(input("Enter the learning rate (0.0... to 1.0): "))

    (train_images, train_labels), (test_images, test_labels) = (
        ks.datasets.mnist.load_data()
    )

    # Reshape data to include a single color channel
    match model_type:
        case "conv":
            train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
            test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

        case "dense":
            train_images = train_images.reshape((train_images.shape[0], 28 * 28))
            test_images = test_images.reshape((test_images.shape[0], 28 * 28))

    # Normalize pixel values to be between 0 and 1
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # --- Model Building ---

    model = ks.models.Sequential()

    # Add an input layer with the shape of the input data
    # model.add(ks.layers.InputLayer(shape=INPUT_LAYER_SHAPE))

    # Add intermediate layers
    match model_type:
        case "conv":
            model = add_conv_layers(model)

        case "dense":
            model = add_dense_layers(model)

    # Output layer with 10 units for the 10 digit classes, with softmax activation + dropout to prevent overfitting
    model.add(ks.layers.Dropout(0.5))
    model.add(ks.layers.Dense(OUTPUT_LAYER_SIZE, activation="softmax"))

    model.compile(
        optimizer=ks.optimizers.Adam(learning_rate=learning_rate),  # pyright: ignore
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # --- Callbacks ---

    # Add EarlyStopping callback to stop training when validation accuracy stops improving
    early_stopping = ks.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=3, restore_best_weights=True, mode="max"
    )

    # Add TensorBoard callback to visualize training
    tensorboard = ks.callbacks.TensorBoard(log_dir="./logs")

    # --- Train the model ---

    model.fit(
        train_images,
        train_labels,
        epochs=epochs_it,
        validation_data=(test_images, test_labels),
        callbacks=[early_stopping, tensorboard],
    )

    model.summary()

    model.save("model.keras")


if __name__ == "__main__":
    main()
