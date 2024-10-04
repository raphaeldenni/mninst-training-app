import keras as ks

def main() -> None:
    epochs_it: int = int(input("Enter the number of epochs: "))

    (train_images, train_labels), (test_images, test_labels) = ks.datasets.mnist.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Reshape data to include a single color channel
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

    model = ks.models.Sequential()

    # Add a Convolutional layer with 32 filters, a 3x3 kernel, and ReLU activation
    model.add(ks.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
    model.add(ks.layers.MaxPooling2D((2, 2)))  # Add a Max Pooling layer

    # Add another convolutional layer and pooling layer
    model.add(ks.layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(ks.layers.MaxPooling2D((2, 2)))

    # Flatten the results and add a fully connected layer
    model.add(ks.layers.Flatten())
    model.add(ks.layers.Dense(64, activation="relu"))

    # Output layer with 10 units for the 10 digit classes, with softmax activation
    model.add(ks.layers.Dense(10, activation="softmax"))

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    model.fit(
        train_images,
        train_labels,
        epochs=epochs_it,
        validation_data=(test_images, test_labels),
    )

    model.save("model.keras")

if __name__ == "__main__":
    main()
