import keras
import matplotlib.pyplot as plt

epochs_it: int = 10

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = (
    keras.datasets.mnist.load_data()
)

# Normalize pixel values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape data to include a single color channel
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

model = keras.models.Sequential()

# Add a Convolutional layer with 32 filters, a 3x3 kernel, and ReLU activation
model.add(keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
model.add(keras.layers.MaxPooling2D((2, 2)))  # Add a Max Pooling layer

# Add another convolutional layer and pooling layer
model.add(keras.layers.Conv2D(64, (3, 3), activation="relu"))
model.add(keras.layers.MaxPooling2D((2, 2)))

# Flatten the results and add a fully connected layer
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation="relu"))

# Output layer with 10 units for the 10 digit classes, with softmax activation
model.add(keras.layers.Dense(10, activation="softmax"))

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

model.save("model.keras")

model1: keras.models.Sequential = keras.models.load_model("model.keras")  # pyright: ignore

history = model1.fit(
    train_images,
    train_labels,
    epochs=epochs_it,
    validation_data=(test_images, test_labels),
)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\nTest accuracy: {test_acc}")

# Plot training & validation accuracy values
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Test"], loc="upper left")
plt.show()

# Plot training & validation loss values
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Test"], loc="upper left")
plt.show()
