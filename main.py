import tensorflow as tf
import keras
import tensorflow_datasets as tfds

# Load the MNIST dataset
dataset_vars: object = tfds.load(
    "mnist",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

(ds_train, ds_test), ds_info = dataset_vars


# Normalize the images
def normalize_img(image, label) -> tuple[object, tf.Tensor]:
    """Normalizes images: `uint8` -> `float32`."""
    norm_image: object = tf.cast(image, tf.float32)
    norm_image = tf.divide(norm_image, 255.0)

    return norm_image, label


ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

# Define the model
model = keras.models.Sequential(
    [
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10),
    ]
)

model.compile(
    optimizer=keras.optimizers.Adam(0.001),  # pyright: ignore
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

# Train the model
model.fit(
    ds_train,
    epochs=6,
    validation_data=ds_test,
)
