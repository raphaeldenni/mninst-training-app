import tkinter as tk

import keras as ks
import matplotlib.pyplot as plt
import numpy as np
from signxai.methods.wrappers import calculate_relevancemap
from signxai.utils.utils import normalize_heatmap


def conv_explain(image) -> None:
    """Explain the prediction of a convolutional neural network.

    Args:
        image: The image to explain.
    """
    # Remove softmax
    model.layers[-1].activation = None

    # Calculate relevancemaps
    R1 = calculate_relevancemap("gradient_x_sign_mu_0_5", np.array(image), model)
    R2 = calculate_relevancemap(
        "grad_cam", np.array(image), model, last_conv_layer_name="conv2"
    )

    # Visualize heatmaps
    fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(12, 4))

    axs[0].imshow(image, cmap="gist_gray_r", clim=(-1, 1))
    axs[0].set_title("input")

    axs[1].matshow(normalize_heatmap(R1), cmap="seismic", clim=(-1, 1))
    axs[1].set_title("Gradient x SIGN")

    axs[2].matshow(normalize_heatmap(R2), cmap="seismic", clim=(-1, 1))
    axs[2].set_title("Grad CAM")

    plt.show()


def conv_predict() -> tuple[int, int]:
    """Predict the digit drawn on the canvas.

    Returns:
        tuple[int, int]: The predicted digit and the second predicted digit.
    """
    # Get the image from the canvas
    image = canvas.postscript(file="digit.eps", colormode="color")
    image = ks.preprocessing.image.load_img(
        "digit.eps", color_mode="grayscale", target_size=(28, 28)
    )

    # Convert the image to an array
    image_array = ks.preprocessing.image.img_to_array(image)
    image_array = image_array / 255.0
    image_array = image_array.reshape((1, 28, 28, 1))

    # Predict the digit
    prediction = model.predict(image_array)
    digit = prediction.argmax()
    second_digit = prediction.argsort()[0][-2]

    conv_explain(image_array)

    return digit, second_digit


def canva_paint(event: tk.Event) -> None:
    """Paint on the canvas.

    Args:
        event (tk.Event): The event that triggered the function.
    """
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    canvas.create_oval(x1, y1, x2, y2, fill="black", width=5)


model: ks.models.Sequential = ks.models.load_model("model.keras")

# Display a window to draw with mouse a digit and add some buttons
window: tk.Tk = tk.Tk()

canvas: tk.Canvas = tk.Canvas(window, width=280, height=280, bg="white")
canvas.pack()

prediction_label: tk.Label = tk.Label(window, text="Prediction: ")
prediction_label.pack()

second_prediction_label: tk.Label = tk.Label(window, text="Second Prediction: ")
second_prediction_label.pack()

predict_button: tk.Button = tk.Button(
    window,
    text="Predict",
    command=lambda: [
        prediction_label.config(text=f"Prediction: {conv_predict()[0]}"),
        second_prediction_label.config(text=f"Second Prediction: {conv_predict()[1]}"),
    ],
)
predict_button.pack()

clear_button: tk.Button = tk.Button(
    window, text="Clear", command=lambda: canvas.delete("all")
)
clear_button.pack()

# Bind the mouse to the canvas to draw only when the button is pressed
window.bind("<B1-Motion>", canva_paint)

window.mainloop()
