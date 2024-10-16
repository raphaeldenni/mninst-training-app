import tkinter as tk

import keras as ks
import numpy as np
from tf_explain.core.grad_cam import GradCAM


def draw_canva_paint(event: tk.Event) -> None:
    """Paint on the canvas.

    Args:
        event (tk.Event): The event that triggered the function.
    """
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    draw_canva.create_oval(x1, y1, x2, y2, fill="black", width=5)


def get_digit_image() -> np.ndarray:
    """Get the digit drawn on the canvas.

    Returns:
        np.ndarray: The image of the digit.
    """
    # Load the image from the file
    image = draw_canva.postscript(file="digit.eps", colormode="color")
    image = ks.preprocessing.image.load_img(
        "digit.eps", color_mode="grayscale", target_size=(28, 28)
    )

    # Convert the image to an array
    image = ks.preprocessing.image.img_to_array(image)
    image = image / 255.0

    return image


def conv_predict() -> tuple[int, int]:
    """Predict the digit drawn on the canvas.

    Returns:
        tuple[int, int]: The predicted digit and the second predicted digit.
    """
    # Get the image from the canvas
    image = get_digit_image()
    image = image.reshape((1, 28, 28, 1))

    # Predict the digit
    prediction = model.predict(image)  # type: ignore
    digit = prediction.argmax()
    second_digit = prediction.argsort()[0][-2]

    return digit, second_digit


def conv_explain() -> None:
    """Explain the prediction of a convolutional neural network.

    Args:
        image: The image to explain.
    """
    image = get_digit_image()

    # Use GradCAM for explanation
    explainer = GradCAM()
    grid = explainer.explain(
        (image, None), model, class_index=0, layer_name="conv2d_last"
    )

    explainer.save(grid, ".", "grad_cam_explanation.png")


# --- Main Program ---

# Load the model
model = ks.models.load_model("model.keras")

# Display a window to draw with mouse a digit and predict it
window: tk.Tk = tk.Tk()

draw_canva: tk.Canvas = tk.Canvas(window, width=280, height=280, bg="white")
draw_canva.grid(row=0, column=0, rowspan=4)

prediction_label: tk.Label = tk.Label(window, text="Prediction: ")
prediction_label.grid(row=0, column=1, sticky="NW", padx=20)

second_prediction_label: tk.Label = tk.Label(window, text="Second Prediction: ")
second_prediction_label.grid(row=1, column=1, sticky="NW", padx=20)

predict_button: tk.Button = tk.Button(
    window,
    text="Predict",
    command=lambda: [
        prediction_label.config(text=f"Prediction: {conv_predict()[0]}"),
        second_prediction_label.config(text=f"Second Prediction: {conv_predict()[1]}"),
    ],
)
predict_button.grid(row=2, column=1, sticky="NW", padx=20)

clear_button: tk.Button = tk.Button(
    window, text="Clear", command=lambda: draw_canva.delete("all")
)
clear_button.grid(row=3, column=1, sticky="NW", padx=20)

explain_button: tk.Button = tk.Button(
    window,
    text="Explain",
    command=lambda: conv_explain(),
)
explain_button.grid(row=4, column=0, columnspan=2, sticky="W", padx=100, pady=20)

# Bind the mouse to the canvas to draw only when the button is pressed
window.bind("<B1-Motion>", draw_canva_paint)

window.mainloop()
