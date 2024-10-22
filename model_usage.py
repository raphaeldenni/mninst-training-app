import tkinter as tk

import keras as ks
import numpy as np
import shap

DEFAULT_PEN_WIDTH: int = 15


def draw_canva_paint(event: tk.Event) -> None:
    """Paint on the canvas.

    Args:
        event (tk.Event): The event that triggered the function.
    """
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)

    draw_canva.create_oval(x1, y1, x2, y2, fill="black", width=pen_width)


def get_pen_width(event: tk.Event) -> None:
    """Get the pen width from the scale.

    Args:
        event (tk.Event): The event that triggered the function.
    """
    global pen_width
    pen_width = int(pen_width_slider.get())


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

    # Convert the image
    image = ks.preprocessing.image.img_to_array(image)
    image = image / 255.0
    image = image.reshape((1, 28, 28, 1))

    return image


def model_predict() -> tuple[int, int]:
    """Predict the digit drawn on the canvas.

    Returns:
        tuple[int, int]: The predicted digit and the second predicted digit.
    """
    # Get the image from the canvas
    image = get_digit_image()

    # Predict the digit
    prediction = model.predict(image)  # type: ignore
    digit = prediction.argmax()
    second_digit = prediction.argsort()[0][-2]

    return digit, second_digit


def predict_explain() -> None:
    """Explain the prediction of a convolutional neural network."""
    digit_image = get_digit_image()
    (train_images, _), _ = ks.datasets.mnist.load_data()

    train_images = (
        train_images[:1000].reshape((-1, 28, 28, 1)).astype(np.float32) / 255.0
    )

    # Use SHAP DeepExplainer for CNN models
    explainer = shap.DeepExplainer(model, train_images)
    shap_values = explainer.shap_values(digit_image)

    # Display the SHAP summary plot for the image
    shap.image_plot(shap_values, digit_image)


# --- Main Program ---

# Load the model
model = ks.models.load_model("model.keras")

# Display a window to draw with mouse a digit and predict it
window: tk.Tk = tk.Tk()

pen_width: int = DEFAULT_PEN_WIDTH

draw_canva: tk.Canvas = tk.Canvas(window, width=280, height=280, bg="white")
draw_canva.grid(row=0, column=0, rowspan=6)

prediction_label: tk.Label = tk.Label(window, text="Prediction: ")
prediction_label.grid(row=0, column=1, sticky="NW", padx=20)

second_prediction_label: tk.Label = tk.Label(window, text="Second Prediction: ")
second_prediction_label.grid(row=1, column=1, sticky="NW", padx=20)

pen_width_slider: tk.Scale = tk.Scale(
    window,
    from_=10,
    to=20,
    orient="horizontal",
    label="Width",
    length=200,
    command=get_pen_width,  # type: ignore
)
pen_width_slider.set(DEFAULT_PEN_WIDTH)
pen_width_slider.grid(row=2, column=1, sticky="NW", padx=20)

predict_button: tk.Button = tk.Button(
    window,
    text="Predict",
    command=lambda: [
        prediction_label.config(text=f"Prediction: {model_predict()[0]}"),
        second_prediction_label.config(text=f"Second Prediction: {model_predict()[1]}"),
    ],
)
predict_button.grid(row=3, column=1, sticky="NW", padx=20)

explain_button: tk.Button = tk.Button(
    window,
    text="Explain",
    command=lambda: predict_explain(),
)
explain_button.grid(row=4, column=1, sticky="NW", padx=20)

clear_button: tk.Button = tk.Button(
    window, text="Clear", command=lambda: draw_canva.delete("all")
)
clear_button.grid(row=5, column=1, sticky="NW", padx=20)

# Bind the mouse to the canvas to draw only when the button is pressed
draw_canva.bind("<B1-Motion>", draw_canva_paint)

window.mainloop()
