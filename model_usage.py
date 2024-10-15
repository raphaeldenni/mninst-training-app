import tkinter as tk

import keras as ks
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from signxai.methods.wrappers import calculate_relevancemap
from signxai.utils.utils import normalize_heatmap


def conv_explain(image) -> Figure:
    """Explain the prediction of a convolutional neural network.

    Args:
        image: The image to explain.
    """
    # Remove softmax
    model.layers[-1].activation = None  # type: ignore

    # Calculate relevancemaps
    R1 = calculate_relevancemap("gradient_x_sign_mu_0_5", np.array(image), model)
    R2 = calculate_relevancemap(
        "grad_cam", np.array(image), model, last_conv_layer_name="conv2"
    )

    # Construct visualization figure
    fig = Figure(figsize=(12, 4))
    plot1 = fig.add_subplot(131)
    plot2 = fig.add_subplot(132)
    plot3 = fig.add_subplot(133)

    plot1.imshow(image, cmap="gist_gray_r", clim=(-1, 1))
    plot1.set_title("input")

    plot2.matshow(normalize_heatmap(R1), cmap="seismic", clim=(-1, 1))
    plot2.set_title("Gradient x SIGN")

    plot3.matshow(normalize_heatmap(R2), cmap="seismic", clim=(-1, 1))
    plot3.set_title("Grad CAM")

    return fig


def embed_plot_in_tk(fig: Figure) -> FigureCanvasTkAgg:
    """Embed a plot in a tkinter window.

    Args:
        fig (Figure): The figure to embed.

    Returns:
        FigureCanvasTkAgg: The embedded plot.
    """
    fig_canvas = FigureCanvasTkAgg(fig, master=window)
    fig_canvas.draw()
    fig_canvas.get_tk_widget().grid(row=2, column=0)

    return fig_canvas


def conv_predict() -> tuple[int, int]:
    """Predict the digit drawn on the canvas.

    Returns:
        tuple[int, int]: The predicted digit and the second predicted digit.
    """
    # Get the image from the canvas
    image = draw_canva.postscript(file="digit.eps", colormode="color")
    image = ks.preprocessing.image.load_img(
        "digit.eps", color_mode="grayscale", target_size=(28, 28)
    )

    # Convert the image to an array
    image_array = ks.preprocessing.image.img_to_array(image)
    image_array = image_array / 255.0
    image_array = image_array.reshape((1, 28, 28, 1))

    # Predict the digit
    prediction = model.predict(image_array)  # type: ignore
    digit = prediction.argmax()
    second_digit = prediction.argsort()[0][-2]

    return digit, second_digit


def draw_canva_paint(event: tk.Event) -> None:
    """Paint on the canvas.

    Args:
        event (tk.Event): The event that triggered the function.
    """
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    draw_canva.create_oval(x1, y1, x2, y2, fill="black", width=5)


model = ks.models.load_model("model.keras")

# Display a window to draw with mouse a digit and add some buttons
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
    command=lambda: embed_plot_in_tk(
        conv_explain(draw_canva.postscript(file="digit.eps"))
    ),
)
explain_button.grid(row=4, column=0, columnspan=2, sticky="W", padx=100, pady=20)

# Bind the mouse to the canvas to draw only when the button is pressed
window.bind("<B1-Motion>", draw_canva_paint)

window.mainloop()
