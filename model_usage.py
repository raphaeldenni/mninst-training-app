import tkinter as tk

import keras as ks

# Function to draw on the canvas
def paint(event) -> None:
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    canvas.create_oval(x1, y1, x2, y2, fill="black", width=5)


# Function to predict the digit drawn
def predict() -> tuple[int, int]:
    # Get the image from the canvas
    image = canvas.postscript(file="digit.eps", colormode="color")
    image = ks.preprocessing.image.load_img(
        "digit.eps", color_mode="grayscale", target_size=(28, 28)
    )
    image = ks.preprocessing.image.img_to_array(image)
    image = image / 255.0
    image = image.reshape((1, 28, 28, 1))

    # Predict the digit
    prediction = model.predict(image)
    digit = prediction.argmax()
    second_digit = prediction.argsort()[0][-2]

    return digit, second_digit


model: ks.models.Sequential = ks.models.load_model("model.keras")

# Display a window to draw with mouse a digit
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
        prediction_label.config(text=f"Prediction: {predict()[0]}"),
        second_prediction_label.config(text=f"Second Prediction: {predict()[1]}"),
    ],
)
predict_button.pack()

clear_button: tk.Button = tk.Button(
    window, text="Clear", command=lambda: canvas.delete("all")
)
clear_button.pack()

# Bind the mouse to the canvas to draw only when the button is pressed
window.bind("<B1-Motion>", paint)

window.mainloop()
