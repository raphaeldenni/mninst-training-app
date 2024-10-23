# Digit MNIST Classifier

This is a simple digit classifier using the MNIST dataset. It uses a convolutional or dense neural network to classify the digits.

## Installation

It is recommended to use UV package manager to run this project. You can install it by following the instructions [here](https://docs.astral.sh/uv/getting-started/installation/).
After installing UV, you can go to the [Usage](#usage) section of this README to run the project.

Or you can install the required packages manually by using pyenv and pip:
```
pyenv install 3.12
pyenv local 3.12
pip install -r requirements.txt
```

## Usage

### Training

To train the model, run the following command with UV (recommended):

```
uv run model_trainer.py
```

Or with Python 3:
```
python3 model_trainer.py
```

This will train the model and save it to a file called `model.keras`.
You can also specify the following arguments:

- `model`: The model to use. Can be either `conv` or `dense`. Default is `conv`.
- `epochs`: The number of epochs to train for. Default is 20.
- `learning rate`: The learning rate. Default is 0.0003/0.0005 for convolutional/dense models.

### Testing

To test the model, run the following command with UV (recommended):

```
uv run model_usage.py
```

Or with Python 3:
```
python3 model_usage.py
```

## Best Hyperparameters

Best values for convolutional model:
- 20 epochs (stop at 19)
- 0.0003 learning rate

Best values for dense model:
- 20 epochs (stop at 17)
- 0.0005 learning rate
