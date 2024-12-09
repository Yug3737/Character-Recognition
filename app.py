# file: app.py
# author: Yug Patel
# last modified: 12 Dec 2024

import os
import json
import base64
import cv2
import dill
import pickle
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, redirect, url_for, request, jsonify

app = Flask(__name__)

SAVE_DIR = "saved_images"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


# # Activation functions and derivatives
def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


# def softmax(x):
#     exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stability adjustment
#     return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# Neural Network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.weights1 = np.random.randn(input_size, hidden_sizes[0]) * 0.01
        self.bias1 = np.zeros((1, hidden_sizes[0]))

        self.weights2 = np.random.randn(hidden_sizes[0], hidden_sizes[1]) * 0.01
        self.bias2 = np.zeros((1, hidden_sizes[1]))

        self.weights3 = np.random.randn(hidden_sizes[1], output_size) * 0.01
        self.bias3 = np.zeros((1, output_size))

        self.learning_rate = 0.01

    # Forward propagation
    def forward(self, X):
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = relu(self.z1)

        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = relu(self.z2)

        self.z3 = np.dot(self.a2, self.weights3) + self.bias3
        self.a3 = softmax(self.z3)

        return self.a3

    # Backpropagation
    # y is the true/actual output vector.
    # For digit 2, y would be [0,0,1,0,0,0,0,0,0,0,0]
    def backward(self, X, y, y_pred):
        m = y.shape[0]  # batch size

        # Gradients for output layer
        d_z3 = y_pred - y
        d_weights3 = np.dot(self.a2.T, d_z3) / m
        d_bias3 = np.sum(d_z3, axis=0, keepdims=True) / m

        # Gradients for second hidden layer
        d_a2 = np.dot(d_z3, self.weights3.T)
        d_z2 = d_a2 * relu_derivative(self.z2)
        d_weights2 = np.dot(self.a1.T, d_z2) / m
        d_bias2 = np.sum(d_z2, axis=0, keepdims=True) / m

        # Gradients for first hidden layer
        d_a1 = np.dot(d_z2, self.weights2.T)
        d_z1 = d_a1 * relu_derivative(self.z1)
        d_weights1 = np.dot(X.T, d_z1) / m
        d_bias1 = np.sum(d_z1, axis=0, keepdims=True) / m

        # Update weights and biases
        self.weights3 -= self.learning_rate * d_weights3
        self.bias3 -= self.learning_rate * d_bias3
        self.weights2 -= self.learning_rate * d_weights2
        self.bias2 -= self.learning_rate * d_bias2
        self.weights1 -= self.learning_rate * d_weights1
        self.bias1 -= self.learning_rate * d_bias1

    # Training function
    def train(self, X, y, epochs, batch_size):
        for epoch in range(epochs):
            # Shuffle data
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]

            # Mini-batch gradient descent
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i : i + batch_size]
                y_batch = y[i : i + batch_size]

                # Forward and backward propagation
                y_pred = self.forward(X_batch)
                self.backward(X_batch, y_batch, y_pred)

            # Compute loss for the epoch
            y_pred = self.forward(X)
            loss = -np.mean(
                np.sum(y * np.log(y_pred + 1e-8), axis=1)
            )  # Categorical cross-entropy
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

    # Predict function
    def predict(self, X):
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)

    def predict_top_3(self, test_input):
        y_pred = self.forward(test_input)
        top_3_indices = np.argsort(y_pred, axis=1)[:, -3:][
            :, ::-1
        ]  # Sort descending, get top 3
        top_3_probabilities = np.sort(y_pred, axis=1)[:, -3:][:, ::-1]
        result = []
        for i in range(top_3_indices.shape[0]):
            result.append(
                {
                    int(top_3_indices[i, j]): float(top_3_probabilities[i, j])
                    for j in range(3)
                }
            )
        print(result)
        print(type(result))
        return result


with open("digit_model.pkl", "rb") as file:
    digit_nn = dill.load(file)

with open("letter_model.pkl", "rb") as file:
    letter_nn = dill.load(file)


def softmax(x):
    print("X for softmax is: ", x)
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/digit_recognition")
def digit_recognition():
    return render_template("digit_recognition.html")


@app.route("/letter_recognition")
def letter_recognition():
    return render_template("letter_recognition.html")


@app.route("/save_digit_image", methods=["POST"])
def save_digit_image():
    data = request.get_json()
    if data:
        image_data = data["image"]
        image_data = image_data.split(",")[1]
        image_binary = base64.b64decode(image_data)
        image = np.asarray(bytearray(image_binary), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

        plt.imshow(image)
        image = cv2.resize(image, (28, 28))
        image = cv2.bitwise_not(image)

        image = np.array(image)

        image = image / 255.0
        pre_file_path = os.path.join(SAVE_DIR, "pre_digit_img.png")
        cv2.imwrite(pre_file_path, image * 255)
        image = image.flatten()

        print("SHAPE BEFORE predict", image.shape)

        output = digit_nn.predict_top_3(image)
        print(output)
        output = output[0]
        assert type(output) == dict
        top_3_indices = [index for index, prob in output.items()]
        top_3_probabilities = [prob for index, prob in output.items()]
        return (
            jsonify(
                {
                    "message": "Digit Image saved and prediction finished successfully",
                    "predictions": [
                        {"predicted_digit": digit, "probability": prob}
                        for digit, prob in zip(top_3_indices, top_3_probabilities)
                    ],
                }
            ),
            200,
        )


@app.route("/save_letter_image", methods=["POST"])
def save_letter_image():
    data = request.get_json()
    if data:
        image_data = data["image"]
        image_data = image_data.split(",")[1]
        image_binary = base64.b64decode(image_data)
        image = np.asarray(bytearray(image_binary), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (28, 28))
        image = cv2.bitwise_not(image)

        image = np.array(image)
        # print(image)

        image = image / 255.0

        pre_file_path = os.path.join(SAVE_DIR, "pre_letter_img.png")
        cv2.imwrite(pre_file_path, image * 255)
        image = image.flatten()

        print("SHAPE of img before forward pass", image.shape)

        letter_output = letter_nn.predict_top_3(image)
        print("letter output", letter_output)
        letter_output = letter_output[0]
        assert type(letter_output) == dict
        top_3_indices_ints = [index for index, prob in letter_output.items()]

        def index_to_letter(index):
            if 10 <= index <= 35:
                return chr(index + 55)
            elif 36 <= index <= 61:
                return chr(index + 61)
            else:
                raise ValueError(
                    f"Given: {index}, but wanted index b/w 10-61"
                )

        top_3_indices_letters = [index_to_letter(index) for index in top_3_indices_ints]
        top_3_probabilities = [prob for index, prob in letter_output.items()]
        return (
            jsonify(
                {
                    "message": "Letter Image saved and prediction finished successfully",
                    "predictions": [
                        {"predicted_letter": letter, "probability": prob}
                        for letter, prob in zip(
                            top_3_indices_letters, top_3_probabilities
                        )
                    ],
                }
            ),
            200,
        )


if __name__ == "__main__":
    app.run(debug=True, port=3333)
