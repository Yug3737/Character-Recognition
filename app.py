# file: app.py
# author: Yug Patel
# last modified: 13 Dec 2024

import os
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

with open("character_model.pkl", "rb") as file:
    character_nn = dill.load(file)

print(type(character_nn))
print(character_nn.keys() if isinstance(character_nn, dict) else character_nn)
weights = character_nn["weights"]
biases = character_nn["biases"]

def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stability adjustment
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def manual_predict(input_data):
    output = input_data
    for i in range(len(weights) - 1):
        output = relu(np.dot(output, weights[i]) + biases[i])
    output = softmax(np.dot(output, weights[-1]) + biases[-1])
    return output


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/character_recognition")
def character_recognition():
    return render_template("character_recognition.html")


@app.route("/process_character_image", methods=["POST"])
def process_character_image():
    data = request.get_json()
    if data:
        image_data = data["image"]
        image_data = image_data.split(",")[1]
        image_binary = base64.b64decode(image_data)
        image = np.asarray(bytearray(image_binary), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

        left_half = image[:, :400]
        right_half = image[:, 400:]

        def process_half(half_image, name):
            resized = cv2.resize(half_image, (28, 28))
            resized = cv2.bitwise_not(resized)
            resized = np.array(resized) / 255.0
            pre_file_path = os.path.join(SAVE_DIR, f"{name}_char_image.png")
            cv2.imwrite(pre_file_path, image * 255)
            return resized.flatten()

        left_image = process_half(left_half, "left")
        right_image = process_half(right_half, "right")

        left_output = manual_predict(left_image)
        right_output = manual_predict(right_image)


        return (
            jsonify(
                {
                    "message": "Images saved and prediction finished successfully",
                    "first_prediction": left_output,
                    "second_prediction": right_output,
                }
            ),
            200,
        )



if __name__ == "__main__":
    app.run(debug=True, port=3333)
