# file: app.py
# author: Yug Patel
# last modified: 5 Dec 2024

import os
import base64
from io import BytesIO
from PIL import Image
from flask import Flask, render_template, redirect, url_for, request, jsonify

app = Flask(__name__)

SAVE_DIR = "saved_images"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


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
    try:
        data = request.get_json()
        if data:
            image_data = data["image"]
            image_data = image_data.split(",")[1]
            image_binary = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_binary))
            file_path = os.path.join(SAVE_DIR, "digit_img.png")
            image.save(file_path)
            return jsonify({"message": "Digit Image saved successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/save_letter_image", methods=["POST"])
def save_letter_image():
    try:
        data = request.get_json()
        if data:
            image_data = data["image"]
            image_data = image_data.split(",")[1]
            image_binary = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_binary))
            file_path = os.path.join(SAVE_DIR, "letter_img.png")
            image.save(file_path)
            return jsonify({"message": "Letter Image saved successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=3333)
