import base64

from flask import Flask, request, jsonify
import numpy as np
import cv2

from predict import PredictionEngine, img_types


app = Flask(__name__)
engine = PredictionEngine()


def base64_to_cv2_img(encoded):
    encoded_data = encoded.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


@app.route("/")
def frontend():
    return app.send_static_file("index.html")


@app.route("/predict", methods=["POST"])
def predict_request():
    if not request.is_json:
        return jsonify("JSON data is required for this endpoint")

    encoded_img = request.json.get("image", None)
    if encoded_img is None:
        return jsonify("Missing attribute 'image' with base64 encoded image")

    img_type = request.json.get("type", None)
    if img_type not in img_types:
        return jsonify(f"Missing attribute 'type' with possible values {img_types}")

    img = base64_to_cv2_img(encoded_img)
    text = engine.predict_image(img, img_type)

    return jsonify(text=text)


def main():
    app.run("0.0.0.0", port=80)


if __name__ == "__main__":
    main()
