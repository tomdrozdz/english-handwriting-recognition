import base64

from flask import Flask, request, jsonify
import numpy as np
import cv2

from predict import PredictionEngine, img_types


app = Flask(__name__)
engine = PredictionEngine()


def base64_to_cv2_img(encoded):
    """Tries to decode base64 encoded image and read it using opencv2."""
    split = encoded.split(",")
    encoded_data = split[1] if len(split) == 2 else split[0]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


@app.route("/")
def frontend():
    """Endpoint for accessing the main frontend page."""
    return app.send_static_file("index.html")


@app.route("/predict", methods=["POST"])
def predict_request():
    """
    Endpoint for making requests for text prediction. Accpets JSON data with base64
    encoded image and type of the image. Returns the predicted text or appropriate
    error message.
    """
    if not request.is_json:
        return jsonify("JSON data is required for this endpoint")

    encoded_img = request.json.get("image", None)
    if encoded_img is None:
        return jsonify("Missing attribute 'image' with base64 encoded image")

    img_type = request.json.get("type", None)
    if img_type not in img_types:
        return jsonify(f"Missing attribute 'type' with possible values {img_types}")

    try:
        img = base64_to_cv2_img(encoded_img)
    except Exception:
        return jsonify("Unable to decode the image from base64")

    text = engine.predict_image(img, img_type)
    return jsonify(text=text)


def main():
    """Run the server in debug mode."""
    app.run("0.0.0.0", port=3000, debug=True)


if __name__ == "__main__":
    main()
