from io import BytesIO
from pathlib import Path
from uuid import uuid4

from flask import Flask, jsonify, render_template, request, url_for
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from werkzeug.exceptions import (
    BadRequest, HTTPException, RequestEntityTooLarge,
    UnprocessableEntity, UnsupportedMediaType,
)

BASE_DIR = Path(__file__).resolve().parent
IMG_SIZE = (150, 150)
MAX_IMAGE_PIXELS = 10_000_000
ALLOWED_FORMATS = {"JPEG": "jpg", "PNG": "png", "BMP": "bmp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = BASE_DIR / "static" / "uploads"
Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)

# Load model ONCE
model = tf.keras.models.load_model(BASE_DIR / "casting_defect_cnn.keras", compile=False)


def preprocess_image(img_source):
    """Prepare an image path or byte stream for the trained CNN."""
    img = image.load_img(
        img_source,
        color_mode="grayscale",
        target_size=IMG_SIZE
    )
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def prepare_upload(file):
    """Validate the actual image content before decoding it for prediction."""
    if file is None or not file.filename:
        raise BadRequest("Provide an image file in the 'image' form field.")

    data = file.read()
    if not data:
        raise BadRequest("The uploaded image is empty.")

    try:
        with Image.open(BytesIO(data)) as uploaded_image:
            if uploaded_image.format not in ALLOWED_FORMATS:
                raise UnsupportedMediaType("Supported image formats are JPEG, PNG, and BMP.")
            if uploaded_image.width * uploaded_image.height > MAX_IMAGE_PIXELS:
                raise RequestEntityTooLarge("Images must contain at most 10 million pixels.")
            extension = ALLOWED_FORMATS[uploaded_image.format]
            uploaded_image.verify()

        img_array = preprocess_image(BytesIO(data))
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
        raise UnprocessableEntity("The file is not a valid, readable image.") from error

    return img_array, data, extension


def predict_image(img_array):
    """Use the same label mapping and confidence calculation for both routes."""
    # Training labels: def_front = 0, ok_front = 1.
    ok_score = float(model.predict(img_array, verbose=0)[0][0])
    is_ok = ok_score >= 0.5
    confidence = ok_score if is_ok else 1 - ok_score
    return {
        "prediction": "PASS - OK" if is_ok else "DEFECTIVE",
        "status": "ok" if is_ok else "defective",
        "confidence": round(confidence * 100, 2),
    }


@app.errorhandler(HTTPException)
def handle_http_error(error):
    """Keep API errors in JSON, including routing and upload-size errors."""
    if request.path.startswith("/api/"):
        response = error.get_response()
        response.data = app.json.dumps({"error": error.description})
        response.content_type = "application/json"
        return response
    return error


@app.post("/api/predict")
def api_predict():
    img_array, _, _ = prepare_upload(request.files.get("image"))
    try:
        return jsonify(predict_image(img_array))
    except Exception:
        app.logger.exception("API prediction failed")
        return jsonify(error="Prediction failed. Please try again."), 500


@app.route("/", methods=["GET", "POST"])
def index():
    context = {
        "prediction": None,
        "status_code": None,
        "confidence": None,
        "confidence_ratio": 0,
        "image_path": None,
        "image_name": None,
        "image_info": None,
    }

    if request.method == "POST":
        try:
            file = request.files.get("image")
            img_array, data, extension = prepare_upload(file)
            result = predict_image(img_array)
            filename = f"{uuid4().hex}.{extension}"
            (Path(app.config["UPLOAD_FOLDER"]) / filename).write_bytes(data)
            with Image.open(BytesIO(data)) as uploaded_image:
                width, height = uploaded_image.size
        except HTTPException as error:
            return render_template("index.html", **context, error=error.description), error.code
        except Exception:
            app.logger.exception("Web prediction failed")
            return render_template(
                "index.html", **context, error="Prediction failed. Please try again."
            ), 500

        context.update(
            prediction=result["prediction"],
            status_code=result["status"],
            confidence=result["confidence"],
            confidence_ratio=result["confidence"] / 100,
            image_path=url_for("static", filename=f"uploads/{filename}"),
            image_name=Path(file.filename.replace("\\", "/")).name,
            image_info=f"{width} × {height} px · {len(data) / 1024:.1f} KB",
        )

    return render_template("index.html", **context)


if __name__ == "__main__":
    app.run()
