from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import app as application


def image_bytes(image_format="PNG"):
    stream = BytesIO()
    Image.new("RGB", (32, 32), color="white").save(stream, format=image_format)
    return stream.getvalue()


class PredictionApiTests(unittest.TestCase):
    def setUp(self):
        self.upload_dir = TemporaryDirectory()
        self.addCleanup(self.upload_dir.cleanup)
        config_patch = patch.dict(application.app.config, {
            "TESTING": True,
            "UPLOAD_FOLDER": self.upload_dir.name,
            "MAX_CONTENT_LENGTH": 5 * 1024 * 1024,
        })
        config_patch.start()
        self.addCleanup(config_patch.stop)
        prediction_patch = patch.object(
            application.model, "predict", return_value=np.array([[0.9]])
        )
        self.predict = prediction_patch.start()
        self.addCleanup(prediction_patch.stop)
        self.client = application.app.test_client()

    def upload(self, data=None, filename="casting.png", route="/api/predict"):
        if data is None:
            data = image_bytes()
        return self.client.post(route, data={"image": (BytesIO(data), filename)})

    def assert_error(self, response, status):
        self.assertEqual(response.status_code, status)
        self.assertTrue(response.is_json)
        self.assertIsInstance(response.get_json()["error"], str)
        self.predict.assert_not_called()

    def test_ok_prediction_and_model_input(self):
        response = self.upload()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {
            "prediction": "PASS - OK", "status": "ok", "confidence": 90.0,
        })
        model_input = self.predict.call_args.args[0]
        self.assertEqual(model_input.shape, (1, 150, 150, 1))
        np.testing.assert_allclose(model_input, 1.0)

    def test_defective_prediction(self):
        self.predict.return_value = np.array([[0.2]])
        response = self.upload()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {
            "prediction": "DEFECTIVE", "status": "defective", "confidence": 80.0,
        })

    def test_threshold_is_classified_as_ok(self):
        self.predict.return_value = np.array([[0.5]])
        self.assertEqual(self.upload().get_json()["status"], "ok")

    def test_all_supported_image_formats(self):
        for image_format in ("JPEG", "PNG", "BMP"):
            with self.subTest(image_format=image_format):
                self.assertEqual(self.upload(image_bytes(image_format)).status_code, 200)

    def test_missing_image(self):
        self.assert_error(self.client.post("/api/predict"), 400)

    def test_json_body_is_not_an_image_upload(self):
        self.assert_error(self.client.post("/api/predict", json={"image": "test.jpg"}), 400)

    def test_missing_filename(self):
        self.assert_error(self.upload(filename=""), 400)

    def test_empty_file(self):
        self.assert_error(self.upload(b""), 400)

    def test_text_disguised_as_an_image(self):
        self.assert_error(self.upload(b"This is not an image", "photo.jpg"), 422)

    def test_corrupt_image(self):
        self.assert_error(self.upload(image_bytes()[:40]), 422)

    def test_unsupported_format_is_detected_from_contents(self):
        self.assert_error(self.upload(image_bytes("GIF"), "photo.jpg"), 415)

    def test_request_size_limit_returns_json(self):
        with patch.dict(application.app.config, {"MAX_CONTENT_LENGTH": 128}):
            self.assert_error(self.upload(b"x" * 1024), 413)

    def test_image_pixel_limit(self):
        with patch.object(application, "MAX_IMAGE_PIXELS", 16):
            self.assert_error(self.upload(), 413)

    def test_unsupported_method_returns_json_and_allow_header(self):
        response = self.client.get("/api/predict")
        self.assert_error(response, 405)
        self.assertIn("POST", response.headers["Allow"])

    def test_prediction_failure_returns_generic_json_error(self):
        self.predict.side_effect = RuntimeError("private model details")
        with self.assertLogs(application.app.logger, level="ERROR"):
            response = self.upload()
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json(), {"error": "Prediction failed. Please try again."})
        self.assertNotIn("private model details", response.get_data(as_text=True))

    def test_api_does_not_save_uploads(self):
        self.assertEqual(self.upload().status_code, 200)
        self.assertEqual(list(Path(self.upload_dir.name).iterdir()), [])

    def test_web_page_still_loads(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Analyze Image", response.data)
        self.predict.assert_not_called()

    def test_web_prediction_uses_unique_safe_filenames(self):
        for _ in range(2):
            response = self.upload(filename="../../photo.jpg", route="/")
            self.assertEqual(response.status_code, 200)
            self.assertIn(b"PASS - OK", response.data)
            self.assertIn(b"/static/uploads/", response.data)
        saved_files = list(Path(self.upload_dir.name).iterdir())
        self.assertEqual(len(saved_files), 2)
        for saved_file in saved_files:
            self.assertEqual(saved_file.suffix, ".png")
            self.assertEqual(saved_file.read_bytes(), image_bytes())

    def test_invalid_web_upload_displays_error(self):
        response = self.upload(b"invalid", route="/")
        self.assertEqual(response.status_code, 422)
        self.assertIn(b'role="alert"', response.data)
        self.assertIn(b"not a valid, readable image", response.data)


if __name__ == "__main__":
    unittest.main()
