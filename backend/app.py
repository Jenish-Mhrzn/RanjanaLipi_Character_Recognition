from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from utils.predict import predict_ranjana

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file   = request.files['image']
        image  = Image.open(file.stream)
        result = predict_ranjana(image)

        if result and 'error' in result:
            return jsonify(result), 400

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/test")
def test():
    return "Flask backend is working!"

if __name__ == "__main__":
    app.run(debug=True)