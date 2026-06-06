import React, { useState } from "react";
import "./App.css";

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    if (!file.type.startsWith("image/")) {
      setError("Please upload a valid image file (PNG, JPG, JPEG)");
      return;
    }

    if (file.size > 5 * 1024 * 1024) {
      setError("Image size should be less than 5MB");
      return;
    }

    setImage(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setError("");
  };

  const handleSubmit = async () => {
    if (!image) {
      setError("Please upload an image first!");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append("image", image);

    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      // Backend returned an error (quality fail or not Ranjana)
      if (data.error) {
        setError(data.error);
        return;
      }

      // Warning or success 
      console.log(data)
      setResult(data);
     
    } catch (err) {
      console.error(err);
      setError(err.message || "Failed to connect to backend.");
    } finally {
      setLoading(false);
    }
  };

  const resetAll = () => {
    setImage(null);
    setPreview(null);
    setResult(null);
    setError("");
  };

  return (
    <div className="app-container">
      <div className="card">
        {/* Header */}
        <h1 className="title">Handwritten RanjanaLipi</h1>
        {/* <h1 className="title">Ranjanalipi</h1> */}
        <p className="title-sub">Character Recognition</p>
        <p className="subtitle">
          Upload a clear image of <strong>one single character</strong>
        </p>

        {/* Guidelines */}
        <div className="instructions">
          <strong>Guidelines for best results:</strong>
          <ul>
            <li>
              Upload only <strong>one character</strong> — not words or
              sentences.
            </li>
            <li>Use plain white or black background.</li>
            <li>Write clearly with good contrast.</li>
            <li>Center the character in the image.</li>
            <li>Crop tightly around the character.</li>
            <li>Ensure good lighting with no shadows.</li>
          </ul>
        </div>

        {/* Upload Area */}
        <div className="upload-area">
          <input
            type="file"
            id="file-input"
            accept="image/*"
            onChange={handleUpload}
            hidden
          />
          <label htmlFor="file-input" className="upload-label">
            {preview ? (
              <img src={preview} alt="preview" className="preview-image" />
            ) : (
              <div className="upload-placeholder">
                <span className="upload-icon">📤</span>
                <p>Click to upload character image</p>
                <small>PNG, JPG • Max 5MB</small>
              </div>
            )}
          </label>
        </div>

        {/* Error message */}
        {error && <p className="error-message">{error}</p>}

        {/* Buttons */}
        <div className="button-group">
          <button
            className="predict-button"
            onClick={handleSubmit}
            disabled={!image || loading}
          >
            {loading ? " Predicting..." : " Predict Character"}
          </button>

          {preview && (
            <button className="reset-button" onClick={resetAll}>
              Clear
            </button>
          )}
        </div>

        {/* Result — shown for both warning and success */}
        {result && (
          <div className="result-card">
            <h2>Prediction Result</h2>

            {/* Warning banner — only shown when confidence is low */}
            {result.warning && (
              <div className="warning-banner">{result.warning}</div>
            )}

            {/* Main prediction */}
            <div className="main-result">
              <div className="character-box">
                <span className="ranjana-char">{result.roman}</span>
              </div>
              <p className="phonetic">{result.phonetic}</p>
              <p className="type">{result.category}</p>
              <p className="confidence">
                Confidence: <strong>{result.confidence.toFixed(2)}%</strong>
              </p>
              {/* <p className="confidence">
                Ranjana Score:{" "}
                <strong>{result.ranjana_score.toFixed(1)}%</strong>
              </p> */}
            </div>

            {/* Top 5 */}
            <h4>Top 5 Predictions</h4>
            <ul className="top-predictions">
              {result.top_5_predictions.map((pred, index) => (
                <li key={index} className="prediction-item">
                  <span className="rank">{index + 1}.</span>
                  <span className="char">{pred.character}</span>
                  <span className="conf">{pred.confidence.toFixed(2)}%</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
