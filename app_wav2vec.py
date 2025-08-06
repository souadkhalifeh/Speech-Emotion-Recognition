from flask import Flask, render_template, request
import torch
import torchaudio
import os
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import joblib
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-er")
model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-er").to(device)
label_map = {0: "angry", 1: "happy", 2: "neutral", 3: "sad"}



@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        audio_file = request.files["audio"]
        file_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
        audio_file.save(file_path)

        # Load audio
        waveform, sr = torchaudio.load(file_path)

        # Convert stereo to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if not 16kHz
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)

        # Remove channel dimension
        waveform = waveform.squeeze()

        # Convert to numpy before passing to processor
        inputs = feature_extractor(waveform.numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)

        # Inference
        with torch.no_grad():
            logits = model(input_values).logits
            probs = torch.nn.functional.softmax(logits, dim=-1).squeeze().cpu().numpy()
            predicted_idx = np.argmax(probs)
            predicted_label = label_map[predicted_idx]

        # Format emotion scores
        emotion_scores = {label_map[i]: round(float(probs[i]), 4) for i in range(len(label_map))}


        return render_template("index.html", prediction=predicted_label, scores=emotion_scores)

    return render_template("index.html", prediction=None, scores=None)

if __name__ == "__main__":
    app.run(debug=True)