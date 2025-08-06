from flask import Flask, render_template, request
import torch
import torchaudio
import os
import numpy as np
import joblib
from model import CNN_LSTM  
import torchaudio.transforms as T

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and label encoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_LSTM(num_classes=6).to(device)
model.load_state_dict(torch.load("models/best_model.pt", map_location=device))
model.eval()
le = joblib.load("models/label_encoder.pkl")

# Audio transforms
mel_spectrogram = T.MelSpectrogram(
    sample_rate=22050,
    n_fft=1024,
    hop_length=512,
    n_mels=64
)
amplitude_to_db = T.AmplitudeToDB()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        audio_file = request.files["audio"]
        file_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
        audio_file.save(file_path)

        # Load and process audio
        waveform, sr = torchaudio.load(file_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 22050:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=22050)
            waveform = resampler(waveform)

        # Convert to Mel Spectrogram
        mel_spec = mel_spectrogram(waveform)
        mel_spec_db = amplitude_to_db(mel_spec)
        mel_spec_db = mel_spec_db.squeeze(0).transpose(0, 1)  # shape: (Time, Mel)

        # Add batch dimension
        input_tensor = mel_spec_db.unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.nn.functional.softmax(logits, dim=-1).squeeze().cpu().numpy()
            predicted_idx = np.argmax(probs)
            predicted_label = le.inverse_transform([predicted_idx])[0]

        # Format output
        emotion_scores = {le.inverse_transform([i])[0]: round(float(probs[i]), 4) for i in range(len(probs))}

        return render_template("index.html", prediction=predicted_label, scores=emotion_scores)

    return render_template("index.html", prediction=None, scores=None)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

