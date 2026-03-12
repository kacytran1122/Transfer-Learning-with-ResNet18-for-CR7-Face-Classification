from flask import Flask, render_template, request, jsonify
import os
from huggingface_hub import hf_hub_download
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

app = Flask(__name__)

d = torch.device("cpu")
MODEL_NAME = "cr_resnet18.pth"
HF_REPO = "kacytran1122/cr7"

def load_model():
    if not os.path.exists(MODEL_NAME):
        hf_hub_download(
            repo_id=HF_REPO,
            filename=MODEL_NAME,
            local_dir=".",
            local_dir_use_symlinks=False
        )

    m = models.resnet18(weights=None)
    f = m.fc.in_features
    m.fc = nn.Linear(f, 2)
    m.load_state_dict(torch.load(MODEL_NAME, map_location=d))
    m.eval()
    return m

m = load_model()

# If training classes were ['other', 'target'], keep 1.
# If they were ['target', 'other'], change this to 0.
cr7_idx = 1

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    f = request.files["image"]

    try:
        img = Image.open(f.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    x = tf(img).unsqueeze(0).to(d)

    with torch.no_grad():
        z = m(x)
        pr = torch.softmax(z, dim=1)[0].cpu()

    p_cr7 = float(pr[cr7_idx].item())
    p_not = 1.0 - p_cr7
    pred = "CR7" if p_cr7 >= 0.5 else "Not CR7"

    return jsonify({
        "prediction": pred,
        "p_cr7": round(p_cr7, 4),
        "p_not": round(p_not, 4)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
