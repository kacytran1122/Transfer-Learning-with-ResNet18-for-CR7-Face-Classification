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

# If training classes were ['other', 'target'], keep 1.
# If training classes were ['target', 'other'], change to 0.
cr7_idx = 1

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

m = None
MODEL_ERR = ""
MODEL_READY = False

def load_model_once():
    global m, MODEL_ERR, MODEL_READY

    if MODEL_READY and m is not None:
        return m

    try:
        model_path = hf_hub_download(
            repo_id=HF_REPO,
            filename=MODEL_NAME
        )

        mdl = models.resnet18(weights=None)
        f = mdl.fc.in_features
        mdl.fc = nn.Linear(f, 2)
        mdl.load_state_dict(torch.load(model_path, map_location=d))
        mdl.eval()

        m = mdl
        MODEL_READY = True
        MODEL_ERR = ""
        return m

    except Exception as e:
        MODEL_READY = False
        MODEL_ERR = str(e)
        print("MODEL LOAD ERROR:", e)
        raise

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health")
def health():
    global MODEL_READY, MODEL_ERR
    return jsonify({
        "ok": True,
        "model_ready": MODEL_READY,
        "model_error": MODEL_ERR
    })

@app.route("/warmup")
def warmup():
    try:
        load_model_once()
        return jsonify({"ok": True, "message": "Model loaded"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    global MODEL_READY, MODEL_ERR

    try:
        mdl = load_model_once()
    except Exception:
        return jsonify({"error": f"Model failed to load: {MODEL_ERR}"}), 500

    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        f = request.files["image"]
        if f.filename == "":
            return jsonify({"error": "Please choose an image file"}), 400

        img = Image.open(f.stream).convert("RGB")
        x = tf(img).unsqueeze(0).to(d)

        with torch.no_grad():
            z = mdl(x)
            pr = torch.softmax(z, dim=1)[0].cpu()

        p_cr7 = float(pr[cr7_idx].item())
        p_not = 1.0 - p_cr7
        pred = "CR7" if p_cr7 >= 0.5 else "Not CR7"

        return jsonify({
            "prediction": pred,
            "p_cr7": round(p_cr7, 4),
            "p_not": round(p_not, 4)
        })

    except Exception as e:
        print("PREDICT ERROR:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
