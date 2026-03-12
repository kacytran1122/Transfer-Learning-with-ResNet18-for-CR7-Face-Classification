from flask import Flask, request, jsonify
from huggingface_hub import hf_hub_download
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

app = Flask(__name__)

d = torch.device("cpu")

# If your training classes were ['other', 'target'], keep 1.
# If they were ['target', 'other'], change this to 0.
cr7_idx = 1

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

m = None
model_ready = False
model_error = ""

def load_model():
    global m, model_ready, model_error

    if m is not None:
        return m

    try:
        path = hf_hub_download(
            repo_id="kacytran1122/cr7",
            filename="cr_resnet18.pth"
        )

        net = models.resnet18(weights=None)
        f = net.fc.in_features
        net.fc = nn.Linear(f, 2)
        net.load_state_dict(torch.load(path, map_location=d))
        net.eval()

        m = net
        model_ready = True
        model_error = ""
        return m

    except Exception as e:
        model_ready = False
        model_error = str(e)
        print("MODEL LOAD ERROR:", e, flush=True)
        raise

@app.route("/")
def home():
    return "CR7 backend is running"

@app.route("/health")
def health():
    return jsonify({
        "ok": True,
        "model_ready": model_ready,
        "model_error": model_error
    })

@app.route("/warmup")
def warmup():
    try:
        load_model()
        return jsonify({
            "ok": True,
            "message": "Model loaded successfully"
        })
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500

@app.route("/predict", methods=["POST"])
def predict():
    global model_error

    try:
        net = load_model()
    except Exception:
        return jsonify({
            "error": f"Model failed to load: {model_error}"
        }), 500

    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        f = request.files["image"]

        if f.filename == "":
            return jsonify({"error": "Please choose an image file"}), 400

        img = Image.open(f.stream).convert("RGB")
        x = tf(img).unsqueeze(0).to(d)

        with torch.no_grad():
            z = net(x)
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
        print("PREDICT ERROR:", e, flush=True)
        return jsonify({"error": str(e)}), 500
