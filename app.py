from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io

app = Flask(__name__)

d = torch.device("cpu")

m = models.resnet18(weights=None)
f = m.fc.in_features
m.fc = nn.Linear(f, 2)
m.load_state_dict(torch.load("cr_resnet18.pth", map_location=d))
m.eval()

# If training classes were ['other', 'target'], CR7 is index 1
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

    file = request.files["image"]
    img = Image.open(file.stream).convert("RGB")
    x = tf(img).unsqueeze(0).to(d)

    with torch.no_grad():
        z = m(x)
        pr = torch.softmax(z, dim=1)[0].cpu()

    p_cr7 = float(pr[cr7_idx].item())
    p_not = 1.0 - p_cr7

    if p_cr7 >= 0.5:
        pred = "CR7"
    else:
        pred = "Not CR7"

    return jsonify({
        "prediction": pred,
        "p_cr7": round(p_cr7, 4),
        "p_not": round(p_not, 4)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
