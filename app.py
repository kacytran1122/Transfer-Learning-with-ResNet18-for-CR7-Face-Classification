from flask import Flask, jsonify
import os
import urllib.request

app = Flask(__name__)

URL = "https://huggingface.co/kacytran1122/cr7/resolve/main/cr_resnet18.pth"
FN = "cr_resnet18.pth"

@app.route("/")
def home():
    return "backend up"

@app.route("/test-download")
def test_download():
    try:
        if not os.path.exists(FN):
            urllib.request.urlretrieve(URL, FN)
        sz = os.path.getsize(FN)
        return jsonify({"ok": True, "size_bytes": sz})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
