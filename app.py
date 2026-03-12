import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

st.set_page_config(
    page_title="CR7 Classifier",
    layout="centered"
)

st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #f8fbff 0%, #eef4ff 45%, #fdf2f8 100%);
}
.block-container {
    max-width: 850px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.title-wrap {
    text-align: center;
    padding: 1rem 0 0.5rem 0;
}
.main-title {
    font-size: 2.6rem;
    font-weight: 800;
    color: #1f2937;
    margin-bottom: 0.25rem;
}
.sub-title {
    font-size: 1.05rem;
    color: #4b5563;
    margin-bottom: 1.2rem;
}
.card {
    background: rgba(255,255,255,0.9);
    border-radius: 22px;
    padding: 1.4rem;
    box-shadow: 0 10px 30px rgba(31,41,55,0.10);
    margin-bottom: 1.2rem;
}
.good {
    padding: 0.45rem 0.9rem;
    border-radius: 999px;
    background: #dcfce7;
    color: #166534;
    font-weight: 700;
    display: inline-block;
}
.bad {
    padding: 0.45rem 0.9rem;
    border-radius: 999px;
    background: #fee2e2;
    color: #991b1b;
    font-weight: 700;
    display: inline-block;
}
.small {
    color: #6b7280;
    font-size: 0.95rem;
}
</style>
""", unsafe_allow_html=True)

d = torch.device("cpu")

m = models.resnet18(weights=None)
f = m.fc.in_features
m.fc = nn.Linear(f, 2)
m.load_state_dict(torch.load("cr_resnet18.pth", map_location=d))
m.eval()

cr7_idx = 1

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

st.markdown("""
<div class="title-wrap">
    <div class="main-title">CR7 or Not?</div>
    <div class="sub-title">
        Upload one face image and the model will decide whether it is Cristiano Ronaldo.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
up = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
st.markdown('</div>', unsafe_allow_html=True)

if up is not None:
    img = Image.open(up).convert("RGB")
    x = tf(img).unsqueeze(0).to(d)

    with torch.no_grad():
        z = m(x)
        pr = torch.softmax(z, dim=1)[0].cpu()

    p_cr7 = float(pr[cr7_idx].item())
    p_not = 1.0 - p_cr7

    th = 0.5

    c1, c2 = st.columns([1.1, 0.9])

    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(img, caption="Uploaded image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Result")

        if p_cr7 >= th:
            st.markdown('<span class="good">CR7</span>', unsafe_allow_html=True)
            st.success("The model predicts this image is Cristiano Ronaldo.")
        else:
            st.markdown('<span class="bad">Not CR7</span>', unsafe_allow_html=True)
            st.error("The model predicts this image is not Cristiano Ronaldo.")

        st.write(f"**CR7 probability:** `{p_cr7:.4f}`")
        st.progress(min(max(p_cr7, 0.0), 1.0))

        st.write(f"**Not CR7 probability:** `{p_not:.4f}`")
        st.progress(min(max(p_not, 0.0), 1.0))

        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="card">
        <h3 style="margin-top:0;">Binary Classification Task (2 classes)</h3>
        <div class="small">
            This app does only one thing:
            <ul>
                <li><b>CR7</b></li>
                <li><b>Not CR7</b></li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
