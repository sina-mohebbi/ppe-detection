"""
PPE Detection — Streamlit demo.

Upload an image or a short video and see the model flag PPE (helmet, vest, mask,
gloves, goggles, safety shoes). Runs on the CPU ONNX model, so it works anywhere,
including a free Streamlit Community Cloud instance with no GPU.

Run locally:
    streamlit run demo/app.py
"""

from __future__ import annotations

import tempfile
import urllib.request
from pathlib import Path

import cv2
import streamlit as st
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "models" / "best.onnx"
# Weights are hosted as a GitHub release asset (kept out of the code repo). The
# app fetches them on first run, so it works on a fresh Streamlit Cloud deploy.
MODEL_URL = ("https://github.com/sina-mohebbi/ppe-detection/releases/"
             "download/models-v1/best.onnx")

st.set_page_config(page_title="PPE Detection", page_icon="🦺", layout="wide")


@st.cache_resource(show_spinner="Fetching + loading model (first run downloads it) ...")
def load_model(weights: str) -> YOLO:
    # Download the weights on first run if absent, then cache the loaded model.
    p = Path(weights)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, str(p))
    return YOLO(str(p))


def main() -> None:
    st.title("🦺 PPE Detection")
    st.caption(
        "Detects personal protective equipment — helmet, vest, mask, gloves, "
        "goggles, safety shoes — with a YOLO11 model optimized for CPU inference."
    )

    with st.sidebar:
        st.header("Settings")
        conf = st.slider("Confidence threshold", 0.1, 0.9, 0.35, 0.05)
        imgsz = st.select_slider("Inference size", [416, 512, 640], value=640)
        weights = st.text_input("Weights", str(DEFAULT_WEIGHTS))
        st.markdown("---")
        st.markdown(
            "Tip: smaller inference size = faster on CPU, slightly lower accuracy."
        )

    model = load_model(weights)

    tab_img, tab_vid = st.tabs(["🖼️ Image", "🎬 Video"])

    # ---- Image tab ----
    with tab_img:
        up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if up:
            tmp = Path(tempfile.gettempdir()) / up.name
            tmp.write_bytes(up.getbuffer())
            res = model.predict(str(tmp), imgsz=imgsz, conf=conf,
                                device="cpu", verbose=False)[0]
            c1, c2 = st.columns(2)
            c1.image(str(tmp), caption="Input", use_container_width=True)
            # plot() returns BGR; convert for correct colors in Streamlit.
            c2.image(res.plot()[:, :, ::-1], caption="Detections",
                     use_container_width=True)
            counts = {model.names[int(c)]: 0 for c in res.boxes.cls}
            for c in res.boxes.cls:
                counts[model.names[int(c)]] += 1
            st.success("Detected: " + (", ".join(f"{v}× {k}" for k, v in counts.items())
                                       or "nothing above threshold"))

    # ---- Video tab ----
    with tab_vid:
        upv = st.file_uploader("Upload a short video", type=["mp4", "mov", "avi"])
        skip = st.slider("Frame skip (higher = faster)", 1, 5, 2)
        if upv and st.button("Run detection"):
            src = Path(tempfile.gettempdir()) / upv.name
            src.write_bytes(upv.getbuffer())
            out = Path(tempfile.gettempdir()) / "ppe_out.mp4"

            cap = cv2.VideoCapture(str(src))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps / skip, (w, h))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            bar = st.progress(0.0, text="Processing ...")

            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                idx += 1
                if idx % skip:
                    continue
                res = model.predict(frame, imgsz=imgsz, conf=conf,
                                    device="cpu", verbose=False)[0]
                writer.write(res.plot())
                if total:
                    bar.progress(min(idx / total, 1.0), text=f"Frame {idx}/{total}")
            cap.release()
            writer.release()
            bar.empty()
            st.video(str(out))
            st.success("Done.")


if __name__ == "__main__":
    main()
