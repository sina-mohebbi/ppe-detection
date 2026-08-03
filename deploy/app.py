"""
PPE Detection — Streamlit demo (Hugging Face Spaces version).

Upload an image or a short video and see the model flag PPE (helmet, vest, mask,
gloves, goggles, safety shoes). Runs on the CPU ONNX model, so it works on a free
CPU Space with no GPU.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import streamlit as st
from ultralytics import YOLO

# On a Space the layout is flat: app.py + models/best.onnx sit together.
HERE = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = HERE / "models" / "best.onnx"

st.set_page_config(page_title="PPE Detection", page_icon="🦺", layout="wide")


@st.cache_resource(show_spinner="Loading model ...")
def load_model(weights: str) -> YOLO:
    return YOLO(weights)


def main() -> None:
    st.title("🦺 PPE Detection")
    st.caption(
        "Detects personal protective equipment — helmet, vest, mask, gloves, "
        "goggles, safety shoes — with a YOLO11 model optimized for CPU inference. "
        "Trained on ~12k images; ~0.81 mAP@0.5 on held-out test data."
    )

    with st.sidebar:
        st.header("Settings")
        conf = st.slider("Confidence threshold", 0.1, 0.9, 0.35, 0.05)
        imgsz = st.select_slider("Inference size", [416, 512, 640, 960], value=640)
        st.markdown("---")
        st.markdown(
            "Higher inference size helps small/distant objects (helmets far away) "
            "but is slower on CPU."
        )
        st.markdown("Model runs on CPU via ONNX — no GPU required.")

    weights = str(DEFAULT_WEIGHTS)
    if not Path(weights).exists():
        st.error(f"Model not found at `{weights}`. Upload models/best.onnx to the Space.")
        st.stop()

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
            c2.image(res.plot()[:, :, ::-1], caption="Detections",
                     use_container_width=True)
            counts: dict[str, int] = {}
            for c in res.boxes.cls:
                name = model.names[int(c)]
                counts[name] = counts.get(name, 0) + 1
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
