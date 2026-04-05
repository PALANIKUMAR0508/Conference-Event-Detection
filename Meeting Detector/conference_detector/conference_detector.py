from __future__ import annotations
import os, cv2, uuid, tempfile, time
import numpy as np
import pandas as pd
import streamlit as st
from collections import defaultdict
from ultralytics import YOLO

MODEL_CONFIG = {
    "chair":  "chair.pt",
    "people": "people.pt",
    "light":  "light.pt",
    "mic":    "mic.pt",
}

CLASS_COLORS = {
    "chair":  (0,   255, 100),   
    "people": (0,   200, 255),   
    "light":  (180, 100, 255),  
    "mic":    (255,  80, 180),  
}

COLOR_STYLES = ["mc-green","mc-cyan","mc-purple","mc-pink","mc-amber","mc-blue","mc-rose"]
ICONS        = {"chair":"🪑","people":"🧑","light":"💡","mic":"🎙️"}

st.set_page_config(
    page_title="Election Assets Audit",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Space+Mono:wght@400;700&display=swap');

html, body, .main, .block-container {
    background:#0a0c10 !important;
    color:#e2e8f0 !important;
    font-family:'Rajdhani',sans-serif !important;
}
.block-container { padding:1.5rem 2rem !important; max-width:1400px !important; }

/* Header */
.app-header {
    display:flex; align-items:center; gap:16px;
    padding:22px 28px;
    background:linear-gradient(135deg,#111827,#1a2035);
    border:1px solid #1e3a5f; border-radius:14px; margin-bottom:24px;
    box-shadow:0 0 40px rgba(0,120,255,0.08);
}
.app-header h1 {
    font-size:26px; font-weight:700; letter-spacing:2px;
    background:linear-gradient(90deg,#38bdf8,#818cf8);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    font-family:'Space Mono',monospace !important;
}
.app-header p { color:#64748b; font-size:12px; letter-spacing:1px; margin-top:4px; }

/* Assets section */
.assets-section {
    background:#111827; border:1px solid #1e293b;
    border-radius:14px; padding:20px 24px; margin-bottom:20px;
}
.assets-heading {
    font-family:'Space Mono',monospace; font-size:13px; font-weight:700;
    letter-spacing:3px; color:#38bdf8; text-transform:uppercase;
    border-left:4px solid #38bdf8; padding-left:12px; margin-bottom:16px;
}
.model-strip {
    display:grid; grid-template-columns:repeat(4,1fr);
    gap:10px;
}
.model-badge {
    background:#0d1117; border:1px solid #1e293b;
    border-radius:10px; padding:12px 14px;
    display:flex; align-items:center; gap:10px;
}
.model-badge.ok   { border-color:#10b981; }
.model-badge.fail { border-color:#ef4444; }
.badge-dot { width:10px; height:10px; border-radius:50%; flex-shrink:0; }
.badge-dot.ok   { background:#10b981; box-shadow:0 0 8px #10b981; }
.badge-dot.fail { background:#ef4444; box-shadow:0 0 8px #ef4444; }
.badge-name { font-family:'Space Mono',monospace; font-size:12px; font-weight:700; color:#e2e8f0; }
.badge-sub  { font-size:10px; color:#475569; font-family:'Space Mono',monospace; }

/* Upload zone — super attractive */
.upload-section {
    background: linear-gradient(145deg, #0d1f42 0%, #122040 40%, #162548 100%);
    border-radius: 20px 20px 0 0;
    padding: 28px 32px 20px;
    margin-bottom: 0;
    position: relative;
    overflow: hidden;
    box-shadow:
        2px 0 0 0 rgba(99,102,241,0.6),
        -2px 0 0 0 rgba(99,102,241,0.6),
        0 -2px 0 0 rgba(99,102,241,0.6);
    border: 2px solid rgba(99,102,241,0.6);
    border-bottom: none;
}
.upload-section::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 4px;
    background: linear-gradient(90deg, #6366f1, #3b82f6, #06b6d4, #10b981, #3b82f6, #6366f1);
    background-size: 300% 100%;
    animation: shimmer 4s linear infinite;
    border-radius: 20px 20px 0 0;
}
@keyframes shimmer { 0%{background-position:0% 0%} 100%{background-position:300% 0%} }

.upload-top-row {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 6px;
}
.upload-heading {
    font-family:'Space Mono',monospace; font-size:18px; font-weight:700;
    letter-spacing:4px; color:#ffffff; text-transform:uppercase; text-shadow:0 0 20px rgba(99,102,241,0.8);
    display:flex; align-items:center; gap:12px;
}
.upload-heading-icon { font-size:26px; filter: drop-shadow(0 0 8px rgba(99,102,241,0.8)); }
.upload-badge {
    background: linear-gradient(135deg,#1d4ed8,#4338ca);
    color:#c7d2fe; font-family:'Space Mono',monospace;
    font-size:10px; letter-spacing:2px; padding:4px 12px;
    border-radius:20px; border:1px solid rgba(99,102,241,0.4);
}
.upload-sub {
    font-size:11px; color:#7aa3d4; font-family:'Space Mono',monospace;
    margin-bottom:20px; letter-spacing:2px; padding-left:4px;
}
.upload-formats {
    display:flex; gap:8px; margin-bottom:20px; flex-wrap:wrap;
}
.fmt-pill {
    background:rgba(99,102,241,0.12); border:1px solid rgba(99,102,241,0.25);
    border-radius:6px; padding:3px 10px;
    font-family:'Space Mono',monospace; font-size:10px;
    color:#818cf8; letter-spacing:1px;
}

/* File uploader — outer wrapper to act as bottom of upload container */
[data-testid="stFileUploader"] {
    background: linear-gradient(145deg, #0d1f42 0%, #122040 100%);
    border: 2px solid rgba(99,102,241,0.6) !important;
    border-top: none !important;
    border-radius: 0 0 20px 20px !important;
    padding: 0 32px 24px !important;
    margin-bottom: 20px !important;
    box-shadow:
        2px 0 0 0 rgba(99,102,241,0.6),
        -2px 0 0 0 rgba(99,102,241,0.6),
        0 2px 0 0 rgba(99,102,241,0.6),
        0 0 80px rgba(59,130,246,0.12),
        0 20px 40px rgba(0,0,0,0.4);
}
/* File uploader dropzone */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(15,28,65,0.7) !important;
    border: 2px dashed rgba(99,102,241,0.5) !important;
    border-radius: 14px !important;
    transition: all 0.3s ease !important;
    padding: 28px !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #818cf8 !important;
    background: rgba(99,102,241,0.1) !important;
    box-shadow: 0 0 30px rgba(99,102,241,0.25), inset 0 0 20px rgba(99,102,241,0.05) !important;
}
/* Dropzone icon/text */
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] span {
    color: #c4d4f0 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}
/* Browse files button */
[data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(135deg,#4f46e5,#6366f1) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono',monospace !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    padding: 8px 22px !important;
    box-shadow: 0 0 20px rgba(99,102,241,0.5) !important;
    transition: all 0.2s !important;
    text-transform: uppercase !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    background: linear-gradient(135deg,#6366f1,#818cf8) !important;
    box-shadow: 0 0 35px rgba(99,102,241,0.7) !important;
    transform: translateY(-2px) !important;
}
.section-title {
    font-family:'Space Mono',monospace; font-size:11px;
    letter-spacing:2px; color:#38bdf8; text-transform:uppercase;
    border-left:3px solid #38bdf8; padding-left:10px; margin:0 0 14px;
}

/* Big summary table */
.summary-section {
    background:linear-gradient(145deg,#0d1225,#111827);
    border:1px solid rgba(99,102,241,0.3);
    border-radius:16px; padding:28px; margin-top:20px;
    box-shadow:0 0 40px rgba(99,102,241,0.07), inset 0 1px 0 rgba(255,255,255,0.04);
}
.result-row {
    display:grid; grid-template-columns:2fr 1fr 1fr 1fr;
    gap:0; border-bottom:1px solid #1e293b; padding:14px 0;
    align-items:center;
}
.result-row.header {
    border-bottom:2px solid #334155;
    padding-bottom:10px; margin-bottom:4px;
}
.result-row.header span {
    font-family:'Space Mono',monospace; font-size:10px;
    color:#475569; text-transform:uppercase; letter-spacing:1.5px;
}
.result-asset { display:flex; align-items:center; gap:10px; }
.result-icon  { font-size:22px; }
.result-name  { font-family:'Space Mono',monospace; font-size:14px; font-weight:700; color:#e2e8f0; }
.result-val   { font-family:'Space Mono',monospace; font-size:20px; font-weight:700; text-align:center; }
.result-val.v-green  { color:#34d399; }
.result-val.v-cyan   { color:#22d3ee; }
.result-val.v-purple { color:#a78bfa; }
.result-val.v-pink   { color:#f472b6; }
.result-val.v-amber  { color:#fbbf24; }
.result-sub { font-size:11px; color:#475569; font-family:'Rajdhani',sans-serif; text-align:center; }

/* Total strip */
.total-strip {
    display:grid; grid-template-columns:repeat(3,1fr);
    gap:14px; margin-top:24px;
}
.total-card {
    background:#0d1117; border:1px solid #1e293b;
    border-radius:12px; padding:18px; text-align:center;
    position:relative; overflow:hidden;
}
.total-card::before { content:''; position:absolute; top:0; left:0; right:0; height:3px; }
.total-card.tc-blue::before   { background:linear-gradient(90deg,#3b82f6,#818cf8); }
.total-card.tc-green::before  { background:linear-gradient(90deg,#10b981,#34d399); }
.total-card.tc-amber::before  { background:linear-gradient(90deg,#f59e0b,#fbbf24); }
.total-label { font-size:10px; color:#475569; font-family:'Space Mono',monospace; letter-spacing:1.5px; text-transform:uppercase; }
.total-value { font-size:32px; font-weight:700; font-family:'Space Mono',monospace; margin:8px 0 4px; }
.tc-blue  .total-value { color:#60a5fa; }
.tc-green .total-value { color:#34d399; }
.tc-amber .total-value { color:#fbbf24; }

/* Metrics */
.metrics-grid {
    display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr));
    gap:12px; margin:20px 0;
}
.metric-card {
    background:#111827; border:1px solid #1e293b;
    border-radius:12px; padding:16px 12px;
    text-align:center; position:relative; overflow:hidden;
}
.metric-card::before { content:''; position:absolute; top:0; left:0; right:0; height:3px; }
.mc-green::before  { background:linear-gradient(90deg,#10b981,#34d399); }
.mc-cyan::before   { background:linear-gradient(90deg,#06b6d4,#22d3ee); }
.mc-purple::before { background:linear-gradient(90deg,#8b5cf6,#a78bfa); }
.mc-pink::before   { background:linear-gradient(90deg,#ec4899,#f472b6); }
.mc-amber::before  { background:linear-gradient(90deg,#f59e0b,#fbbf24); }
.mc-blue::before   { background:linear-gradient(90deg,#3b82f6,#818cf8); }
.mc-rose::before   { background:linear-gradient(90deg,#ef4444,#f87171); }

.metric-label { font-size:10px; color:#475569; text-transform:uppercase; letter-spacing:1.5px; font-family:'Space Mono',monospace; }
.metric-value { font-size:30px; font-weight:700; margin:6px 0 2px; font-family:'Space Mono',monospace; }
.mc-green  .metric-value { color:#34d399; }
.mc-cyan   .metric-value { color:#22d3ee; }
.mc-purple .metric-value { color:#a78bfa; }
.mc-pink   .metric-value { color:#f472b6; }
.mc-amber  .metric-value { color:#fbbf24; }
.mc-blue   .metric-value { color:#60a5fa; }
.mc-rose   .metric-value { color:#f87171; }
.metric-icon { font-size:20px; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #6366f1, #3b82f6) !important;
    background-size: 200% 100% !important;
    color: white !important; border: none !important;
    border-radius: 12px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 14px !important; font-weight: 700 !important;
    letter-spacing: 2px !important; padding: 14px 24px !important;
    box-shadow: 0 0 30px rgba(99,102,241,0.4), 0 4px 15px rgba(0,0,0,0.3) !important;
    transition: all 0.25s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #6366f1, #818cf8, #60a5fa) !important;
    box-shadow: 0 0 50px rgba(99,102,241,0.7), 0 4px 20px rgba(0,0,0,0.4) !important;
    transform: translateY(-2px) !important;
}
.stButton > button:disabled {
    background: #1e293b !important; color: #334155 !important;
    box-shadow: none !important;
}

[data-testid="stFileUploaderDropzone"] {
    background:#0d1117 !important;
    border:2px dashed #1e3a5f !important;
    border-radius:12px !important;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color:#3b82f6 !important; }

.stAlert { border-radius:10px !important; }
video { border-radius:12px; border:1px solid #1e3a5f; }
.stProgress > div > div { background:linear-gradient(90deg,#3b82f6,#6366f1) !important; }

.app-footer {
    text-align:center; padding:20px; color:#334155; font-size:11px;
    font-family:'Space Mono',monospace; letter-spacing:1px;
    margin-top:40px; border-top:1px solid #1e293b;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD ALL MODELS (cached)
# ============================================================
@st.cache_resource(show_spinner=False)
def load_all_models():
    """Load all .pt models once and cache them."""
    loaded  = {}
    failed  = {}
    script_dir = os.path.dirname(os.path.abspath(__file__))

    for cls_name, pt_file in MODEL_CONFIG.items():
        # Look in same folder as script
        full_path = os.path.join(script_dir, pt_file)
        if not os.path.exists(full_path):
            failed[cls_name] = f"File not found: {pt_file}"
            continue
        try:
            loaded[cls_name] = YOLO(full_path)
        except Exception as e:
            failed[cls_name] = str(e)

    return loaded, failed


st.markdown("""
<div class="app-header">
  <div style="font-size:42px">🎥</div>
  <div>
    <h1>ELECTION ASSETS AUDIT</h1>
    <p>MULTI-MODEL DETECTION &nbsp;·&nbsp; CHAIR · PEOPLE · LIGHT · MIC</p>
  </div>
</div>
""", unsafe_allow_html=True)

with st.spinner("Loading models..."):
    models, failed_models = load_all_models()


strip_html = '<div class="assets-section"><div class="assets-heading">📋 List of Assets</div><div class="model-strip">'
all_model_icons = {"chair":"🪑","people":"🧑","light":"💡","mic":"🎙️"}
for cls_name, pt_file in MODEL_CONFIG.items():
    if cls_name in models:
        strip_html += f"""
        <div class="model-badge ok">
          <div class="badge-dot ok"></div>
          <div>
            <div class="badge-name">{all_model_icons.get(cls_name,'')} {cls_name.upper()}</div>
            <div class="badge-sub">✅ {pt_file}</div>
          </div>
        </div>"""
    else:
        strip_html += f"""
        <div class="model-badge fail">
          <div class="badge-dot fail"></div>
          <div>
            <div class="badge-name">{all_model_icons.get(cls_name,'')} {cls_name.upper()}</div>
            <div class="badge-sub">❌ {pt_file} missing</div>
          </div>
        </div>"""
strip_html += "</div></div>"
st.markdown(strip_html, unsafe_allow_html=True)

if failed_models:
    missing_files = [MODEL_CONFIG[k] for k in failed_models]
    st.warning(
        f"⚠️ Missing model files: **{', '.join(missing_files)}**  \n"
        f"Place them in the same folder as `conference_detector.py`"
    )

if not models:
    st.error("❌ No models loaded. Place .pt files in the same folder and restart.")
    st.stop()


st.markdown("""
<div class="upload-section">
  <div class="upload-heading">
    <span class="upload-heading-icon">🎬</span> Upload Conference Video
  </div>
  <div class="upload-sub">MP4 &nbsp;·&nbsp; AVI &nbsp;·&nbsp; MOV &nbsp;·&nbsp; MKV &nbsp;·&nbsp; WEBM</div>
""", unsafe_allow_html=True)

video_file = st.file_uploader(
    "Drag & drop your video here or click Browse Files",
    type=["mp4", "avi", "mov", "mkv", "webm"],
    key="video_uploader",
)
video_tmp = None
if video_file is not None:
    video_tmp = os.path.join(tempfile.gettempdir(), f"vid_{uuid.uuid4().hex}.mp4")
    with open(video_tmp, "wb") as f:
        f.write(video_file.read())

    cap = cv2.VideoCapture(video_tmp)
    tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_val    = cap.get(cv2.CAP_PROP_FPS) or 25
    duration   = tot_frames / fps_val
    W          = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H          = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fname = video_file.name[:22]+"…" if len(video_file.name)>22 else video_file.name
    st.markdown(f"""
    <div style="display:flex;gap:10px;flex-wrap:wrap;margin:10px 0 4px;">
      <span style="background:#111827;border:1px solid #1e293b;border-radius:8px;
        padding:5px 14px;font-family:Space Mono,monospace;font-size:11px;color:#64748b;">
        📁 <span style="color:#94a3b8">{fname}</span>
      </span>
      <span style="background:#111827;border:1px solid #1e293b;border-radius:8px;
        padding:5px 14px;font-family:Space Mono,monospace;font-size:11px;color:#64748b;">
        🎞️ <span style="color:#94a3b8">{tot_frames:,} frames</span>
      </span>
      <span style="background:#111827;border:1px solid #1e293b;border-radius:8px;
        padding:5px 14px;font-family:Space Mono,monospace;font-size:11px;color:#64748b;">
        ⏱️ <span style="color:#94a3b8">{duration:.1f}s</span>
      </span>
      <span style="background:#111827;border:1px solid #1e293b;border-radius:8px;
        padding:5px 14px;font-family:Space Mono,monospace;font-size:11px;color:#64748b;">
        📐 <span style="color:#94a3b8">{W}×{H}</span>
      </span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("🎬 Upload a video to detect chairs, people, lights, and mics.")

# ── Run button (all settings hidden, super fast defaults) ────
conf_threshold = 0.35 
frame_skip     = 2      # Normal — every 2nd frame
resize_on      = True   

run_btn = st.button(
    "🚀  RUN DETECTION",
    use_container_width=True,
    disabled=(video_file is None or len(models) == 0)
)

if video_file is None:
    st.warning("⚠️ Please upload a video to run detection.")

# ============================================================
# DETECTION FUNCTION — runs all 4 models per frame
# ============================================================
def run_multi_model_detection(input_path, model_dict, conf, prog_bar, status_txt,
                              frame_skip=3, do_resize=True):
    """
    Run each model on every Nth frame (frame_skip), combine all bounding boxes,
    annotate and write output video.
    frame_skip=3 means process every 3rd frame → 3x faster
    do_resize=True resizes frame to 640px width before inference → much faster
    """
    cap   = cv2.VideoCapture(input_path)
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS   = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Resize scale factor for inference (keeps original size for output)
    infer_w    = 640
    scale      = infer_w / W if do_resize and W > infer_w else 1.0
    infer_h    = int(H * scale)

    out_path = os.path.join(tempfile.gettempdir(), f"out_{uuid.uuid4().hex}.mp4")
    writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    records         = []
    peak            = defaultdict(int)
    fidx            = 0
    last_frame_cnts = defaultdict(int)  # reuse last result for skipped frames
    last_boxes      = []                # reuse last boxes for skipped frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        should_detect = (fidx % frame_skip == 0)

        if should_detect:
            frame_cnts = defaultdict(int)
            boxes_draw = []  # list of (x1,y1,x2,y2,color,label)

            # Resize frame for faster inference
            if scale < 1.0:
                small = cv2.resize(frame, (infer_w, infer_h))
            else:
                small = frame

            # ── Run each model separately ──
            for cls_name, yolo_model in model_dict.items():
                try:
                    results = yolo_model.predict(small, conf=conf, verbose=False)
                    color   = CLASS_COLORS.get(cls_name, (200, 200, 200))

                    for box in results[0].boxes:
                        conf_val = float(box.conf[0])
                        # Scale boxes back to original size
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1 = int(x1 / scale); y1 = int(y1 / scale)
                        x2 = int(x2 / scale); y2 = int(y2 / scale)

                        frame_cnts[cls_name] += 1
                        label = f"{cls_name} {conf_val:.0%}"
                        boxes_draw.append((x1, y1, x2, y2, color, label))

                except Exception:
                    pass

            last_frame_cnts = frame_cnts
            last_boxes      = boxes_draw
        else:
            # Reuse previous frame's detections
            frame_cnts = last_frame_cnts
            boxes_draw = last_boxes

        # ── Draw boxes on original-size frame ──
        for (x1, y1, x2, y2, color, label) in boxes_draw:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1-th-10), (x1+tw+8, y1), color, -1)
            cv2.putText(frame, label, (x1+4, y1-4),
                        cv2.FONT_HERSHEY_DUPLEX, 0.55, (15,15,15), 1, cv2.LINE_AA)

        # ── Top-left frame overlay ──
        overlay_items = list(frame_cnts.items())
        box_h = 22 * (len(overlay_items) + 1) + 10
        cv2.rectangle(frame, (0, 0), (230, box_h), (10,10,10), -1)
        cv2.putText(frame, f"Frame {fidx+1}/{total}", (8, 18),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, (120,120,120), 1)
        for i, (k, v) in enumerate(sorted(overlay_items)):
            c = CLASS_COLORS.get(k, (200,200,200))
            cv2.putText(frame, f"{k}: {v}", (8, 18 + (i+1)*20),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, c, 1)

        # Peak tracking
        for k, v in frame_cnts.items():
            if v > peak[k]: peak[k] = v

        row = {"frame": fidx + 1}
        row.update(frame_cnts)
        records.append(row)
        writer.write(frame)
        fidx += 1

        if prog_bar and total > 0:
            prog_bar.progress(min(fidx / total, 1.0))
        if status_txt and fidx % 10 == 0:
            pct = int(fidx / total * 100) if total > 0 else 0
            status_txt.markdown(
                f'<p style="font-family:Space Mono,monospace;font-size:12px;color:#64748b">'
                f'⚙️ {fidx}/{total} frames processed ({pct}%) — '
                f'Running {len(model_dict)} models per frame</p>',
                unsafe_allow_html=True
            )

    cap.release()
    writer.release()

    df = pd.DataFrame(records).fillna(0)
    if "frame" in df.columns:
        df["frame"] = df["frame"].astype(int)
    for col in df.columns:
        if col != "frame":
            df[col] = df[col].astype(int)

    return out_path, dict(peak), df

# ── Session state ────────────────────────────────────────────
if "det_results" not in st.session_state:
    st.session_state.det_results = None

if run_btn and video_tmp is not None and models:
    st.session_state.det_results = None
    st.markdown('<div class="section-title" style="margin-top:24px">⚙️ Processing Video</div>', unsafe_allow_html=True)
    prog = st.progress(0.0)
    stxt = st.empty()
    t0   = time.time()

    try:
        out_path, peak, frame_df = run_multi_model_detection(
            video_tmp, models, conf_threshold, prog, stxt,
            frame_skip=frame_skip, do_resize=resize_on
        )
        elapsed = time.time() - t0
        prog.progress(1.0)
        stxt.markdown(
            f'<p style="font-family:Space Mono,monospace;font-size:12px;color:#34d399">'
            f'✅ Detection complete! Processed in {elapsed:.1f}s using {len(models)} models.</p>',
            unsafe_allow_html=True
        )
        st.session_state.det_results = {
            "out_path": out_path,
            "peak":     peak,
            "frame_df": frame_df,
            "elapsed":  elapsed,
        }
    except Exception as e:
        st.error(f"Detection failed: {e}")

# ============================================================
# RESULTS
# ============================================================
if st.session_state.det_results:
    r        = st.session_state.det_results
    out_path = r["out_path"]
    peak     = r["peak"]
    frame_df = r["frame_df"]
    elapsed  = r["elapsed"]

    # ── Total strip (3 big cards) ────────────────────────────
    total_assets = sum(peak.values())
    total_frames_processed = len(frame_df)

    st.markdown(f"""
    <div style="margin-top:28px">
    <div style="font-family:Space Mono,monospace;font-size:14px;font-weight:700;
        letter-spacing:3px;color:#ffffff;text-transform:uppercase;
        border-left:4px solid #6366f1;padding-left:14px;margin-bottom:20px;
        text-shadow:0 0 20px rgba(99,102,241,0.6);">
        📊 Detection Results
    </div>
    <div class="total-strip">
      <div class="total-card tc-blue">
        <div class="total-label">Asset Types Found</div>
        <div class="total-value">{len(peak)}</div>
        <div style="font-size:11px;color:#64748b;font-family:Space Mono,monospace">Chair · People · Light · Mic</div>
      </div>
      <div class="total-card tc-amber">
        <div class="total-label">Process Time</div>
        <div class="total-value">{elapsed:.1f}s</div>
        <div style="font-size:11px;color:#64748b;font-family:Space Mono,monospace">{total_frames_processed} frames analysed</div>
      </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Big Summary Table ────────────────────────────────────
    val_colors = {
        "chair":  "v-green",
        "people": "v-cyan",
        "light":  "v-purple",
        "mic":    "v-pink",
    }

    rows_html = """
    <div class="summary-section">
    <div style="font-family:Space Mono,monospace;font-size:14px;font-weight:700;
        letter-spacing:3px;text-transform:uppercase;margin-bottom:20px;
        display:flex;align-items:center;gap:12px;">
        <span style="background:linear-gradient(135deg,#6366f1,#38bdf8);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            📋 ASSET COUNT SUMMARY
        </span>
    </div>
    <div class="result-row header" style="grid-template-columns:2fr 1fr 1fr;">
      <span style="color:#818cf8;font-weight:700;letter-spacing:2px;">ASSET</span>
      <span style="text-align:center;color:#22d3ee;font-weight:700;letter-spacing:2px;">TOTAL DETECTIONS</span>
      <span style="text-align:center;color:#fbbf24;font-weight:700;letter-spacing:2px;">AVG / FRAME</span>
    </div>"""

    for cls, cnt in sorted(peak.items(), key=lambda x: -x[1]):
        icon      = ICONS.get(cls, "📦")
        vc        = val_colors.get(cls, "v-amber")
        total_det = int(frame_df[cls].sum()) if cls in frame_df.columns else 0
        avg_frm   = round(float(frame_df[cls].mean()), 1) if cls in frame_df.columns else 0.0
        rows_html += f"""
    <div class="result-row" style="grid-template-columns:2fr 1fr 1fr;">
      <div class="result-asset">
        <span class="result-icon">{icon}</span>
        <span class="result-name">{cls.upper()}</span>
      </div>
      <div>
        <div class="result-val {vc}">{total_det:,}</div>
        <div class="result-sub">Across all frames</div>
      </div>
      <div>
        <div class="result-val {vc}">{avg_frm}</div>
        <div class="result-sub">Per frame</div>
      </div>
    </div>"""

    rows_html += "</div>"
    st.markdown(rows_html, unsafe_allow_html=True)

    # ── CSV Download (full width, styled) + Reset ────────────
    summary_df = pd.DataFrame({
        "Asset":            list(peak.keys()),
        "Total Detections": [
            int(frame_df[col].sum()) if col in frame_df.columns else 0
            for col in peak.keys()
        ],
        "Avg Per Frame": [
            round(float(frame_df[col].mean()), 1) if col in frame_df.columns else 0.0
            for col in peak.keys()
        ],
    }).sort_values("Total Detections", ascending=False).reset_index(drop=True)

    csv_bytes = summary_df.to_csv(index=False).encode()

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style="
        background:linear-gradient(135deg,#0c1f3f,#111e3a);
        border:2px solid #2563eb; border-radius:14px;
        padding:24px 28px; margin-bottom:16px;
        box-shadow:0 0 30px rgba(59,130,246,0.15);
        position:relative; overflow:hidden;
    ">
    <div style="position:absolute;top:0;left:0;right:0;height:3px;
        background:linear-gradient(90deg,#10b981,#38bdf8,#818cf8);"></div>
    <div style="font-family:Space Mono,monospace;font-size:13px;font-weight:700;
        letter-spacing:2px;color:#ffffff;margin-bottom:6px;">
        ⬇️ DOWNLOAD ASSET COUNT SUMMARY
    </div>
    <div style="font-family:Space Mono,monospace;font-size:11px;color:#64748b;margin-bottom:0px;">
        CSV format · Chair · People · Light · Mic counts included
    </div>
    </div>
    """, unsafe_allow_html=True)

    dl_col, reset_col = st.columns([3, 1])
    with dl_col:
        st.download_button(
            label="📥  Download Summary CSV",
            data=csv_bytes,
            file_name="asset_count_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with reset_col:
        if st.button("🔄 New Video", use_container_width=True):
            st.session_state.det_results = None
            st.rerun()

# ── Footer ───────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
  ELECTION ASSETS AUDIT &nbsp;·&nbsp; CHAIR · PEOPLE · LIGHT · MIC
</div>
""", unsafe_allow_html=True)
