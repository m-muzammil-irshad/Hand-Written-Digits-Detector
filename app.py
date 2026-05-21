import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import os

# ── Load model ───────────────────────────────────────────────────────────────
model = tf.saved_model.load("saved_model")
infer = model.signatures["serving_default"]
OUTPUT_KEY = list(infer.structured_outputs.keys())[0]

DIGIT_NAMES = ["Zero","One","Two","Three","Four","Five","Six","Seven","Eight","Nine"]

# ── Generate sample images on disk (gr.Examples needs file paths) ────────────
# Sample images — MNIST-style hand drawn digits (pre-generated, shipped with repo)
SAMPLE_PATHS = [[f"samples/{d}.png"] for d in ["3","7","1","9","0","5"]]

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(pil_img):
    img_np      = np.array(pil_img.convert("RGB"))
    img_resized = cv2.resize(img_np, (28, 28))
    img_gray    = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    if img_gray.mean() > 127:           # real photo: black digit on white → invert
        img_gray = cv2.bitwise_not(img_gray)
    img_float = img_gray.reshape(28, 28, 1).astype("float32") / 255.0
    return img_float[np.newaxis, ...]

# ── Inference ─────────────────────────────────────────────────────────────────
def predict(image):
    if image is None:
        return empty_html()
    tensor = tf.constant(preprocess(image), dtype=tf.float32)
    probs  = infer(tensor)[OUTPUT_KEY].numpy()[0]
    pred   = int(np.argmax(probs))
    conf   = float(probs[pred]) * 100
    return result_html(pred, conf, probs)

# ── HTML builders ─────────────────────────────────────────────────────────────
def empty_html():
    bars = "".join(f"""
      <div class="bar-row">
        <span class="bar-digit bar-muted">{i}</span>
        <div class="bar-track"><div class="bar-fill bar-empty"></div></div>
        <span class="bar-pct bar-muted">—</span>
      </div>""" for i in range(10))
    return f"""
    <div class="rcard">
      <div class="rlabel">✨ Result</div>
      <div class="empty-box">
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none"
             stroke="#c4b5fd" stroke-width="1.5" stroke-linecap="round">
          <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
        </svg>
        <div class="empty-txt">No prediction yet<br>Upload an image and click Predict</div>
      </div>
      <div class="conf-hdr">Confidence per digit</div>
      {bars}
    </div>"""

def result_html(pred, conf, probs):
    bars = "".join(f"""
      <div class="bar-row">
        <span class="bar-digit {'bar-hl' if i==pred else ''}">{i}</span>
        <div class="bar-track">
          <div class="bar-fill {'bar-top' if i==pred else ''}" style="width:{float(probs[i])*100:.1f}%"></div>
        </div>
        <span class="bar-pct {'bar-hl' if i==pred else ''}">{float(probs[i])*100:.0f}%</span>
      </div>""" for i in range(10))
    return f"""
    <div class="rcard">
      <div class="rlabel">✨ Result</div>
      <div class="digit-box">
        <div class="digit-badge">{pred}</div>
        <div class="digit-meta">
          <div class="digit-sublabel">Predicted digit</div>
          <div class="digit-name">{DIGIT_NAMES[pred]} ({pred})</div>
          <div class="digit-conf">{conf:.2f}% confidence</div>
        </div>
      </div>
      <div class="conf-hdr">Confidence per digit</div>
      {bars}
    </div>"""

# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
/* ── Gradio shell cleanup ── */
.gradio-container { max-width: 880px !important; margin: 0 auto !important; }
footer { display: none !important; }

/* ── App header ── */
#app-header { text-align:center; padding:1.5rem 0 1rem; }
#app-header h1 { font-size:1.5rem; font-weight:700; color:#1e1b4b; margin:0 0 4px; }
#app-header p  { font-size:.875rem; color:#64748b; margin:0; }

/* ── Main row stretch ── */
#main-row { align-items:stretch !important; }
#left-col, #right-col { display:flex !important; flex-direction:column !important; }

/* ── Upload card ── */
#upload-card {
  background:#fff !important;
  border:0.5px solid #e2e0f0 !important;
  border-radius:14px !important;
  padding:1.25rem !important;
  margin-bottom:10px !important;
}

/* ── Examples card ── */
#examples-card {
  background:#fff !important;
  border:0.5px solid #e2e0f0 !important;
  border-radius:14px !important;
  padding:1.25rem !important;
  flex:1 !important;
}

/* ── Right col: no extra wrapper styles ── */
#right-col > .block {
  background:transparent !important;
  border:none !important;
  padding:0 !important;
  flex:1 !important;
}

/* ── Upload zone ── */
#upload-img .wrap {
  border:1.5px dashed #c4b5fd !important;
  border-radius:10px !important;
  background:#faf9ff !important;
  min-height:185px !important;
}
#upload-img { border-radius:10px; overflow:hidden; }

/* ── Predict button ── */
#predict-btn {
  background:#534AB7 !important;
  color:#eeedfe !important;
  border:none !important;
  border-radius:10px !important;
  font-weight:700 !important;
  font-size:.9rem !important;
  width:100% !important;
  margin-top:10px !important;
}
#predict-btn:hover { background:#4338a8 !important; }

/* ── Examples (sample strip) ── */
#examples-row { margin-top:14px; }
#examples-row .label-wrap { display:none !important; }
#examples-row .examples-holder {
  display:grid !important;
  grid-template-columns: repeat(3,1fr) !important;
  gap:8px !important;
  justify-items:center !important;
}
#examples-row .examples-holder button {
  aspect-ratio:1 !important;
  width:100% !important;
  border-radius:10px !important;
  border:0.5px solid #e2e0f0 !important;
  background:#f8f7ff !important;
  padding:0 !important;
  cursor:pointer !important;
  overflow:hidden !important;
  display:flex !important;
  align-items:center !important;
  justify-content:center !important;
}
#examples-row .examples-holder button:hover {
  border-color:#7F77DD !important;
  background:#f0eeff !important;
}
#examples-row .examples-holder button img {
  width:100% !important; height:100% !important;
  object-fit:contain !important;
  border-radius:8px !important;
  display:block !important;
  margin:auto !important;
}
.samples-lbl { font-size:12px; color:#94a3b8; margin:14px 0 6px; display:flex; align-items:center; gap:5px; }

/* ── Right card HTML ── */
.rcard {
  background:#fff;
  border:0.5px solid #e2e0f0;
  border-radius:14px;
  padding:1.25rem;
  height:100%;
  box-sizing:border-box;
}
.rlabel {
  font-size:11px; font-weight:700; color:#94a3b8;
  text-transform:uppercase; letter-spacing:.07em;
  margin-bottom:12px;
}

/* ── Empty state ── */
.empty-box {
  display:flex; flex-direction:column; align-items:center; justify-content:center;
  gap:10px; padding:1.5rem 1rem;
  background:#f8f7ff; border-radius:10px; margin-bottom:14px;
}
.empty-txt { font-size:13px; color:#94a3b8; text-align:center; line-height:1.6; }

/* ── Digit result ── */
.digit-box {
  display:flex; align-items:center; gap:1rem;
  padding:1rem; background:#f8f7ff; border-radius:10px; margin-bottom:14px;
}
.digit-badge {
  width:64px; height:64px; flex-shrink:0;
  border-radius:10px; background:#eeedfe; border:0.5px solid #afa9ec;
  display:flex; align-items:center; justify-content:center;
  font-size:36px; font-weight:700; color:#3c3489; font-family:monospace;
}
.digit-sublabel { font-size:12px; color:#94a3b8; margin-bottom:2px; }
.digit-name  { font-size:20px; font-weight:700; color:#1e1b4b; line-height:1.2; }
.digit-conf  { font-size:13px; color:#64748b; margin-top:2px; }

/* ── Confidence bars ── */
.conf-hdr { font-size:11px; font-weight:700; color:#94a3b8; text-transform:uppercase; letter-spacing:.07em; margin-bottom:10px; }
.bar-row   { display:flex; align-items:center; gap:8px; margin-bottom:7px; }
.bar-digit { font-size:13px; font-weight:500; color:#475569; width:14px; text-align:center; flex-shrink:0; }
.bar-digit.bar-hl    { color:#3c3489; font-weight:800; }
.bar-digit.bar-muted { color:#cbd5e1; }
.bar-track { flex:1; height:8px; background:#f1f0f9; border-radius:99px; overflow:hidden; border:0.5px solid #e2e0f0; }
.bar-fill  { height:100%; border-radius:99px; background:#afa9ec; width:0%; }
.bar-fill.bar-top   { background:#534AB7; }
.bar-fill.bar-empty { background:#e8e6f0; width:100%; opacity:.4; }
.bar-pct  { font-size:12px; color:#94a3b8; width:32px; text-align:right; flex-shrink:0; }
.bar-pct.bar-hl    { color:#534AB7; font-weight:700; }
.bar-pct.bar-muted { color:#cbd5e1; }

/* ── Tips ── */
#tips-bar {
  padding:10px 16px; background:#faf9ff; border-radius:10px;
  border-left:2px solid #afa9ec;
  font-size:12px; color:#64748b; line-height:1.8; margin-bottom:1rem;
}
#tips-bar strong { color:#334155; }

/* ── Footer ── */
#app-footer { text-align:center; font-size:11px; color:#94a3b8; padding-bottom:1rem; }
"""

# ── Build UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(title="Handwritten Digit Detector") as demo:

    gr.HTML(f"<style>{CSS}</style>")

    gr.HTML("""
    <div id="app-header">
      <h1>✍️ Handwritten Digit Detector</h1>
      <p>Upload a digit image — CNN model predicts 0 through 9 with confidence scores</p>
    </div>
    """)

    with gr.Row(elem_id="main-row"):

        # ── Left column ──────────────────────────────────────────────────────
        with gr.Column(scale=1, elem_id="left-col"):

            # Upload card
            with gr.Group(elem_id="upload-card"):
                image_input = gr.Image(
                    type="pil",
                    label="Upload",
                    elem_id="upload-img",
                    height=220,
                    show_label=False,
                )
                predict_btn = gr.Button(
                    "🔍  Predict Digit",
                    elem_id="predict-btn",
                    variant="primary",
                )

            # Examples card
            with gr.Group(elem_id="examples-card"):
                gr.HTML('<div class="samples-lbl">💡 Try an example</div>')
                gr.Examples(
                    examples=SAMPLE_PATHS,
                    inputs=[image_input],
                    elem_id="examples-row",
                    label="",
                )

        # ── Right column ─────────────────────────────────────────────────────
        with gr.Column(scale=1, elem_id="right-col"):
            result_out = gr.HTML(
                value=empty_html(),
                elem_id="result-html",
            )

    gr.HTML("""
    <div id="tips-bar">
      <strong>Tips for best results —</strong>
      write clearly on white paper with dark ink &nbsp;·&nbsp;
      center the digit &nbsp;·&nbsp; single digit only &nbsp;·&nbsp;
      avoid shadows or extra marks
    </div>
    <div id="app-footer">
      Built with TensorFlow &nbsp;·&nbsp; Trained on MNIST &nbsp;·&nbsp; Deployed on 🤗 Hugging Face Spaces
    </div>
    """)

    # ── Events ───────────────────────────────────────────────────────────────
    predict_btn.click(fn=predict, inputs=[image_input], outputs=[result_out])

    # Clear result instantly — JS only, zero server round trip
    image_input.clear(
        fn=lambda: gr.HTML(empty_html()),
        outputs=[result_out],
        queue=False,
    )

if __name__ == "__main__":
    demo.launch()
