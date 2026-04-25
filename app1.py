import streamlit as st
import numpy as np
import cv2
from scipy.signal import resample, savgol_filter
from tensorflow.keras.models import load_model
import os

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
MODEL_PATH = 'ecg_model.h5'
CLASSES    = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unknown']
CLASS_META = {
    'Normal':           {'icon': '✅', 'color': '#22c55e'},
    'Supraventricular': {'icon': '⚠️', 'color': '#f59e0b'},
    'Ventricular':      {'icon': '🚨', 'color': '#ef4444'},
    'Fusion':           {'icon': '🔀', 'color': '#a855f7'},
    'Unknown':          {'icon': '❓', 'color': '#6b7280'},
}

# ─────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def get_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"'{MODEL_PATH}' not found next to app.py")
    return load_model(MODEL_PATH)


# ─────────────────────────────────────────────
#  CORE: Grid removal
#  ECG paper has a regular horizontal + vertical grid.
#  We detect those lines morphologically and erase them
#  BEFORE trying to extract the trace — this is the key fix.
# ─────────────────────────────────────────────
def remove_grid_lines(binary_img: np.ndarray) -> np.ndarray:
    """
    Detects horizontal and vertical grid lines via morphological ops
    and removes them, leaving only the ECG trace.
    """
    h, w = binary_img.shape

    # ── Horizontal lines ────────────────────────────────────────────────────
    # A horizontal line spans most of the image width
    h_kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 5, 1))
    h_lines   = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, h_kernel)

    # ── Vertical lines ──────────────────────────────────────────────────────
    v_kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 5))
    v_lines   = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, v_kernel)

    # ── Subtract grid from original ─────────────────────────────────────────
    grid      = cv2.add(h_lines, v_lines)
    cleaned   = cv2.subtract(binary_img, grid)

    # ── Small cleanup pass ──────────────────────────────────────────────────
    k         = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cleaned   = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k)

    return cleaned


# ─────────────────────────────────────────────
#  CORE: Column-wise trace extraction
#  After grid removal, each column should have only
#  the ECG trace pixels. We use the MEDIAN of remaining
#  bright pixels — more robust than min() or mean().
# ─────────────────────────────────────────────
def extract_signal_from_clean(clean_binary: np.ndarray) -> np.ndarray:
    height, width = clean_binary.shape
    signal = []

    for x in range(width):
        col    = clean_binary[:, x]
        pixels = np.where(col > 0)[0]
        if len(pixels) > 0:
            # median of trace pixels = centre of the line stroke
            signal.append(height - int(np.median(pixels)))
        else:
            signal.append(None)   # gap — will be interpolated

    # ── Fill None gaps by linear interpolation ───────────────────────────────
    signal = np.array(signal, dtype=float)
    nans   = np.isnan(signal.astype(float))

    # convert None → nan properly
    signal_f = np.where([v is None for v in signal.tolist()], np.nan, signal)

    # interpolate
    ok = np.isfinite(signal_f)
    if ok.sum() < 10:
        raise ValueError("Couldn't extract a trace — image may be too noisy or wrong format.")

    xp = np.where(ok)[0]
    fp = signal_f[ok]
    signal_interp = np.interp(np.arange(len(signal_f)), xp, fp)

    return signal_interp


# ─────────────────────────────────────────────
#  CORE: Find best single-beat window
#  Instead of just taking ±93 around the global peak,
#  we smooth first so noise spikes don't fool us,
#  find the dominant QRS peak, then window around it.
# ─────────────────────────────────────────────
def extract_beat_window(signal: np.ndarray, window: int = 186) -> np.ndarray:
    # Smooth to find dominant peak (not noise)
    smooth_len = min(21, len(signal) // 10 * 2 + 1)   # must be odd
    smoothed   = savgol_filter(signal, smooth_len, 3)

    peak_idx   = int(np.argmax(smoothed))
    half       = window // 2

    start = max(0, peak_idx - half)
    end   = start + window

    # Clamp to signal length
    if end > len(signal):
        end   = len(signal)
        start = max(0, end - window)

    cropped = signal[start:end]

    # Pad if still short
    if len(cropped) < window:
        cropped = np.pad(cropped, (0, window - len(cropped)), mode='edge')

    return cropped[:window]


# ─────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────
def process_image_to_signal(uploaded_file):
    """
    Full pipeline: image → cleaned binary → signal → 186-pt normalised vector.
    Returns: (model_input (1,186,1), debug_imgs dict, raw_signal array)
    """
    uploaded_file.seek(0)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img        = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Could not decode image. Try JPG or PNG.")

    # ── Grayscale ────────────────────────────────────────────────────────────
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── Adaptive brightness detection (centre sample) ────────────────────────
    h, w   = gray.shape
    sample = gray[h//4 : 3*h//4, w//4 : 3*w//4]
    if np.mean(sample) > 127:
        gray = cv2.bitwise_not(gray)   # dark trace on light paper → invert

    # ── Binarize ─────────────────────────────────────────────────────────────
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ── Grid removal (the key step) ──────────────────────────────────────────
    cleaned = remove_grid_lines(binary)

    # ── Extract signal ───────────────────────────────────────────────────────
    raw_signal = extract_signal_from_clean(cleaned)

    # ── Get 186-pt beat window ───────────────────────────────────────────────
    beat = extract_beat_window(raw_signal, window=186)

    # ── Normalise (0–1) ──────────────────────────────────────────────────────
    lo, hi     = beat.min(), beat.max()
    normalised = (beat - lo) / (hi - lo + 1e-5)

    debug = {
        'original': cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        'binary':   binary,
        'cleaned':  cleaned,
    }

    return normalised.reshape(1, 186, 1), debug, normalised


# ─────────────────────────────────────────────
#  UI
# ─────────────────────────────────────────────
st.set_page_config(page_title="ECG Classifier", page_icon="❤️", layout="centered")

st.markdown("""
<style>
    .result-box { border-radius:12px; padding:20px 24px; border:2px solid; margin:14px 0; }
    .result-title { font-size:1.6rem; font-weight:700; }
    .bar-bg { background:#1e293b; border-radius:6px; height:12px; width:100%;
               overflow:hidden; margin:3px 0 8px; }
    .bar-fill { height:100%; border-radius:6px; }
</style>
""", unsafe_allow_html=True)

st.title("❤️ ECG Arrhythmia Classifier")
st.caption("Upload an ECG image — the pipeline removes grid lines before extracting the signal.")

# ── Model ────────────────────────────────────────────────────────────────────
try:
    model = get_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.stop()

# ── Upload ───────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload ECG Image", type=["jpg","jpeg","png"])

if uploaded_file:

    # ── Process ──────────────────────────────────────────────────────────────
    try:
        input_vector, debug, signal_1d = process_image_to_signal(uploaded_file)
    except ValueError as e:
        st.error(f"Processing failed: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        st.stop()

    # ── Debug view: show what happened at each stage ─────────────────────────
    with st.expander("🔬 Show processing stages (debug)"):
        c1, c2, c3 = st.columns(3)
        c1.image(debug['original'], caption="Original", use_container_width=True)
        c2.image(debug['binary'],   caption="Binarized", use_container_width=True, clamp=True)
        c3.image(debug['cleaned'],  caption="Grid removed", use_container_width=True, clamp=True)

        st.caption("If 'Grid removed' still shows a flat or noisy line, the image quality is too low for reliable digitisation.")

    # ── Signal sanity check ──────────────────────────────────────────────────
    st.subheader("Extracted Signal (186 points)")
    st.line_chart(signal_1d)

    is_dead = (signal_1d.max() - signal_1d.min()) < 0.05
    if is_dead:
        st.error("⚠️ Signal is flat — the trace wasn't extracted correctly. Check the debug view above.")
        st.stop()

    st.divider()

    # ── Predict ──────────────────────────────────────────────────────────────
    with st.spinner("Classifying…"):
        try:
            prediction = model.predict(input_vector, verbose=0)
            result_idx = int(np.argmax(prediction))
            probs      = prediction[0]
            confidence = float(probs[result_idx]) * 100
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    label = CLASSES[result_idx]
    meta  = CLASS_META[label]

    # ── Verdict ──────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class='result-box'
         style='background:{meta["color"]}18; border-color:{meta["color"]}'>
        <div class='result-title' style='color:{meta["color"]}'>
            {meta["icon"]} &nbsp;{label}
        </div>
        <div style='color:#94a3b8; margin-top:6px; font-size:0.9rem'>
            Confidence: <b style='color:{meta["color"]}'>{confidence:.1f}%</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if label != 'Normal':
        st.warning("⚠️ Abnormal activity detected.")
    else:
        st.success("✅ Rhythm appears normal.")

    # ── Per-class probabilities ──────────────────────────────────────────────
    st.subheader("Class Probabilities")
    for i in np.argsort(probs)[::-1]:
        m   = CLASS_META[CLASSES[i]]
        pct = float(probs[i]) * 100
        w   = "font-weight:700;" if i == result_idx else "opacity:0.7;"
        st.markdown(f"""
        <div style='{w} font-size:0.87rem; margin-bottom:2px'>
            {m["icon"]} &nbsp;{CLASSES[i]}
            <span style='color:{m["color"]}'>&nbsp;{pct:.1f}%</span>
        </div>
        <div class='bar-bg'>
            <div class='bar-fill'
                 style='width:{min(pct,100):.1f}%; background:{m["color"]}'></div>
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.caption("⚕️ Academic use only — not a medical device.")