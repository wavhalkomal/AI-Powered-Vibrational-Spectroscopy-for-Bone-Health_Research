# ##########################  app_k.py  ###################################################################################
# #########################################################################################################################

# ======================= START: Import Libraries ==============================================================================

# Streamlit ML Studio — sticky top nav, parallax hero background, NO fixed top tables
# pip install streamlit scikit-learn xgboost shap pandas numpy plotly
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import streamlit as st  # Import Streamlit for building the web app UI
import streamlit as st
# Store the original button exactly once (in module attr, not in session)
if not hasattr(st, "_original_button"):
    st._original_button = st.button

def _auto_button(label, *args, **kwargs):
    """
    Your wrapped version of st.button. Add any custom behavior here.
    This wrapper will never fail if session_state is empty.
    """
    return st._original_button(label, *args, **kwargs)

# Apply monkey-patch once
st.button = _auto_button

import pandas as pd  # Import Pandas for data handling
import numpy as np  # Import NumPy for numerical operations
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
import plotly.express as px  # Import Plotly Express for interactive visualizations
import plotly.graph_objects as go  # Import Plotly Graph Objects for more customized plots
from sklearn.decomposition import PCA  # Import PCA for dimensionality reduction
from sklearn.cross_decomposition import PLSRegression  # Import PLSRegression for regression analysis with collinear data
from sklearn.model_selection import train_test_split  # Import train_test_split for splitting datasets
from sklearn.svm import SVC  # Import Support Vector Classifier for classification
from sklearn.linear_model import LogisticRegression  # Import Logistic Regression for classification
from sklearn.neighbors import KNeighborsClassifier  # Import KNN for classification
from sklearn.ensemble import RandomForestClassifier  # Import Random Forest for classification
from xgboost import XGBClassifier  # Import XGBoost for classification
from sklearn.metrics import classification_report, confusion_matrix  # Import metrics for model evaluation
import seaborn as sns  # Import Seaborn for additional plotting capabilities
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
import base64
import io
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# Optional (safe guards if not installed)
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

import plotly.express as px
import plotly.graph_objects as go

# =======================END: Import Libraries ==============================================================================

# -------------------- Page config --------------------•
st.set_page_config(
    page_title="AI-Powered Vibrational Spectroscopy for Bone Health Analysis ",
    page_icon="🧬",
    layout="wide"
)

theme = "Light"


# --- Method info + selector (sidebar) ---

# --- Keep method panel state once (can be anywhere before the buttons) ---
if "method_panel_active" not in st.session_state:
    st.session_state.method_panel_active = False
if "method_choice_sidebar" not in st.session_state:
    st.session_state.method_choice_sidebar = "SVM"

def _open_method_panel(method: str):
    st.session_state.method_choice_sidebar = method
    st.session_state.method_panel_active = True

# ======================START: Image paths (local file paths or URLs)==============================================================================

# ✅ Set these to your desired images. Local files will be embedded as base64; URLs used directly.
HERO_IMAGE_PATH = "assets/hero.jpg"   # e.g., "assets/hero.jpg" or a full URL
PAPER_IMAGE_PATH = "assets/paper.png" # e.g., "assets/paper.png" or a full URL
# -------------------- Image paths (local file paths or URLs) --------------------
# Use your local DNA image (downloaded and saved as Background_1.jpg in your project folder)
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
HERO_DEFAULT = str(BASE_DIR / "Background_1.jpg")  # Resolve relative to app.py
PAPER_IMAGE_PATH = "assets/paper.png" # optional page wallpaper

# Fallback URLs if local files are missing
HERO_FALLBACK_URL = "https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?q=80&w=1400&auto=format&fit=crop"
PAPER_FALLBACK_URL = "https://images.unsplash.com/photo-1524820824462-64a7f3c6f42b?q=80&w=1200&auto=format&fit=crop"

# ======================END: Image paths (local file paths or URLs)==============================================================================



def path_to_css_url(path: str, fallback_url: str) -> str:
    """Return a URL usable in CSS. If `path` is http(s), return it.
    Otherwise try to read a local file and embed as data URI. On failure, return fallback_url."""
    import os, base64
    if isinstance(path, str) and path.lower().startswith(("http://", "https://")):
        return path
    try:
        with open(path, "rb") as f:
            data = f.read()
        ext = os.path.splitext(path)[1].lower()
        if ext in (".jpg", ".jpeg"):
            mime = "jpeg"
        elif ext == ".png":
            mime = "png"
        elif ext == ".webp":
            mime = "webp"
        else:
            mime = "octet-stream"
        b64 = base64.b64encode(data).decode("utf-8")
        return f"data:image/{mime};base64,{b64}"
    except Exception:
        return fallback_url

# Resolve hero + paper URLs (local or fallback)
# Locate the uploaded hero asset even when GitHub stores it in an assets folder
# or the filename uses different capitalization. Preserve the original fallback.
def _find_bonespec_hero():
    import pathlib
    root = pathlib.Path(__file__).resolve().parent
    preferred = [root / "Background_1.jpg", root / "assets" / "Background_1.jpg",
                 root / "assets" / "hero.jpg"]
    for candidate in preferred:
        if candidate.is_file():
            return str(candidate)
    for folder in (root, root / "assets", root / "images"):
        if folder.is_dir():
            for candidate in folder.iterdir():
                if candidate.is_file() and candidate.name.casefold() in (
                    "background_1.jpg", "background_1.jpeg", "background_1.png",
                    "background_1.webp", "hero.jpg", "hero.png"):
                    return str(candidate)
    return None

_hero_asset = _find_bonespec_hero()

# User-selected remote DNA background: use directly rather than requiring a GitHub image upload.
hero_url = "https://png.pngtree.com/thumb_back/fw800/background/20260620/pngtree-abstract-blue-digital-dna-strand-medical-technology-background-image_21942756.webp"
paper_url = path_to_css_url(PAPER_IMAGE_PATH, PAPER_FALLBACK_URL)

# -------------------- Accent map --------------------
import streamlit as st
import pandas as pd

def coerce_wavenumber_headers(cols):
    """
    Convert numeric-like column headers to whole-number strings (e.g., '3997.030239' -> '3997').
    Non-numeric headers remain as-is. Duplicates get suffixes (_1, _2, ...).
    """
    new_cols = []
    seen = {}
    for c in cols:
        original = str(c)
        try:
            candidate = str(int(round(float(original))))  # round to integer wavenumber
        except Exception:
            candidate = original

        if candidate in seen:
            seen[candidate] += 1
            candidate = f"{candidate}_{seen[candidate]}"
        else:
            seen[candidate] = 0

        new_cols.append(candidate)
    return new_cols


# -------------------- Shell (Navbar + Hero + FAB) — NO TOP DOCK --------------------
def inject_shell():
    st.markdown(
        f"""
<style>
    .stApp {{ background-color: white; }}

  :root {{ 
    --txt: {"#e5e7eb" if theme=="Dark" else "#0f172a"};
    --subtxt: {"#cbd5e1" if theme=="Dark" else "#334155"};
    --bg: {"#0b1220" if theme=="Dark" else "#ffffff"};
    --card: {"#0f172a" if theme=="Dark" else "#ffffff"};
    --muted: {"#0b1220" if theme=="Dark" else "#f8fafc"};
    --border: {"#172033" if theme=="Dark" else "#e6e9f2"};
    --shadow: {"rgba(0,0,0,.65)" if theme=="Dark" else "rgba(0,0,0,.06)"};
  }}
  html {{ scroll-behavior: smooth; }}
  .main {{ padding-top: 0 !important; }}
  body {{
    color: var(--txt);
    background:
      radial-gradient(1200px 600px at -10% -10%, rgba(99,102,241,.20), transparent 60%),
      radial-gradient(1200px 600px at 110% -10%, rgba(20,184,166,.15), transparent 60%),
      url('{paper_url}') repeat fixed;
    background-size: cover;
  }}
  .block-container {{ max-width: 1200px; }}

  /* Sticky navbar */
  .sticky-nav {{
    position: fixed; top: 0; left: 0; right: 0; z-index: 1000;
    backdrop-filter: saturate(140%) blur(10px);
    background: linear-gradient(180deg, rgba(255,255,255,{0.08 if theme=='Dark' else 0.8}), rgba(255,255,255,{0.04 if theme=='Dark' else 0.65}));
    border: 1px solid var(--border);
    box-shadow: 0 6px 20px var(--shadow);
  }}
  .nav-wrap {{
    max-width: 1200px; margin: 0 auto; padding: 10px 16px;
    display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
  }}
  .brand {{ font-weight: 800; letter-spacing:.3px; margin-right: 8px; opacity:.9; }}
  .nav-btn {{
    display:inline-block; padding:10px 14px; border-radius:14px;
    border: 1px solid var(--border); text-decoration:none; font-weight:700; color:var(--txt);
    background: linear-gradient(180deg, var(--card), {('#0b1220' if theme=='Dark' else '#f6f7fb')});
    box-shadow: 0 2px 0 rgba(0,0,0,.04), inset 0 0 0 1px rgba(255,255,255,.08);
    transition: transform .06s ease, border-color .2s ease, box-shadow .2s ease;
  }}
  .nav-btn:hover {{ transform: translateY(-1px); border-color: var(--accent); box-shadow: 0 8px 26px rgba(0,0,0,.08); }}

  .top-spacer {{ height: 2px; }}

  .block-container {{ 
    max-width: 100% !important;
    padding-left: 5 !important;
    padding-right: 5 !important;
  }}
  
    .hero {{
    position: relative;
    height: 420px;                /* keep your chosen height */
    width: 98%;                   /* 90% of page width */
    margin: 0 auto;               /* center horizontally */
    border-radius: 25px;          /* round border all around */
    overflow: hidden;
    border: 1px solid var(--border);
    box-shadow: 0 20px 60px var(--shadow);
    background-image: linear-gradient(
                        180deg,
                        rgba(0,0,0,{0.45 if theme=='Dark' else 0.15}),
                        rgba(0,0,0,{0.35 if theme=='Dark' else 0.10})
                      ),
                      url('{hero_url}');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    display: grid;
    place-items: center;
  }}
 
  .hero h1 {{ color: white; text-shadow: 0 10px 30px rgba(0,0,0,.40); font-size: 56px; margin: 0; }}
  .hero p {{ color: #eef2ff; margin-top: 10px; font-weight: 600; letter-spacing:.2px; }}

  /* Floating Back-to-Top */
  .fab {{
    position: fixed; right: 18px; bottom: 18px; z-index: 998;
    width: 48px; height: 48px; border-radius: 50%; display: grid; place-items: center;
    background: var(--accent); color: white; font-size: 20px; font-weight: 800; text-decoration:none;
    box-shadow: 0 16px 40px rgba(0,0,0,.25); transition: transform .15s ease;
  }}
  .fab:hover {{ transform: translateY(-2px) scale(1.02); }}
 
  /* Floating Back-to-Top */
  .fab {{
    position: fixed; right: 18px; bottom: 18px; z-index: 998;
    width: 48px; height: 48px; border-radius: 50%; 
    display: grid; place-items: center;
    background: var(--accent); color: white; 
    font-size: 20px; font-weight: 800; text-decoration:none;
    box-shadow: 0 16px 40px rgba(0,0,0,.25); 
    transition: transform .15s ease;
  }}
  .fab:hover {{ 
    transform: translateY(-2px) scale(1.02); 
  }}
 
  /* Cards */
  .card {{
    border: 1px solid var(--border); border-radius: 16px; padding: 18px 16px;
    background: var(--card); box-shadow: 0 10px 30px var(--shadow); margin-bottom: 18px;
  }}
  .badge {{
    display:inline-block; padding: 4px 10px; border-radius: 999px;
    background: var(--accent); color: white; font-weight:700; font-size:12px; letter-spacing:.3px; margin-bottom:10px;
  }}

  /* Anchor spacing (for sticky nav) */
  h2, h3, [data-anchor] {{ scroll-margin-top: 100px; }}
     scroll-behavior: smooth;
</style>
  
<div id="top" class="sticky-nav">
  <div class="nav-wrap">
    <span class="brand">🤖 ML Studio</span>
    <a class="nav-btn" href="#data">Data</a>
    <a class="nav-btn" href="#preprocess">Preprocess</a>
    <a class="nav-btn" href="#train">Train</a>
    <a class="nav-btn" href="#compare">Compare</a>
    <a class="nav-btn" href="#explain">Explain</a>
    <a class="nav-btn" href="#predict">Predict</a>
  </div>
</div>
 
<div class="hero">
  <div style="text-align:center">
    <h1 id="hero-title">🧬 AI-Powered Vibrational Spectroscopy <br>for Bone Health</h1>
    <br><br> 
    <p>• 🔬Bone Spectral Classification and Composition Analysis Workspace</p>
  </div>
</div>
 
""",

        unsafe_allow_html=True,
    )


# <a class="fab" href="#hero-title">↑</a>

# Inject the shell UI (no top dock tables)
inject_shell()

# -------------------- Sections --------------------
def section(title: str, anchor: str):
    st.markdown(f"<a id='{anchor}'></a>", unsafe_allow_html=True)
    st.markdown(f"<div class='card'><span class='badge'>{title.split(':')[0]}</span><h3>{title}</h3>", unsafe_allow_html=True)
    return st.container()



# ===== Dedicated Method Panel (separate from main project UI) =====
if st.session_state.get("method_panel_active", False):
    _method = st.session_state.get("method_choice_sidebar", "SVM")
    st.markdown("<a id='method-panel'></a>", unsafe_allow_html=True)
    # Create two columns on the same line
    col1, col2 = st.columns([6, 1])  # adjust ratio for spacing

    with col1:
        st.markdown(
            f"<h3 style='margin:0'>{_method}</h3>",
            unsafe_allow_html=True
        )

    with col2:
        if st.button("Back", use_container_width=True):
            st.session_state.method_panel_active = False
            st.rerun()

    if _method == "SVM":
        st.info("**SVM** — Support Vector Machine. "
                "")

    elif _method == "PCA":
        st.info("**PCA** — Principal Component Analysis ")

    else:  # PLS-DA
        st.info("**PLS-DA** — Partial Least Squares Discriminant Analysis")

    # Optional placeholders you can remove later
    st.markdown("##### 🔎 Quick Notes")
    st.write("- Keep this panel for method-specific outputs, comparisons, or exports.")

    st.markdown("</div>", unsafe_allow_html=True)

    # 🔒 IMPORTANT: stop rendering the rest of the Analysis page while panel is open
    st.stop()

############################################# [ Browse the file ] #############################################################################


############################################# [ Browse the file ] #############################################################################

# ======================= MAIN PAGE ==========================
if "page" not in st.session_state:
    st.session_state.page = "home"

if st.session_state.page == "home":
    # ⬇️ Center the button using columns
    col_left, col_center, col_right = st.columns([4, 4, 1])
    with col_center:

        st.markdown("<div style='margin-top:50px;'>", unsafe_allow_html=True)

        start_click = st.button("Start Analysis")

    if start_click:
        st.session_state.page = "analysis"
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.page == "know_more":

    if st.button("⬅️ Back to Analysis"):
        st.session_state.page = "analysis"
        st.rerun()

# ======================= ANALYSIS PAGE ==========================
elif st.session_state.page == "analysis":

    # ===== Sticky in-page tabs (no changes to your sections required) =====
    import streamlit.components.v1 as components

    ####################################### [ Upload the document ] #######################################

    uploaded_file = st.file_uploader(f"**Upload FTIR Excel file**", type=["xlsx"])  # File uploader for FTIR data


    if uploaded_file:

            ####################################### [ # Sidebar header for controls] #######################################

            st.sidebar.header("🔧 Controls")

            ####################################### [ # Sidebar for main information] #######################################

            with st.sidebar:

                import streamlit.components.v1 as components

                components.html("""
                <style>
                  .sb-jumps { display:flex; flex-direction:column; gap:8px; }
                  .sb-btn {
                    border:1px solid #e5e7eb; border-radius:10px; padding:8px 10px;
                    font-size:14px; background:#f8fafc; cursor:pointer; text-align:left;
                    transition:transform .12s ease, background .12s ease;
                  }
                  .sb-btn:hover{ background:#eef2ff; transform:translateY(-1px) }
                  
                    scroll-behavior: smooth;
                </style>

                <div class="sb-jumps">
                  <button class="sb-btn" id="go-methods">📘 Overview of Machine Learning Methods</button>
                  <button class="sb-btn" id="go-shap">🔎 SHAP Analysis</button>
                  <button class="sb-btn" id="go-models">📋 Model Results with Detailed Metrics</button>
                  <button class="sb-btn" id="go-pca">🔬 PCA: Principal Component Analysis</button>
                  <button class="sb-btn" id="go-pls">📊 PLS: Partial Least Squares Regression</button>
                  <button class="sb-btn" id="go-plsda">📊 PLS-DA: Partial Least Squares Discriminant Analysis</button> 
                  <button class="sb-btn" id="go-bands">🧪 Molecular Interpretation from Spectral Bands</button>
                </div>

                <script>
                  const TARGETS = {
                    "go-methods": "📘 Overview of Machine Learning Methods",
                    "go-shap": "🔎 SHAP Analysis",
                    "go-models": "📋 Model Results with Detailed Metrics",
                    "go-pca": "🔬 PCA: Principal Component Analysis",
                    "go-pls": "📊 PLS: Partial Least Squares Regression",
                    "go-plsda": "📊 PLS-DA: Partial Least Squares Discriminant Analysis",
                    "go-bands": "🧪 Molecular Interpretation from Spectral Bands"
                  };

                  function findHeaderByText(doc, text){
                    const hs = doc.querySelectorAll('h1,h2,h3,h4,h5,h6,summary');
                    for(const h of hs){
                      if(h.innerText.trim().startsWith(text)) return h;
                    }
                    return null;
                  }

                  function smoothScrollTo(text){
                    try {
                      const el = findHeaderByText(window.parent.document, text) || findHeaderByText(document, text);
                      if(el){ el.scrollIntoView({behavior:'smooth', block:'start'}); }
                    } catch(e){
                      const el = findHeaderByText(document, text);
                      if(el){ el.scrollIntoView({behavior:'smooth', block:'start'}); }
                    }
                  }

                  for(const id in TARGETS){
                    const btn = document.getElementById(id);
                    if(btn){ btn.addEventListener('click', ()=> smoothScrollTo(TARGETS[id])); }
                  }
                </script>
                """, height=300)

            st.subheader("📘 Overview of Machine Learning Methods")
            # === “Know More About Methods” (tabs) — show after file upload ===
            with st.expander("📚 Know More", expanded=False):

                # st.title("📚 Know More About Machine Learning Methods")
                know_tab = st.tabs(["SVM", "PCA", "PLS-DA"])

                with know_tab[0]:
                    st.subheader("🧠 Support Vector Machine (SVM)")
                    st.markdown("""
                    - **What it is:** Supervised classifier that finds a decision boundary with the **widest margin**.
                    - **Why it’s good for spectra:** Handles **high-dimensional** FTIR features; robust with the **kernel trick** (RBF).
                    - **Tune:** `C` (regularization), `gamma` (RBF width), `kernel` (linear/RBF/poly).
                    - **Tips:** Scale features; try **Stratified** CV; watch for overfitting with large `gamma`.
                    """)

                with know_tab[1]:
                    st.subheader("📊 Principal Component Analysis (PCA)")
                    st.markdown("""
                    - **What it is:** Unsupervised **dimensionality reduction** to components (PCs) with max variance.
                    - **Why it’s useful:** Visualizing separability, denoising, and **feature compression**.
                    - **Outputs:** Scores, loadings, explained variance; inspect **PC1–PC3** first.
                    """)

                with know_tab[2]:
                    st.subheader("📈 Partial Least Squares Discriminant Analysis (PLS-DA)")
                    st.markdown("""
                    - **What it is:** Supervised projection maximizing covariance between **X (spectra)** and **Y (labels)**.
                    - **Why it’s useful:** Combines dimensionality reduction + classification for spectral workflows.
                    - **Tune:** Number of LVs; validate with **RMSEC/RMSECV**, **R²**, and permutation tests if possible.
                    """)

            ############################################### [upload file ] ######################################################################

            xls = pd.ExcelFile(uploaded_file)  # Load Excel file

            # ✅ Dynamically list available sheet names
            sheet_name = st.selectbox("📄 Select a sheet to analyze:", xls.sheet_names, index=0)

            df = xls.parse(sheet_name)  # Parse selected sheet
            df.columns = df.columns.astype(str)  # Ensure column names are strings

            # ✅ Clean column headers (convert float-like to whole numbers)
            df.columns = coerce_wavenumber_headers(df.columns.astype(str))

            # ✅ Drop rows with missing values (if you want this cleaning step)
            df = df.dropna()  # Remove rows with missing values

#################################################### [Sidebar ] #####################################################

            # selected_range = st.sidebar.slider("Select Wavenumber Range (cm⁻¹)", 400, 8500, (800, 1200),  step=1)  # Slider for wavenumber range
            # ================= Wavenumber Range (two-way sync; no duplicate value setting) =================
            WN_MIN_BOUND, WN_MAX_BOUND = 400, 8500
            DEFAULT_MIN, DEFAULT_MAX = 800, 1200

            # 1) Initialize once (defaults live in session_state)
            if "wn_range" not in st.session_state:
                st.session_state.wn_range = (DEFAULT_MIN, DEFAULT_MAX)
            if "wn_min" not in st.session_state:
                st.session_state.wn_min = DEFAULT_MIN
            if "wn_max" not in st.session_state:
                st.session_state.wn_max = DEFAULT_MAX


            def _clamp(v, lo=WN_MIN_BOUND, hi=WN_MAX_BOUND):
                return max(lo, min(hi, int(v)))


            def _on_numbers_change():
                # When user types numbers, push into slider state
                mn = _clamp(st.session_state.wn_min)
                mx = _clamp(st.session_state.wn_max)
                if mn >= mx:
                    mx = min(WN_MAX_BOUND, mn + 1)
                st.session_state.wn_min = mn
                st.session_state.wn_max = mx
                st.session_state.wn_range = (mn, mx)


            def _on_slider_change():
                # When user drags slider, push into number inputs
                mn, mx = st.session_state.wn_range
                mn = _clamp(mn);
                mx = _clamp(mx)
                if mn >= mx:
                    mx = min(WN_MAX_BOUND, mn + 1)
                st.session_state.wn_range = (mn, mx)
                st.session_state.wn_min = mn
                st.session_state.wn_max = mx


            st.sidebar.markdown("### ⚙️ Select Wavenumber Range (cm⁻¹)")

            # 2) Number inputs (NO `value` since we use keys)
            c1, c2 = st.sidebar.columns(2)
            with c1:
                st.number_input("Min", min_value=WN_MIN_BOUND, max_value=WN_MAX_BOUND,
                                step=1, key="wn_min", on_change=_on_numbers_change)
            with c2:
                st.number_input("Max", min_value=WN_MIN_BOUND, max_value=WN_MAX_BOUND,
                                step=1, key="wn_max", on_change=_on_numbers_change)

            # 3) Slider (NO `value` since we use key)
            st.sidebar.slider("Adjust Range",
                              min_value=WN_MIN_BOUND, max_value=WN_MAX_BOUND,
                              step=1, key="wn_range", on_change=_on_slider_change)

            # 4) Use this everywhere below
            selected_range = st.session_state.wn_range
            st.sidebar.caption(f"Selected: {selected_range[0]} – {selected_range[1]} cm⁻¹")

            ###################################### [ Disply the data ] #########################################################

            st.subheader("🔍 Uploaded Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            label_column = df.columns[0]  # First column as label column
            data_columns = df.columns[1:]  # Remaining columns

            # ✅ Extract numeric wavenumber values (first part before '/') for filtering
            wavenumber_map = {}
            for col in data_columns:
                try:
                    numeric_value = float(
                        col.split("/")[0].strip())  # Extract first number (e.g., "1030" from "1030/1670")
                    wavenumber_map[col] = numeric_value
                except ValueError:
                    st.warning(f"⚠ Skipping column: {col} (cannot extract numeric value)")

            # ✅ Filter columns based on selected range using extracted numeric values
            selected_columns = [
                col for col, wn in wavenumber_map.items()
                if selected_range[0] <= wn <= selected_range[1]
            ]

            # ✅ Features (retain original column names) and labels
            X = df[selected_columns].astype(float)
            y = df[label_column]

            # ✅ Encode labels dynamically (works for multi-class)
            from sklearn.preprocessing import LabelEncoder

            label_enc = LabelEncoder()
            y_encoded = label_enc.fit_transform(y)

            ################################################################################################################
            ################################################################################################################


            ################################################################################################################
            ################################################################################################################

            ###################################### [ Display the selected data ] #############################

            st.subheader("📌 Data Selected for Analysis")
            st.markdown(f"**Selected Wavenumber Range:** {selected_range[0]} cm⁻¹ to {selected_range[1]} cm⁻¹")
            st.dataframe(X.head(10), use_container_width=True)

            ################################################################################################################

            ##### random sampling
            # ✅ Show class counts and enable random sampling BEFORE train-test split
            st.sidebar.markdown("### 🎯 Class Sampling")
            class_counts = df[label_column].value_counts()
            st.sidebar.write("**Class Distribution:**")
            st.sidebar.write(class_counts)

            # Initialize sample counts in session_state if not present
            if "sample_counts" not in st.session_state:
                st.session_state.sample_counts = {cls: int(class_counts[cls]) for cls in class_counts.index}

            # --- Monkeypatch st.button ONLY for the two SHAP buttons in your Final Script
            if "_original_button" not in st.session_state:
                st.session_state._original_button = st.button

            import streamlit as st

            # --- 1) Auto-click only the SHAP plot buttons (works for both Non-Quant & Quant labels) ---
            if "_original_button" not in st.session_state:
                st.session_state._original_button = st.button  # keep original

            def _auto_button(label, *args, **kwargs):
                auto_labels = {
                    "Generate Beeswarm Plot (Non-Quant, All Rows)",
                    "Generate Waterfall Plot (Non-Quant, All Rows)",
                    "Generate Beeswarm Plot",  # Quant block uses these bare labels
                    "Generate Waterfall Plot",  # Quant block uses these bare labels
                }
                if label in auto_labels:
                    return True
                return st.session_state._original_button(label, *args, **kwargs)

            #####################################################################################################################

            st.button = _auto_button

            # Custom small button style
            # Add custom style for the small reset button
            st.markdown("""
            <style>
            /* Force a very small Reset Sample Selection button */
            div.stButton.reset-sample {
                display: inline-block;       /* Make it inline and shrink wrap */
                width: auto !important;      /* Override default width */
                margin: 0 !important;        /* Remove spacing around it */
            }

            div.stButton.reset-sample > button {
                background-color: #9C2007;
                color: white;
                font-size: 12px !important;    /* Extremely small font */
                font-weight: bold;

                border: none;
                cursor: pointer;
                box-shadow: 0px 0px 2px rgba(0, 0, 0, 0.3);
                height: 1px !important;      /* Tiny height */
                min-width: 1px !important;   /* Force narrow width */
                line-height: 1px !important; /* Prevent extra height */
            }

            div.stButton.reset-sample > button:hover {
                background-color: #7a0000;
                transform: scale(1.05);
            }
          scroll-behavior: smooth;
          
        </style> 
                """, unsafe_allow_html=True )

            ################################################################################################################
            #######################################  [ OIM-WT Result ]  ####################################################

            # # Reset button
            # if st.sidebar.button("🔄 Reset"):
            #     st.session_state.sample_counts = {cls: int(class_counts[cls]) for cls in class_counts.index}
            #
            # # Input: number of samples per class
            # sample_counts = {}
            # for cls in class_counts.index:
            #     max_val = int(class_counts[cls])
            #
            #     # ✅ Ensure stored value does not exceed max
            #     if st.session_state.sample_counts.get(cls, max_val) > max_val:
            #         st.session_state.sample_counts[cls] = max_val
            #
            #     # Number input with clamped default
            #     sample_counts[cls] = st.sidebar.number_input(
            #         f"Select samples for '{cls}' (max {max_val})",
            #         min_value=1,
            #         max_value=max_val,
            #         value=int(st.session_state.sample_counts[cls]),  # safe default
            #         step=1,
            #         key=f"sample_{cls}"
            #     )
            #
            #     st.session_state.sample_counts[cls] = sample_counts[cls]


            ################################################################################################################
            ################################################################################################################

            # --- right after computing class_counts ---
            # class_counts = df[label_column].value_counts()

            # Ensure the session dict exists
            if "sample_counts" not in st.session_state:
                st.session_state.sample_counts = {}

            # 1) Re-sync keys with current classes (add missing; drop stale)
            current_classes = list(class_counts.index)
            # add any new classes with full count default
            for cls in current_classes:
                st.session_state.sample_counts.setdefault(cls, int(class_counts[cls]))
            # drop any classes that aren't present anymore
            for stale in list(st.session_state.sample_counts.keys()):
                if stale not in current_classes:
                    del st.session_state.sample_counts[stale]

            # Optional: a HARD reset button that also clears saved choices
            if st.sidebar.button("🔄 Reset"):
                st.session_state.sample_counts = {cls: int(class_counts[cls]) for cls in current_classes}

            # 2) Inputs (safe defaults + clamping)
            sample_counts = {}
            for cls in current_classes:
                max_val = int(class_counts[cls])

                # clamp any stored value to valid range
                stored = int(st.session_state.sample_counts.get(cls, max_val))
                if stored > max_val:
                    stored = max_val
                if stored < 1:
                    stored = 1

                sample_counts[cls] = st.sidebar.number_input(
                    f"Select samples for '{cls}' (max {max_val})",
                    min_value=1,
                    max_value=max_val,
                    value=stored,  # ✅ SAFE DEFAULT
                    step=1,
                    key=f"sample_{cls}"
                )
                st.session_state.sample_counts[cls] = int(sample_counts[cls])

            ################################################################################################################
            ################################################################################################################


            # Random sampling
            df_sampled = pd.concat([
                df[df[label_column] == cls].sample(n=int(sample_counts[cls]), random_state=42)
                for cls in class_counts.index
            ])

            # Update X and y
            X = df_sampled[selected_columns].astype(float)
            y = df_sampled[label_column]
            from sklearn.preprocessing import LabelEncoder

            label_enc = LabelEncoder()
            y_encoded = label_enc.fit_transform(y)

            # Show sampled counts
            st.sidebar.write("✅ **Sampled Class Counts:**")
            st.sidebar.write(df_sampled[label_column].value_counts())

            ################################################################################################################

            ###################################### [ Train-Test Split Slider] ######################################
            # ✅ NEW: Train-Test Split Slider
            split_ratio = st.sidebar.slider(
                "Select Train-Test Split Ratio (Test Size %)",
                min_value=10, max_value=50, value=20, step=5,
                help="Choose percentage of data for testing (default is 20%)"
            )

            st.sidebar.markdown(f"**Training: {100 - split_ratio}% | Testing: {split_ratio}%**")

            model_options = ["SVM", "Logistic Regression", "KNN", "Random Forest", "XGBoost"]  # List of model options
            selected_models = st.sidebar.multiselect("Select Classification Models", model_options, default=["SVM"])  # Multiselect for models

            ##show_histogram = st.sidebar.checkbox("Show Histogram of Raw Features")  # Option to show raw feature histograms

            run_all = st.sidebar.checkbox("Run All Models")  # Checkbox to run all models
            show_models = st.sidebar.checkbox("Show Model Results", value=True)  # Option to show model results
            show_pca = st.sidebar.checkbox("Show PCA Visualizations", value=True)  # Option to show PCA
            show_pls = st.sidebar.checkbox("Show PLS Analysis", value=True)  # Option to show PLS
            show_plsda = st.sidebar.checkbox("Show PLSDA Analysis", value=True)  # Option to show PLSDA

            st.sidebar.header("📊 Spectral Plots Display")  # Sidebar header for spectral plots

            show_mean = st.sidebar.checkbox("Mean of FTIR Spectra", value=False)  # Checkbox to show mean spectra
            show_stddev = st.sidebar.checkbox("Standard Deviation of FTIR Spectra",value=False)  # Checkbox to show standard deviation
            show_mean_stddev = st.sidebar.checkbox("Mean + Standard Deviation of FTIR Spectra", value=False)  # Checkbox for mean+std
            show_minimum = st.sidebar.checkbox("Minimum of FTIR Spectra", value=False)  # Checkbox for minimum spectra
            show_maximum = st.sidebar.checkbox("Maximum of FTIR Spectra", value=False)  # Checkbox for maximum spectra

            #####################################################################################################
            # ====================== Spectrum Plot with Bond Annotations ======================
            show_spectrum = st.sidebar.checkbox("📈 Show Raw Spectrum with Bond Annotations", value=False)

            if show_spectrum:
                st.subheader("📈 FTIR Spectrum with Bond Annotations")

                # --- Plot raw spectra for all samples (X is already filtered by range) ---
                fig, ax = plt.subplots(figsize=(10, 6))

                # Plot each sample
                for idx, row in X.iterrows():
                    ax.plot(X.columns.astype(float), row.values, alpha=0.4, lw=1)

                # Labels
                ax.set_xlabel("Wavenumber (cm⁻¹)")
                ax.set_ylabel("Absorbance")
                ax.set_title("FTIR Spectrum with Bond Annotations")

                # ✅ Bond Annotation Mapping Table
                bond_annotations_table = [
                    (8500, "O–H Stretching and Bending (Water)"),
                    (7000, "O–H Stretching (Water)"),
                    (6688, "N–H Stretching (Protein/Collagen)"),
                    (5800, "CH₂ Stretching (Lipid)"),
                    (5200, "O–H Stretching and Bending (Water)"),
                    (4890, "N–H Bending (Protein/Collagen)"),
                    (4610, "C–H Stretching & Deformation (Protein/Collagen)"),
                    (4310, "Sugar Ring Vibrations (Proteoglycan)"),
                    ((3600, 3200), "O–H Stretching (Water/Hydroxyl)"),
                    ((3500, 3300), "N–H Stretching (Proteins)"),
                    ((3000, 2800), "C–H Stretching (Lipids, CH₂)"),
                    ((1750, 1650), "C=O Stretching (Proteins/Lipids)"),
                    (1550, "Amide II (Proteins)"),
                    (1338, "CH₂ Side Chain Bending (Collagen)"),
                    ((1100, 900), "PO₄³⁻ Stretching (Bone Mineral)"),
                    ((890, 850), "CO₃²⁻ Bending (Carbonate)"),
                    (1740, "C=O Stretching (Ester, Lipids)"),
                    (1650, "Amide I (Proteins)"),
                    (1630, "Water O–H Bending (Water)"),
                    ((1200, 1000), "C–O Stretching (Alcohols/Ethers)"),
                    (1115, "HPO₄²⁻ Stretching (Bone mineral)"),
                    (1060, "Sugar Ring C–O Stretch (Carbohydrates)"),
                    (1030, "PO₄³⁻ Stretching (Bone Mineral)"),
                    (875, "CO₃²⁻ Bending (Carbonates)"),
                    (856, "C–S Bending (Proteoglycans)"),
                    ((900, 800), "Aromatic C–H Bending (Fingerprint region)")
                ]

                # Add vertical lines + labels for annotations
                for entry in bond_annotations_table:
                    if isinstance(entry[0], tuple):  # range
                        wn_min, wn_max = entry[0]
                        ax.axvspan(wn_min, wn_max, color="red", alpha=0.08)
                        ax.text((wn_min + wn_max) / 2, ax.get_ylim()[1] * 0.9, entry[1],
                                rotation=90, fontsize=8, ha="center", va="top", color="red")
                    else:  # single wavenumber
                        wn = entry[0]
                        ax.axvline(wn, color="red", linestyle="--", alpha=0.6)
                        ax.text(wn, ax.get_ylim()[1] * 0.95, entry[1],
                                rotation=90, fontsize=8, ha="center", va="top", color="red")

                st.pyplot(fig)

            #####################################################################################################

            ###################################### [ SHAP Models ] ###############################################
            # st.subheader("🔎 SHAP Analysis")
            #
            # with st.expander("🔎 SHAP Analysis of FTIR Ratios", expanded=False):
            #
            #     ##### Model for Quant peak ratios and raw, 2nd derivative and peak intensity ratio data
            #     #####################################################################################################
            #
            #     ################# this is for ratio dataset
            #     if "Quant peak ratios" in sheet_name:
            #         # Load dataset
            #         df = pd.read_excel(uploaded_file, sheet_name="Quant peak ratios")
            #
            #         import pandas as pd
            #         import shap
            #         import matplotlib.pyplot as plt
            #         import numpy as np
            #         from xgboost import XGBClassifier
            #         from sklearn.model_selection import train_test_split
            #         from sklearn.preprocessing import LabelEncoder
            #
            #         # Encode target variable
            #         X = df.drop(columns=['Type'])  # Features
            #         y = df['Type']  # Target
            #         label_enc = LabelEncoder()
            #         y_encoded = label_enc.fit_transform(y)
            #
            #         # Train-test split
            #         X_train, X_test, y_train, y_test = train_test_split(
            #             X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
            #         )
            #
            #         # Train XGBoost model
            #         model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            #         model.fit(X_train, y_train)
            #
            #         # SHAP Explainer
            #         explainer = shap.TreeExplainer(model)
            #         shap_values = explainer.shap_values(X_test)
            #
            #         # Initialize session state to store plots
            #         if "beeswarm_plot" not in st.session_state:
            #             st.session_state.beeswarm_plot = None
            #         if "waterfall_plot" not in st.session_state:
            #             st.session_state.waterfall_plot = None
            #
            #         # with st.expander("🔎 SHAP Analysis of Quant Peak Ratios", expanded=True):
            #
            #         # Create two columns inside expander for layout control
            #         col1, col2 = st.columns([1, 1])
            #
            #         # ========================
            #         # 1️⃣ SHAP Beeswarm Plot
            #         # ========================
            #         with col1:
            #             st.subheader("📌 SHAP Beeswarm Plot")
            #             if st.button("Generate Beeswarm Plot"):
            #                 fig_beeswarm = plt.figure(figsize=(6, 4))
            #                 shap.summary_plot(shap_values, X_test, feature_names=X.columns, plot_type="dot",
            #                                   show=False)
            #                 st.session_state.beeswarm_plot = fig_beeswarm
            #
            #             # Display beeswarm plot if generated
            #             if st.session_state.beeswarm_plot is not None:
            #                 st.pyplot(st.session_state.beeswarm_plot)
            #
            #         # ========================
            #         # 2️⃣ Feature Importance Table
            #         # ========================
            #         with col2:
            #             st.subheader("📊 Feature Importance Table (Mean Absolute SHAP)")
            #             mean_abs_shap = np.abs(shap_values).mean(axis=0)
            #             feature_importance = pd.DataFrame({
            #                 "Feature": X.columns,
            #                 "Mean |SHAP Value|": mean_abs_shap,
            #             }).sort_values(by="Mean |SHAP Value|", ascending=False).reset_index(drop=True)
            #
            #             st.dataframe(feature_importance.style.format({"Mean |SHAP Value|": "{:.4f}"}),
            #                          use_container_width=True)
            #
            #             csv = feature_importance.to_csv(index=False).encode('utf-8')
            #             st.download_button("📥 Download Feature Importance CSV", data=csv,
            #                                file_name="shap_feature_importance.csv", mime="text/csv")
            #
            #         st.markdown("---")
            #
            #         # ========================
            #         # 3️⃣ SHAP Waterfall Plot (Individual Sample)
            #         # ========================
            #         col_w1, col_w2 = st.columns([1, 1])
            #
            #         with col_w1:
            #             st.subheader("🎚 Select Sample Index")
            #             sample_idx = st.slider("Select Sample Index for Waterfall Plot", 0, len(X_test) - 1, 0)
            #
            #         with col_w2:
            #             st.subheader("💧 SHAP Waterfall Plot (Individual Sample)")
            #             if st.button("Generate Waterfall Plot"):
            #                 explanation = shap.Explanation(
            #                     values=shap_values[sample_idx],
            #                     base_values=explainer.expected_value,
            #                     data=X_test.iloc[sample_idx],
            #                     feature_names=X.columns
            #                 )
            #                 fig_waterfall = plt.figure()
            #                 shap.plots.waterfall(explanation, show=False)
            #                 st.session_state.waterfall_plot = plt.gcf()
            #
            #             # Display waterfall plot if generated
            #             if st.session_state.waterfall_plot is not None:
            #                 st.pyplot(st.session_state.waterfall_plot)
            #
            #     else:
            #
            #         # ===================== Non-Quant Data Modeling =====================
            #
            #         if sheet_name != "Quant peak ratios":
            #
            #             import pandas as pd
            #             import shap
            #             import matplotlib.pyplot as plt
            #             import numpy as np
            #             from xgboost import XGBClassifier
            #             from sklearn.model_selection import train_test_split
            #             from sklearn.preprocessing import LabelEncoder
            #
            #             # Load the selected non-quant sheet
            #             df_nq = pd.read_excel(uploaded_file, sheet_name=sheet_name)
            #
            #             # --- Target column (mirror Quant flow; default 'Type' if present)
            #             if "Type" in df_nq.columns:
            #                 label_col_nq = "Type"
            #             else:
            #                 non_numeric_cols_nq = [c for c in df_nq.columns if
            #                                        not pd.api.types.is_numeric_dtype(df_nq[c])]
            #                 if not non_numeric_cols_nq:
            #                     st.error(
            #                         "No suitable label/target column found (e.g., 'Type'). Please add a categorical target.")
            #                     st.stop()
            #                 label_col_nq = st.selectbox("Select label/target column", non_numeric_cols_nq, index=0)
            #
            #             # --- Features/Target (numeric features only; exclude label)
            #             X_nq = df_nq.drop(columns=[label_col_nq]).select_dtypes(include=[np.number]).astype(float)
            #             y_nq = df_nq[label_col_nq].astype(str)
            #
            #             if X_nq.shape[1] == 0:
            #                 st.error("No numeric feature columns available in this sheet.")
            #                 st.stop()
            #
            #             # Ensure feature names are strings (e.g., wavenumbers can be ints)
            #             feat_names_nq = [str(c) for c in X_nq.columns]
            #
            #             # --- Encode target
            #             label_enc_nq = LabelEncoder()
            #             y_encoded_nq = label_enc_nq.fit_transform(y_nq)
            #
            #             # Mirror your "ALL rows" approach for SHAP tables/plots
            #             X_all = X_nq
            #             y_encoded_all = y_encoded_nq
            #             feat_names_all = feat_names_nq
            #
            #             # --- Train-test split (as in Quant)
            #             stratify_vec_all = y_encoded_all if len(np.unique(y_encoded_all)) > 1 else None
            #             try:
            #                 X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
            #                     X_all, y_encoded_all, test_size=0.2, stratify=stratify_vec_all, random_state=42
            #                 )
            #             except ValueError:
            #                 # Fallback for tiny/single-class cases
            #                 X_train_all, X_test_all, y_train_all, y_test_all = X_all, X_all, y_encoded_all, y_encoded_all
            #                 st.info("Using all rows for training and explanation (insufficient data to stratify/split).")
            #
            #             # --- Train XGBoost model (same as Quant)
            #             model_all = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            #             model_all.fit(X_train_all, y_train_all)
            #
            #             # --- SHAP Explainer (compute SHAP on ALL rows for the table and plots)
            #             explainer_all = shap.TreeExplainer(model_all)
            #             shap_values_raw_all = explainer_all.shap_values(X_all)
            #
            #             # Handle multi-class output; keep UI simple/consistent (pick last class if list)
            #             if isinstance(shap_values_raw_all, list):
            #                 shap_values_all = shap_values_raw_all[-1]
            #                 expected_value_all = explainer_all.expected_value[-1]
            #             else:
            #                 shap_values_all = shap_values_raw_all
            #                 expected_value_all = explainer_all.expected_value
            #
            #             # --- Session state for plots (separate from Quant keys)
            #             if "beeswarm_plot_nq_all" not in st.session_state:
            #                 st.session_state.beeswarm_plot_nq_all = None
            #             if "waterfall_plot_nq_all" not in st.session_state:
            #                 st.session_state.waterfall_plot_nq_all = None
            #
            #             # Create two columns (mirror Quant layout)
            #             col1_all, col2_all = st.columns([1, 1])
            #
            #             # ========================
            #             # 1️⃣ SHAP Beeswarm Plot
            #             # ========================
            #             with col1_all:
            #                 st.subheader("📌 SHAP Beeswarm Plot (All Rows)")
            #                 if st.button("Generate Beeswarm Plot (Non-Quant, All Rows)"):
            #                     fig_beeswarm_all = plt.figure(figsize=(6, 4))
            #                     shap.summary_plot(
            #                         shap_values_all, X_all,
            #                         feature_names=feat_names_all,
            #                         plot_type="dot", show=False
            #                     )
            #                     st.session_state.beeswarm_plot_nq_all = fig_beeswarm_all
            #
            #                 if st.session_state.beeswarm_plot_nq_all is not None:
            #                     st.pyplot(st.session_state.beeswarm_plot_nq_all)
            #
            #             ################################################################################################
            #             ################################  [ OIM-WT Working Script ]  ################################################################
            #
            #             # ========================
            #             # 2️⃣ Feature Importance Table (ALL ROWS)
            #             # ========================
            #             with col2_all:
            #                 st.subheader("📊 Feature Importance Table (Mean Absolute SHAP) – All Rows")
            #                 mean_abs_shap_all = np.abs(shap_values_all).mean(axis=0)
            #                 feature_importance_all = pd.DataFrame({
            #                     "Feature": feat_names_all,
            #                     "Mean |SHAP Value|": mean_abs_shap_all,
            #                 }).sort_values(by="Mean |SHAP Value|", ascending=False).reset_index(drop=True)
            #
            #                 st.dataframe(
            #                     feature_importance_all.style.format({"Mean |SHAP Value|": "{:.4f}"}),
            #                     use_container_width=True
            #                 )
            #
            #                 csv_all = feature_importance_all.to_csv(index=False).encode('utf-8')
            #                 st.download_button(
            #                     "📥 Download Feature Importance CSV (All Rows, Non-Quant)",
            #                     data=csv_all,
            #                     file_name=f"shap_feature_importance_{sheet_name}_ALL_ROWS.csv",
            #                     mime="text/csv"
            #                 )
            #
            #             ################################################################################################
            #             ################################## [ General Script: Working Script ] ##############################################################
            #
            #             ################################################################################################
            #             ################################################################################################
            #             st.markdown("---")
            #
            #             # ========================
            #             # 3️⃣ SHAP Waterfall Plot (Individual Sample, from ALL ROWS)
            #             # ========================
            #             col_w1_all, col_w2_all = st.columns([1, 1])
            #
            #             with col_w1_all:
            #                 st.subheader("🎚 Select Sample Index (All Rows)")
            #                 if len(X_all) > 0:
            #                     sample_idx_all = st.slider(
            #                         "Select Sample Index for Waterfall Plot (Non-Quant, All Rows)",
            #                         0, len(X_all) - 1, 0
            #                     )
            #                 else:
            #                     sample_idx_all = 0
            #                     st.info("No rows available to plot.")
            #
            #             with col_w2_all:
            #                 st.subheader("💧 SHAP Waterfall Plot (Individual Sample)")
            #                 if st.button("Generate Waterfall Plot (Non-Quant, All Rows)"):
            #                     explanation_all = shap.Explanation(
            #                         values=shap_values_all[sample_idx_all],
            #                         base_values=expected_value_all,
            #                         data=X_all.iloc[sample_idx_all].to_numpy(),  # NumPy row to avoid dtype/index issues
            #                         feature_names=feat_names_all  # ensure string names
            #                     )
            #                     fig_waterfall_all = plt.figure()
            #                     shap.plots.waterfall(explanation_all, show=False)
            #                     st.session_state.waterfall_plot_nq_all = plt.gcf()
            #
            #                 if st.session_state.waterfall_plot_nq_all is not None:
            #                     st.pyplot(st.session_state.waterfall_plot_nq_all)

            ###########################################  [END: SHAP Analysis ]  ##########################################################
            #####################################################################################################


            ###########################################  [New Script -START: SHAP Analysis ]  ##########################################################
            #####################################################################################################
            #
            # st.subheader("🔎 SHAP Analysis")
            #
            # with st.expander("🔎 SHAP Analysis of FTIR Ratios", expanded=False):
            #
            #     ##### Model for Quant peak ratios and raw, 2nd derivative and peak intensity ratio data
            #     #####################################################################################################
            #
            #     # ---------- helpers (Py3.9-friendly) ----------
            #     import numpy as np, pandas as pd, matplotlib.pyplot as plt, shap
            #     from typing import Optional, List
            #     from sklearn.model_selection import train_test_split
            #     from sklearn.preprocessing import LabelEncoder
            #
            #     # prefer XGBoost; fallback to RandomForest if unavailable
            #     _HAS_XGB = True
            #     try:
            #         from xgboost import XGBClassifier
            #     except Exception:
            #         _HAS_XGB = False
            #         from sklearn.ensemble import RandomForestClassifier as _RF
            #
            #
            #     def _num_df(df: pd.DataFrame) -> pd.DataFrame:
            #         """Only numeric columns as float."""
            #         return df.select_dtypes(include=[np.number]).astype(float)
            #
            #
            #     def _clean_wavenumber_headers(df: pd.DataFrame,
            #                                   exclude_cols: Optional[List[str]] = None) -> pd.DataFrame:
            #         """
            #         Convert numeric-like column names to WHOLE-NUMBER integers (e.g., '1600.0' -> 1600).
            #         Exclude non-numeric/meta cols (e.g., 'Type', 'label') from renaming.
            #         """
            #         exclude_cols = set(exclude_cols or [])
            #         rename_map = {}
            #         for col in df.columns:
            #             if col in exclude_cols:
            #                 continue
            #             try:
            #                 v = float(str(col).strip())
            #                 rename_map[col] = int(round(v))  # keep as number
            #             except Exception:
            #                 pass
            #         df2 = df.rename(columns=rename_map)
            #         if len(set(df2.columns)) != len(list(df2.columns)):
            #             st.warning("Some headers collapsed to the same whole number after coercion. Please review.")
            #         return df2
            #
            #
            #     def _resolve_shap_matrix(shap_vals, class_idx=None, reduce_mode="mean"):
            #         """
            #         Return SHAP as 2D matrix (n_samples, n_features).
            #         Works for list-of-arrays, 3D (n,d,C), 2D (n,d), or 1D (d,).
            #         """
            #         if isinstance(shap_vals, list):
            #             if class_idx is not None:
            #                 return np.asarray(shap_vals[class_idx])  # (n,d)
            #             stack = np.stack([np.asarray(a) for a in shap_vals], axis=-1)  # (n,d,C)
            #             if reduce_mode == "sum":     return stack.sum(axis=2)
            #             if reduce_mode == "absmax":
            #                 idx = np.argmax(np.abs(stack), axis=2)
            #                 rows = np.arange(stack.shape[0])[:, None];
            #                 cols = np.arange(stack.shape[1])[None, :]
            #                 return stack[rows, cols, idx]
            #             return stack.mean(axis=2)
            #         arr = np.asarray(shap_vals)
            #         if arr.ndim == 3:
            #             if class_idx is not None:    return arr[:, :, class_idx]
            #             if reduce_mode == "sum":     return arr.sum(axis=2)
            #             if reduce_mode == "absmax":
            #                 idx = np.argmax(np.abs(arr), axis=2)
            #                 rows = np.arange(arr.shape[0])[:, None];
            #                 cols = np.arange(arr.shape[1])[None, :]
            #                 return arr[rows, cols, idx]
            #             return arr.mean(axis=2)
            #         if arr.ndim == 2: return arr
            #         if arr.ndim == 1: return arr[None, :]
            #         raise ValueError(f"Unsupported SHAP shape: {arr.shape}")
            #
            #
            #     def _reduce_expected_value(expected_value, class_idx=None, reduce_mode="mean"):
            #         ev = np.asarray(expected_value)
            #         if ev.ndim == 0:
            #             return float(ev)
            #         if class_idx is not None:
            #             return float(ev[class_idx])
            #         if reduce_mode == "sum":
            #             return float(ev.sum())
            #         if reduce_mode == "absmax":
            #             return float(ev[np.argmax(np.abs(ev))])
            #         return float(ev.mean())
            #
            #
            #     def _fit_model(X, y):
            #         """Keep your model choice; gracefully handle single-class edge cases."""
            #         if len(np.unique(y)) < 2:
            #             raise ValueError("Only one class present in data; need ≥2 classes for training.")
            #         if _HAS_XGB:
            #             m = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            #             m.fit(X, y);
            #             return m
            #         rf = _RF(n_estimators=300, random_state=42)
            #         rf.fit(X, y);
            #         return rf
            #
            #
            #     ################# this is for ratio dataset
            #     if "Quant peak ratios" in sheet_name:
            #         # Load dataset
            #         df = pd.read_excel(uploaded_file, sheet_name="Quant peak ratios")
            #
            #         import pandas as pd
            #         import shap
            #         import matplotlib.pyplot as plt
            #         import numpy as np
            #         from xgboost import XGBClassifier
            #         from sklearn.model_selection import train_test_split
            #         from sklearn.preprocessing import LabelEncoder
            #
            #         # --- Make headers numeric (wavenumbers), keep meta/string cols untouched ---
            #         string_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
            #         df = _clean_wavenumber_headers(df, exclude_cols=string_cols)
            #
            #         # Encode target variable (dynamic classes; no hard-coding)
            #         # Keep your logic: drop 'Type' then select numeric features only
            #         X = df.drop(columns=['Type']).select_dtypes(include=[np.number]).astype(float)
            #         y = df['Type'].astype(str)
            #         label_enc = LabelEncoder()
            #         y_encoded = label_enc.fit_transform(y)
            #
            #         # Train-test split (robust if tiny or single-class)
            #         try:
            #             X_train, X_test, y_train, y_test = train_test_split(
            #                 X, y_encoded, test_size=0.2,
            #                 stratify=(y_encoded if len(np.unique(y_encoded)) > 1 else None),
            #                 random_state=42
            #             )
            #         except ValueError:
            #             X_train, X_test, y_train, y_test = X, X, y_encoded, y_encoded
            #             st.info("Insufficient rows to split — using all rows for training and SHAP.")
            #
            #         # Train XGBoost model (or RF fallback)
            #         try:
            #             model = _fit_model(X_train, y_train)
            #         except Exception as e:
            #             st.error(f"Model training failed: {e}")
            #             st.stop()
            #
            #         # SHAP Explainer
            #         try:
            #             explainer = shap.TreeExplainer(model)
            #             shap_raw = explainer.shap_values(X_test)
            #             shap_values = _resolve_shap_matrix(shap_raw, class_idx=None, reduce_mode="mean")
            #             base_val = _reduce_expected_value(explainer.expected_value, class_idx=None, reduce_mode="mean")
            #         except Exception as e:
            #             st.error(f"SHAP computation failed: {e}")
            #             st.stop()
            #
            #         # Initialize session state to store plots
            #         if "beeswarm_plot" not in st.session_state:
            #             st.session_state.beeswarm_plot = None
            #         if "waterfall_plot" not in st.session_state:
            #             st.session_state.waterfall_plot = None
            #
            #         # Create two columns inside expander for layout control
            #         col1, col2 = st.columns([1, 1])
            #
            #         # ========================
            #         # 1️⃣ SHAP Beeswarm Plot
            #         # ========================
            #         with col1:
            #             st.subheader("📌 SHAP Beeswarm Plot")
            #             if st.button("Generate Beeswarm Plot"):
            #                 try:
            #                     fig_beeswarm = plt.figure(figsize=(6, 10))  # tall like your example
            #                     shap.summary_plot(shap_values, X_test, feature_names=list(X.columns), plot_type="dot",
            #                                       show=False)
            #                     st.session_state.beeswarm_plot = fig_beeswarm
            #                 except Exception as e:
            #                     st.error(f"Beeswarm failed: {e}")
            #
            #             # Display beeswarm plot if generated
            #             if st.session_state.beeswarm_plot is not None:
            #                 st.pyplot(st.session_state.beeswarm_plot)
            #
            #         # ========================
            #         # 2️⃣ Feature Importance Table
            #         # ========================
            #         with col2:
            #             st.subheader("📊 Feature Importance Table (Mean Absolute SHAP)")
            #             try:
            #                 mean_abs_shap = np.abs(shap_values).mean(axis=0)
            #                 feature_importance = pd.DataFrame({
            #                     "Feature": list(X.columns),
            #                     "Mean |SHAP Value|": mean_abs_shap,
            #                 }).sort_values(by="Mean |SHAP Value|", ascending=False).reset_index(drop=True)
            #
            #                 st.dataframe(feature_importance.style.format({"Mean |SHAP Value|": "{:.4f}"}),
            #                              use_container_width=True)
            #
            #                 csv = feature_importance.to_csv(index=False).encode('utf-8')
            #                 st.download_button("📥 Download Feature Importance CSV", data=csv,
            #                                    file_name="shap_feature_importance.csv", mime="text/csv")
            #             except Exception as e:
            #                 st.error(f"Feature importance failed: {e}")
            #
            #         st.markdown("---")
            #
            #         # ========================
            #         # 3️⃣ SHAP Waterfall Plot (Individual Sample)
            #         # ========================
            #         col_w1, col_w2 = st.columns([1, 1])
            #
            #         with col_w1:
            #             st.subheader("🎚 Select Sample Index")
            #             sample_idx = st.slider("Select Sample Index for Waterfall Plot", 0, max(0, len(X_test) - 1), 0)
            #
            #         with col_w2:
            #             st.subheader("💧 SHAP Waterfall Plot (Individual Sample)")
            #             if st.button("Generate Waterfall Plot"):
            #                 try:
            #                     vals = np.asarray(shap_values[sample_idx]).ravel()
            #                     data_row = np.asarray(X_test.iloc[sample_idx], dtype=float)
            #                     m = min(len(vals), len(data_row), len(X.columns))
            #                     explanation = shap.Explanation(
            #                         values=vals[:m],
            #                         base_values=base_val,
            #                         data=data_row[:m],
            #                         feature_names=list(X.columns)[:m]
            #                     )
            #                     fig_waterfall = plt.figure()
            #                     shap.plots.waterfall(explanation, show=False)
            #                     st.session_state.waterfall_plot = plt.gcf()
            #                 except Exception as e:
            #                     st.error(f"Waterfall failed: {e}")
            #
            #             # Display waterfall plot if generated
            #             if st.session_state.waterfall_plot is not None:
            #                 st.pyplot(st.session_state.waterfall_plot)
            #
            #     else:
            #
            #         # ===================== Non-Quant Data Modeling =====================
            #
            #         if sheet_name != "Quant peak ratios" :#or sheet_name != "Quant peak Intesity" :
            #
            #             import pandas as pd
            #             import shap
            #             import matplotlib.pyplot as plt
            #             import numpy as np
            #             from xgboost import XGBClassifier
            #             from sklearn.model_selection import train_test_split
            #             from sklearn.preprocessing import LabelEncoder
            #
            #             # Load the selected non-quant sheet
            #             df_nq = pd.read_excel(uploaded_file, sheet_name=sheet_name)
            #
            #             # --- Coerce headers to whole-number wavenumbers first ---
            #             string_cols_nq = [c for c in df_nq.columns if not pd.api.types.is_numeric_dtype(df_nq[c])]
            #             df_nq = _clean_wavenumber_headers(df_nq, exclude_cols=string_cols_nq)
            #
            #             # --- Target column (mirror Quant flow; default 'Type' if present)
            #             if "Type" in df_nq.columns:
            #                 label_col_nq = "Type"
            #             else:
            #                 non_numeric_cols_nq = [c for c in df_nq.columns if
            #                                        not pd.api.types.is_numeric_dtype(df_nq[c])]
            #                 if not non_numeric_cols_nq:
            #                     st.error(
            #                         "No suitable label/target column found (e.g., 'Type'). Please add a categorical target.")
            #                     st.stop()
            #                 label_col_nq = st.selectbox("Select label/target column", non_numeric_cols_nq, index=0)
            #
            #             # --- Features/Target (numeric features only; exclude ALL string/meta cols)
            #             X_nq = df_nq.drop(columns=[c for c in df_nq.columns if c in string_cols_nq]).select_dtypes(
            #                 include=[np.number]).astype(float)
            #             y_nq = df_nq[label_col_nq].astype(str)
            #
            #             if X_nq.shape[1] == 0:
            #                 st.error("No numeric feature columns available in this sheet.")
            #                 st.stop()
            #
            #             # Ensure feature names (keep numeric wavenumbers)
            #             feat_names_nq = list(X_nq.columns)
            #
            #             # --- Encode target (dynamic classes)
            #             label_enc_nq = LabelEncoder()
            #             y_encoded_nq = label_enc_nq.fit_transform(y_nq)
            #
            #             # Mirror your "ALL rows" approach for SHAP tables/plots
            #             X_all = X_nq
            #             y_encoded_all = y_encoded_nq
            #             feat_names_all = feat_names_nq
            #
            #             # --- Train-test split (as in Quant)
            #             stratify_vec_all = y_encoded_all if len(np.unique(y_encoded_all)) > 1 else None
            #             try:
            #                 X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
            #                     X_all, y_encoded_all, test_size=0.2, stratify=stratify_vec_all, random_state=42
            #                 )
            #             except ValueError:
            #                 # Fallback for tiny/single-class cases
            #                 X_train_all, X_test_all, y_train_all, y_test_all = X_all, X_all, y_encoded_all, y_encoded_all
            #                 st.info(
            #                     "Using all rows for training and explanation (insufficient data to stratify/split).")
            #
            #             # --- Train XGBoost model (same as Quant)
            #             try:
            #                 model_all = _fit_model(X_train_all, y_train_all)
            #             except Exception as e:
            #                 st.error(f"Model training failed: {e}")
            #                 st.stop()
            #
            #             # --- SHAP Explainer (compute SHAP on ALL rows for the table and plots)
            #             try:
            #                 explainer_all = shap.TreeExplainer(model_all)
            #                 shap_values_raw_all = explainer_all.shap_values(X_all)
            #                 expected_value_all = explainer_all.expected_value
            #                 shap_values_all = _resolve_shap_matrix(shap_values_raw_all, class_idx=None,
            #                                                        reduce_mode="mean")
            #                 base_val_all = _reduce_expected_value(expected_value_all, class_idx=None,
            #                                                       reduce_mode="mean")
            #             except Exception as e:
            #                 st.error(f"SHAP computation failed: {e}")
            #                 st.stop()
            #
            #             # --- Session state for plots (separate from Quant keys)
            #             if "beeswarm_plot_nq_all" not in st.session_state:
            #                 st.session_state.beeswarm_plot_nq_all = None
            #             if "waterfall_plot_nq_all" not in st.session_state:
            #                 st.session_state.waterfall_plot_nq_all = None
            #
            #             # Create two columns (mirror Quant layout)
            #             col1_all, col2_all = st.columns([1, 1])
            #
            #             # ========================
            #             # 1️⃣ SHAP Beeswarm Plot
            #             # ========================
            #             with col1_all:
            #                 st.subheader("📌 SHAP Beeswarm Plot (All Rows)")
            #                 if st.button("Generate Beeswarm Plot (Non-Quant, All Rows)"):
            #                     try:
            #                         fig_beeswarm_all = plt.figure(figsize=(6, 10))
            #                         shap.summary_plot(
            #                             shap_values_all, X_all,
            #                             feature_names=[str(f) for f in feat_names_all],  # display as labels
            #                             plot_type="dot", show=False
            #                         )
            #                         st.session_state.beeswarm_plot_nq_all = fig_beeswarm_all
            #                     except Exception as e:
            #                         st.error(f"Beeswarm failed: {e}")
            #
            #                 if st.session_state.beeswarm_plot_nq_all is not None:
            #                     st.pyplot(st.session_state.beeswarm_plot_nq_all)
            #
            #             ################################################################################################
            #             ################################  [ OIM-WT Working Script ]  ################################################################
            #
            #             # ========================
            #             # 2️⃣ Feature Importance Table (ALL ROWS)
            #             # ========================
            #             with col2_all:
            #                 st.subheader("📊 Feature Importance Table (Mean Absolute SHAP) – All Rows")
            #                 try:
            #                     mean_abs_shap_all = np.abs(shap_values_all).mean(axis=0)
            #                     feature_importance_all = pd.DataFrame({
            #                         "Feature": [str(f) for f in feat_names_all],
            #                         "Mean |SHAP Value|": mean_abs_shap_all,
            #                     }).sort_values(by="Mean |SHAP Value|", ascending=False).reset_index(drop=True)
            #
            #                     st.dataframe(
            #                         feature_importance_all.style.format({"Mean |SHAP Value|": "{:.4f}"}),
            #                         use_container_width=True
            #                     )
            #
            #                     csv_all = feature_importance_all.to_csv(index=False).encode('utf-8')
            #                     st.download_button(
            #                         "📥 Download Feature Importance CSV (All Rows, Non-Quant)",
            #                         data=csv_all,
            #                         file_name=f"shap_feature_importance_{sheet_name}_ALL_ROWS.csv",
            #                         mime="text/csv"
            #                     )
            #                 except Exception as e:
            #                     st.error(f"Feature importance failed: {e}")
            #
            #             ################################################################################################
            #             ################################## [ General Script: Working Script ] ##############################################################
            #
            #             ################################################################################################
            #             ################################################################################################
            #             st.markdown("---")
            #
            #             # ========================
            #             # 3️⃣ SHAP Waterfall Plot (Individual Sample, from ALL ROWS)
            #             # ========================
            #             col_w1_all, col_w2_all = st.columns([1, 1])
            #
            #             with col_w1_all:
            #                 st.subheader("🎚 Select Sample Index (All Rows)")
            #                 if len(X_all) > 0:
            #                     sample_idx_all = st.slider(
            #                         "Select Sample Index for Waterfall Plot (Non-Quant, All Rows)",
            #                         0, len(X_all) - 1, 0
            #                     )
            #                 else:
            #                     sample_idx_all = 0
            #                     st.info("No rows available to plot.")
            #
            #             with col_w2_all:
            #                 st.subheader("💧 SHAP Waterfall Plot (Individual Sample)")
            #                 if st.button("Generate Waterfall Plot (Non-Quant, All Rows)"):
            #                     try:
            #                         vals = np.asarray(shap_values_all[sample_idx_all]).ravel()
            #                         data_row = np.asarray(X_all.iloc[sample_idx_all], dtype=float)
            #                         m = min(len(vals), len(data_row), len(feat_names_all))
            #                         explanation_all = shap.Explanation(
            #                             values=vals[:m],
            #                             base_values=base_val_all,
            #                             data=data_row[:m],
            #                             feature_names=[str(f) for f in feat_names_all[:m]]
            #                         )
            #                         fig_waterfall_all = plt.figure()
            #                         shap.plots.waterfall(explanation_all, show=False)
            #                         st.session_state.waterfall_plot_nq_all = plt.gcf()
            #                     except Exception as e:
            #                         st.error(f"Waterfall failed: {e}")
            #
            #                 if st.session_state.waterfall_plot_nq_all is not None:
            #                     st.pyplot(st.session_state.waterfall_plot_nq_all)

            ###########################################  [New Script - END: SHAP Analysis ]  ##########################################################
            ###########################################################################################################################################

            # ================================
            # 🔎 SHAP Analysis (Complete Block)
            # ================================

            st.subheader("🔎 SHAP Analysis")

            with st.expander("🔎 SHAP Analysis of FTIR Ratios", expanded=False):

                # ---------- Imports (local to block to avoid top-level clutter) ----------
                import re
                import numpy as np, pandas as pd, matplotlib.pyplot as plt, shap
                from typing import Optional, List, Tuple, Dict
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import LabelEncoder

                # Prefer XGBoost; fallback to RandomForest if unavailable
                _HAS_XGB = True
                try:
                    from xgboost import XGBClassifier
                except Exception:
                    _HAS_XGB = False
                    from sklearn.ensemble import RandomForestClassifier as _RF

                # ---------- Helpers ----------
                _NUM_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)")


                def clean_wavenumber_headers(
                        df: pd.DataFrame,
                        exclude_cols: Optional[List[str]] = None,
                        round_to_int: bool = True
                ) -> Tuple[pd.DataFrame, Dict[str, str], bool]:
                    """
                    Convert numeric-like headers to integers (as str), leave meta cols unchanged,
                    parse first numeric token (e.g., '1030/1670'), and dedupe collisions with suffixes: 1030, 1030_1, ...
                    Returns: (renamed_df, rename_map, had_collisions)
                    """
                    exclude = set(exclude_cols or [])
                    rename_map: Dict[str, str] = {}
                    used: Dict[str, int] = {}
                    had_collisions = False

                    for col in df.columns:
                        if col in exclude:
                            continue
                        new_name = str(col)
                        m = _NUM_RE.match(str(col))
                        if m:
                            num = float(m.group(1))
                            new_name = str(int(round(num))) if round_to_int else str(num)

                        base = new_name
                        if base in used:
                            had_collisions = True
                            used[base] += 1
                            new_name = f"{base}_{used[base]}"
                        else:
                            used[base] = 0

                        if str(col) != new_name:
                            rename_map[col] = new_name

                    df2 = df.rename(columns=rename_map)
                    return df2, rename_map, had_collisions


                def _resolve_shap_matrix(shap_vals, class_idx=None, reduce_mode="mean"):
                    """
                    Return SHAP as 2D matrix (n_samples, n_features). Supports list, 3D, 2D, 1D.
                    If multiclass and class_idx=None, reduces across classes by 'mean' (default), 'sum', or 'absmax'.
                    """
                    if isinstance(shap_vals, list):
                        if class_idx is not None:
                            return np.asarray(shap_vals[class_idx])
                        stack = np.stack([np.asarray(a) for a in shap_vals], axis=-1)  # (n,d,C)
                        if reduce_mode == "sum":     return stack.sum(axis=2)
                        if reduce_mode == "absmax":
                            idx = np.argmax(np.abs(stack), axis=2)
                            rows = np.arange(stack.shape[0])[:, None]
                            cols = np.arange(stack.shape[1])[None, :]
                            return stack[rows, cols, idx]
                        return stack.mean(axis=2)
                    arr = np.asarray(shap_vals)
                    if arr.ndim == 3:
                        if class_idx is not None:    return arr[:, :, class_idx]
                        if reduce_mode == "sum":     return arr.sum(axis=2)
                        if reduce_mode == "absmax":
                            idx = np.argmax(np.abs(arr), axis=2)
                            rows = np.arange(arr.shape[0])[:, None]
                            cols = np.arange(arr.shape[1])[None, :]
                            return arr[rows, cols, idx]
                        return arr.mean(axis=2)
                    if arr.ndim == 2: return arr
                    if arr.ndim == 1: return arr[None, :]
                    raise ValueError(f"Unsupported SHAP shape: {arr.shape}")


                def _reduce_expected_value(expected_value, class_idx=None, reduce_mode="mean"):
                    ev = np.asarray(expected_value)
                    if ev.ndim == 0:         return float(ev)
                    if class_idx is not None: return float(ev[class_idx])
                    if reduce_mode == "sum":  return float(ev.sum())
                    if reduce_mode == "absmax": return float(ev[np.argmax(np.abs(ev))])
                    return float(ev.mean())


                def _fit_model(X, y):
                    """Train XGB (or RF fallback). Guard against single-class targets."""
                    if len(np.unique(y)) < 2:
                        raise ValueError("Only one class present in data; need ≥2 classes for training.")
                    if _HAS_XGB:
                        m = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
                        m.fit(X, y)
                        return m
                    rf = _RF(n_estimators=300, random_state=42)
                    rf.fit(X, y)
                    return rf


                # =====================
                # Branch: Quant peak ratios
                # =====================
                if "Quant peak ratios" in sheet_name:
                    # Load
                    df = pd.read_excel(uploaded_file, sheet_name="Quant peak ratios")

                    # Identify meta (non-numeric) BEFORE cleaning
                    string_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

                    # Clean headers safely
                    df, rename_map_q, had_collisions_q = clean_wavenumber_headers(df, exclude_cols=string_cols)
                    if had_collisions_q:
                        print('----')
                        # st.info("Some wavenumber headers rounded to the same integer in 'Quant peak ratios'; suffixed with _1, _2 to keep them distinct.")

                    # Robust target extraction
                    if "Type" not in df.columns:
                        st.error("Expected target column 'Type' not found in 'Quant peak ratios'.")
                        st.stop()
                    y_raw = df.loc[:, "Type"]  # may be Series or DataFrame (if duplicates)
                    y = y_raw.iloc[:, 0].astype(str) if isinstance(y_raw, pd.DataFrame) else y_raw.astype(str)

                    # Numeric feature matrix (exclude meta + target)
                    X = (
                        df.drop(columns=[c for c in df.columns if c in string_cols or c == "Type"])
                        .select_dtypes(include=[np.number])
                        .astype(float)
                    )
                    if X.shape[1] == 0:
                        st.error("No numeric feature columns available after cleaning in 'Quant peak ratios'.")
                        st.stop()

                    # Encode target
                    label_enc = LabelEncoder()
                    y_encoded = label_enc.fit_transform(y)

                    # Split (or fallback to full set)
                    try:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y_encoded, test_size=0.2,
                            stratify=(y_encoded if len(np.unique(y_encoded)) > 1 else None),
                            random_state=42
                        )
                    except ValueError:
                        X_train, X_test, y_train, y_test = X, X, y_encoded, y_encoded
                        st.info("Insufficient rows to split — using all rows for training and SHAP.")

                    # Train
                    try:
                        model = _fit_model(X_train, y_train)
                    except Exception as e:
                        st.error(f"Model training failed: {e}")
                        st.stop()

                    # SHAP
                    try:
                        explainer = shap.TreeExplainer(model)
                        shap_raw = explainer.shap_values(X_test)
                        shap_values = _resolve_shap_matrix(shap_raw, class_idx=None, reduce_mode="mean")
                        base_val = _reduce_expected_value(explainer.expected_value, class_idx=None, reduce_mode="mean")
                    except Exception as e:
                        st.error(f"SHAP computation failed: {e}")
                        st.stop()

                    # Session state for plots (Quant)
                    if "beeswarm_plot_quant" not in st.session_state:
                        st.session_state.beeswarm_plot_quant = None
                    if "waterfall_plot_quant" not in st.session_state:
                        st.session_state.waterfall_plot_quant = None

                    # Layout
                    col1, col2 = st.columns([1, 1])

                    # 1) Beeswarm
                    with col1:
                        st.subheader("📌 SHAP Beeswarm Plot")
                        if st.button("Generate Beeswarm Plot", key="btn_beeswarm_quant"):
                            try:
                                fig_beeswarm = plt.figure(figsize=(6, 10))
                                shap.summary_plot(shap_values, X_test, feature_names=list(X.columns), plot_type="dot",
                                                  show=False)
                                st.session_state.beeswarm_plot_quant = fig_beeswarm
                            except Exception as e:
                                st.error(f"Beeswarm failed: {e}")
                        if st.session_state.beeswarm_plot_quant is not None:
                            st.pyplot(st.session_state.beeswarm_plot_quant)

                    # 2) Feature Importance
                    with col2:
                        st.subheader("📊 Feature Importance (Mean |SHAP|)")
                        try:
                            mean_abs_shap = np.abs(shap_values).mean(axis=0)
                            feature_importance = pd.DataFrame({
                                "Feature": list(X.columns),
                                "Mean |SHAP Value|": mean_abs_shap,
                            }).sort_values(by="Mean |SHAP Value|", ascending=False).reset_index(drop=True)

                            st.dataframe(feature_importance.style.format({"Mean |SHAP Value|": "{:.4f}"}),
                                         use_container_width=True)
                            csv = feature_importance.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "📥 Download Feature Importance CSV",
                                data=csv,
                                file_name="shap_feature_importance_quant.csv",
                                mime="text/csv",
                                key="dl_feat_imp_quant"
                            )
                        except Exception as e:
                            st.error(f"Feature importance failed: {e}")

                    st.markdown("---")

                    # 3) Waterfall (individual sample)
                    col_w1, col_w2 = st.columns([1, 1])

                    with col_w1:
                        st.subheader("🎚 Select Sample Index")
                        sample_idx = st.slider("Select Sample Index for Waterfall Plot", 0, max(0, len(X_test) - 1), 0,
                                               key="slider_idx_quant")

                    with col_w2:
                        st.subheader("💧 SHAP Waterfall Plot (Individual Sample)")
                        if st.button("Generate Waterfall Plot", key="btn_waterfall_quant"):
                            try:
                                vals = np.asarray(shap_values[sample_idx]).ravel()
                                data_row = np.asarray(X_test.iloc[sample_idx], dtype=float)
                                m = min(len(vals), len(data_row), len(X.columns))
                                explanation = shap.Explanation(
                                    values=vals[:m],
                                    base_values=base_val,
                                    data=data_row[:m],
                                    feature_names=list(X.columns)[:m]
                                )
                                fig_waterfall = plt.figure()
                                shap.plots.waterfall(explanation, show=False)
                                st.session_state.waterfall_plot_quant = plt.gcf()
                            except Exception as e:
                                st.error(f"Waterfall failed: {e}")
                        if st.session_state.waterfall_plot_quant is not None:
                            st.pyplot(st.session_state.waterfall_plot_quant)

                # =====================
                # Branch: All other sheets (includes "Quant peak Intesity")
                # =====================
                else:
                    # Load the selected non-quant sheet (whatever `sheet_name` is)
                    df_nq = pd.read_excel(uploaded_file, sheet_name=sheet_name)

                    # Clean headers safely
                    string_cols_nq = [c for c in df_nq.columns if not pd.api.types.is_numeric_dtype(df_nq[c])]
                    df_nq, rename_map_nq, had_collisions_nq = clean_wavenumber_headers(df_nq,
                                                                                       exclude_cols=string_cols_nq)
                    if had_collisions_nq:
                        st.info(
                            f"Some wavenumber headers rounded to the same integer in '{sheet_name}'; suffixed with _1, _2 to keep them distinct.")

                    # Target column (prefer 'Type', else ask)
                    if "Type" in df_nq.columns:
                        label_col_nq = "Type"
                    else:
                        non_numeric_cols_nq = [c for c in df_nq.columns if not pd.api.types.is_numeric_dtype(df_nq[c])]
                        if not non_numeric_cols_nq:
                            st.error(
                                "No suitable label/target column found (e.g., 'Type'). Please add a categorical target.")
                            st.stop()
                        label_col_nq = st.selectbox("Select label/target column", non_numeric_cols_nq, index=0,
                                                    key=f"sel_target_nq_{sheet_name}")

                    # Robust target extraction (Series even if duplicates)
                    y_raw_nq = df_nq.loc[:, label_col_nq]
                    y_nq = y_raw_nq.iloc[:, 0].astype(str) if isinstance(y_raw_nq, pd.DataFrame) else y_raw_nq.astype(
                        str)

                    # Features: numeric only, exclude meta + label
                    X_nq = (
                        df_nq.drop(columns=[c for c in df_nq.columns if c in string_cols_nq or c == label_col_nq])
                        .select_dtypes(include=[np.number])
                        .astype(float)
                    )
                    if X_nq.shape[1] == 0:
                        st.error("No numeric feature columns available in this sheet after cleaning.")
                        st.stop()

                    feat_names_nq = list(X_nq.columns)

                    # Encode labels
                    label_enc_nq = LabelEncoder()
                    y_encoded_nq = label_enc_nq.fit_transform(y_nq)

                    # Train/test split (robust)
                    stratify_vec_all = y_encoded_nq if len(np.unique(y_encoded_nq)) > 1 else None
                    try:
                        X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
                            X_nq, y_encoded_nq, test_size=0.2, stratify=stratify_vec_all, random_state=42
                        )
                    except ValueError:
                        X_train_all, X_test_all, y_train_all, y_test_all = X_nq, X_nq, y_encoded_nq, y_encoded_nq
                        st.info("Using all rows for training and explanation (insufficient data to stratify/split).")

                    # Train
                    try:
                        model_all = _fit_model(X_train_all, y_train_all)
                    except Exception as e:
                        st.error(f"Model training failed: {e}")
                        st.stop()

                    # SHAP on ALL rows (so table/plots can reference full set)
                    try:
                        explainer_all = shap.TreeExplainer(model_all)
                        shap_values_raw_all = explainer_all.shap_values(X_nq)
                        expected_value_all = explainer_all.expected_value
                        shap_values_all = _resolve_shap_matrix(shap_values_raw_all, class_idx=None, reduce_mode="mean")
                        base_val_all = _reduce_expected_value(expected_value_all, class_idx=None, reduce_mode="mean")
                    except Exception as e:
                        st.error(f"SHAP computation failed: {e}")
                        st.stop()

                    # Session state for plots (Non-Quant)
                    if "beeswarm_plot_nq_all" not in st.session_state:
                        st.session_state.beeswarm_plot_nq_all = None
                    if "waterfall_plot_nq_all" not in st.session_state:
                        st.session_state.waterfall_plot_nq_all = None

                    # Layout
                    col1_all, col2_all = st.columns([1, 1])

                    # 1) Beeswarm (ALL rows)
                    with col1_all:
                        st.subheader("📌 SHAP Beeswarm Plot (All Rows)")
                        if st.button("Generate Beeswarm Plot (Non-Quant, All Rows)",
                                     key=f"btn_beeswarm_nq_{sheet_name}"):
                            try:
                                fig_beeswarm_all = plt.figure(figsize=(6, 10))
                                shap.summary_plot(
                                    shap_values_all, X_nq,
                                    feature_names=[str(f) for f in feat_names_nq],
                                    plot_type="dot", show=False
                                )
                                st.session_state.beeswarm_plot_nq_all = fig_beeswarm_all
                            except Exception as e:
                                st.error(f"Beeswarm failed: {e}")
                        if st.session_state.beeswarm_plot_nq_all is not None:
                            st.pyplot(st.session_state.beeswarm_plot_nq_all)

                    # 2) Feature Importance (ALL rows)
                    with col2_all:
                        st.subheader("📊 Feature Importance (Mean |SHAP|) – All Rows")
                        try:
                            mean_abs_shap_all = np.abs(shap_values_all).mean(axis=0)
                            feature_importance_all = pd.DataFrame({
                                "Feature": [str(f) for f in feat_names_nq],
                                "Mean |SHAP Value|": mean_abs_shap_all,
                            }).sort_values(by="Mean |SHAP Value|", ascending=False).reset_index(drop=True)

                            st.dataframe(
                                feature_importance_all.style.format({"Mean |SHAP Value|": "{:.4f}"}),
                                use_container_width=True
                            )
                            csv_all = feature_importance_all.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "📥 Download Feature Importance CSV (All Rows, Non-Quant)",
                                data=csv_all,
                                file_name=f"shap_feature_importance_{sheet_name}_ALL_ROWS.csv",
                                mime="text/csv",
                                key=f"dl_feat_imp_nq_{sheet_name}"
                            )
                        except Exception as e:
                            st.error(f"Feature importance failed: {e}")

                    st.markdown("---")

                    # 3) Waterfall (individual sample, from ALL rows)
                    col_w1_all, col_w2_all = st.columns([1, 1])

                    with col_w1_all:
                        st.subheader("🎚 Select Sample Index (All Rows)")
                        if len(X_nq) > 0:
                            sample_idx_all = st.slider(
                                "Select Sample Index for Waterfall Plot (Non-Quant, All Rows)",
                                0, len(X_nq) - 1, 0, key=f"slider_idx_nq_{sheet_name}"
                            )
                        else:
                            sample_idx_all = 0
                            st.info("No rows available to plot.")

                    with col_w2_all:
                        st.subheader("💧 SHAP Waterfall Plot (Individual Sample)")
                        if st.button("Generate Waterfall Plot (Non-Quant, All Rows)",
                                     key=f"btn_waterfall_nq_{sheet_name}"):
                            try:
                                vals = np.asarray(shap_values_all[sample_idx_all]).ravel()
                                data_row = np.asarray(X_nq.iloc[sample_idx_all], dtype=float)
                                m = min(len(vals), len(data_row), len(feat_names_nq))
                                explanation_all = shap.Explanation(
                                    values=vals[:m],
                                    base_values=base_val_all,
                                    data=data_row[:m],
                                    feature_names=[str(f) for f in feat_names_nq[:m]]
                                )
                                fig_waterfall_all = plt.figure()
                                shap.plots.waterfall(explanation_all, show=False)
                                st.session_state.waterfall_plot_nq_all = plt.gcf()
                            except Exception as e:
                                st.error(f"Waterfall failed: {e}")
                        if st.session_state.waterfall_plot_nq_all is not None:
                            st.pyplot(st.session_state.waterfall_plot_nq_all)

            #################  [Old Script - START: 📋 Model Results with Detailed Metric ]  #####################################
            #######################################################################################################################

            ###################################### [START:  Select Models ] ##################################################
            #
            # if show_models:
            #     st.subheader("📋 Model Results with Detailed Metrics")
            #
            #     # Create two columns and use only the first column for 50% width
            #     col1, col2 = st.columns([80, 20])  # Split page into 50% and 50%
            #     with col1:
            #         with st.expander("📊 Click to View The Selected Model Results", expanded=True):
            #             results = []
            #             models_to_run = model_options if run_all else selected_models
            #
            #             for model_name in models_to_run:
            #                 # Select model based on user choice
            #                 if model_name == "SVM":
            #                     model = SVC()
            #                 elif model_name == "Logistic Regression":
            #                     model = LogisticRegression(max_iter=1000)
            #                 elif model_name == "KNN":
            #                     model = KNeighborsClassifier(n_neighbors=3)
            #                 elif model_name == "Random Forest":
            #                     model = RandomForestClassifier(n_estimators=100)
            #                 elif model_name == "XGBoost":
            #                     model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            #
            #                 # Split data into train/test sets
            #                 X_train, X_test, y_train, y_test = train_test_split(
            #                     X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
            #                 )
            #
            #                 # Fit model and make predictions
            #                 model.fit(X_train, y_train)
            #                 y_pred = model.predict(X_test)
            #
            #                 # Calculate performance metrics
            #                 acc = model.score(X_test, y_test)
            #                 f1 = f1_score(y_test, y_pred)
            #                 precision = precision_score(y_test, y_pred)
            #                 recall = recall_score(y_test, y_pred)
            #
            #                 # Append results to list
            #                 results.append({
            #                     "Model": model_name,
            #                     "Accuracy": acc,
            #                     "Precision": precision,
            #                     "Recall": recall,
            #                     "F1-Score": f1
            #                 })
            #
            #                 # Display Classification Report & Confusion Matrix Side-by-Side
            #                 st.markdown(f"### 🔎 Analysis for {model_name}")
            #                 report_col, matrix_col = st.columns([2, 3])  # 40% report & 60% matrix
            #
            #                 with report_col:
            #                     st.markdown("**Classification Report:**")
            #
            #                     # Generate classification report as a DataFrame
            #                     report_dict = classification_report(y_test, y_pred, target_names=["WT", "OIM"],
            #                                                         output_dict=True)
            #                     report_df = pd.DataFrame(report_dict).transpose()
            #
            #                     # Style: Add border to table
            #                     styled_report = report_df.style.set_table_styles([
            #                         {'selector': 'th', 'props': [('border', '1px solid black'), ('padding', '8px'),
            #                                                      ('background', '#f2f2f2')]},
            #                         {'selector': 'td', 'props': [('border', '1px solid black'), ('padding', '8px')]},
            #                         {'selector': 'tr', 'props': [('border', '1px solid black')]}
            #                     ]).format("{:.2f}")
            #
            #                     st.write(styled_report)
            #
            #                 with matrix_col:
            #                     # Confusion Matrix in PERCENT (row-normalized)
            #                     cm = confusion_matrix(y_test, y_pred)
            #                     with np.errstate(divide="ignore", invalid="ignore"):
            #                         cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
            #                     cm_pct = np.nan_to_num(cm_pct)
            #
            #                     # Pretty % labels for annotations
            #                     annot_labels = np.array([[f"{v:.1f}%" for v in row] for row in cm_pct])
            #
            #                     fig_cm, ax = plt.subplots()
            #                     sns.heatmap(
            #                         cm_pct,
            #                         annot=annot_labels,
            #                         fmt="",
            #                         cmap="Blues",
            #                         vmin=0, vmax=100,
            #                         xticklabels=["WT", "OIM"],
            #                         yticklabels=["WT", "OIM"],
            #                         cbar_kws={"label": "% of true class"}
            #                     )
            #                     ax.set_xlabel("Predicted")
            #                     ax.set_ylabel("True")
            #                     ax.set_title(f"{model_name} - Confusion Matrix (%)")
            #                     st.pyplot(fig_cm)
            #
            #             # Show metrics summary table
            #             results_df = pd.DataFrame(results)
            #             st.markdown("### 📊 Model Performance Summary")
            #             st.dataframe(results_df.style.background_gradient(cmap='YlGnBu'), use_container_width=True)
            #
            #             # Download metrics CSV
            #             metrics_csv = results_df.to_csv(index=False).encode('utf-8')
            #             st.download_button("📥 Download Model Metrics CSV", metrics_csv, "model_metrics.csv", "text/csv")


            ###################################### [END:  Select Models ] #########################################

            #################  [Old Script - END: 📋 Model Results with Detailed Metric ]  #####################################
            #######################################################################################################################


            #################  [New Script - Start: 📋 Model Results with Detailed Metric ]  #####################################
            #######################################################################################################################

            if show_models:
                st.subheader("📋 Model Results with Detailed Metrics")

                # ---- imports + helpers (local to this block) ----
                import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
                from sklearn.metrics import (
                    classification_report, confusion_matrix, accuracy_score,
                    precision_score, recall_score, f1_score
                )


                def _safe_prf(y_true, y_pred):
                    """
                    Precision/Recall/F1 that work for single-class, binary, and multiclass.
                    - single-class: returns 1.0 if predictions match perfectly, else 0.0
                    - binary: uses 'binary' average with consistent positive label
                    - multiclass: uses 'weighted' average
                    """
                    y_true = np.asarray(y_true);
                    y_pred = np.asarray(y_pred)
                    labels_present = np.unique(np.concatenate([y_true, y_pred]))
                    n = len(labels_present)

                    if n == 1:
                        same = bool(np.all(y_true == y_pred))
                        return {"precision": 1.0 if same else 0.0,
                                "recall": 1.0 if same else 0.0,
                                "f1": 1.0 if same else 0.0,
                                "avg": "single-class", "labels": labels_present}

                    if n == 2:
                        pos = labels_present[1]  # deterministic choice
                        return {
                            "precision": precision_score(y_true, y_pred, labels=labels_present, pos_label=pos,
                                                         average="binary", zero_division=0),
                            "recall": recall_score(y_true, y_pred, labels=labels_present, pos_label=pos,
                                                   average="binary", zero_division=0),
                            "f1": f1_score(y_true, y_pred, labels=labels_present, pos_label=pos,
                                           average="binary", zero_division=0),
                            "avg": "binary", "labels": labels_present
                        }

                    # 3+ classes
                    return {
                        "precision": precision_score(y_true, y_pred, labels=labels_present,
                                                     average="weighted", zero_division=0),
                        "recall": recall_score(y_true, y_pred, labels=labels_present,
                                               average="weighted", zero_division=0),
                        "f1": f1_score(y_true, y_pred, labels=labels_present,
                                       average="weighted", zero_division=0),
                        "avg": "weighted", "labels": labels_present
                    }


                # Try to find a LabelEncoder from earlier (quant/non-quant code paths)
                def _pick_encoder():
                    try:
                        return label_enc
                    except NameError:
                        pass
                    try:
                        return le_all
                    except NameError:
                        pass
                    try:
                        return label_enc_nq
                    except NameError:
                        pass
                    return None


                enc = _pick_encoder()

                # Create two columns and use only the first column for 80/20 split
                col1, col2 = st.columns([80, 20])
                with col1:
                    with st.expander("📊 Click to View The Selected Model Results", expanded=True):
                        results = []
                        models_to_run = model_options if run_all else selected_models

                        for model_name in models_to_run:
                            # Select model based on user choice (unchanged)
                            if model_name == "SVM":
                                model = SVC()
                            elif model_name == "Logistic Regression":
                                model = LogisticRegression(max_iter=1000)
                            elif model_name == "KNN":
                                model = KNeighborsClassifier(n_neighbors=3)
                            elif model_name == "Random Forest":
                                model = RandomForestClassifier(n_estimators=100)
                            elif model_name == "XGBoost":
                                model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

                            # Split data into train/test sets (keep your params, guard tiny splits)
                            try:
                                X_train, X_test, y_train, y_test = train_test_split(
                                    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
                                )
                            except ValueError:
                                # Fallback if single-class or too few samples for stratify
                                X_train, X_test, y_train, y_test = X, X, y_encoded, y_encoded

                            # Fit model and make predictions
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)

                            # ---- Metrics (dynamic & safe) ----
                            acc = accuracy_score(y_test, y_pred)
                            prf = _safe_prf(y_test, y_pred)  # precision/recall/f1 + labels
                            precision = prf["precision"]
                            recall = prf["recall"]
                            f1 = prf["f1"]
                            labels_present = prf["labels"]

                            # Save results row
                            results.append({
                                "Model": model_name,
                                "Accuracy": acc,
                                "Precision": precision,
                                "Recall": recall,
                                "F1-Score": f1,
                                "Averaging": prf["avg"]
                            })

                            # ---- Build readable class names (dynamic) ----
                            if np.issubdtype(labels_present.dtype, np.integer) and enc is not None:
                                try:
                                    class_names = [str(x) for x in enc.inverse_transform(labels_present)]
                                except Exception:
                                    class_names = [str(x) for x in labels_present]
                            else:
                                class_names = [str(x) for x in labels_present]

                            # Display Classification Report & Confusion Matrix Side-by-Side
                            st.markdown(f"### 🔎 Analysis for {model_name}")
                            report_col, matrix_col = st.columns([2, 3])

                            with report_col:
                                st.markdown("**Classification Report:**")
                                # Use the same labels order + names
                                report_dict = classification_report(
                                    y_test, y_pred,
                                    labels=labels_present,
                                    target_names=class_names,
                                    output_dict=True,
                                    zero_division=0
                                )
                                report_df = pd.DataFrame(report_dict).transpose()

                                styled_report = report_df.style.set_table_styles([
                                    {'selector': 'th', 'props': [('border', '1px solid black'),
                                                                 ('padding', '8px'), ('background', '#f2f2f2')]},
                                    {'selector': 'td', 'props': [('border', '1px solid black'), ('padding', '8px')]},
                                    {'selector': 'tr', 'props': [('border', '1px solid black')]}
                                ]).format("{:.2f}")

                                st.write(styled_report)

                            with matrix_col:
                                # Confusion Matrix (%), row-normalized
                                cm = confusion_matrix(y_test, y_pred, labels=labels_present)
                                with np.errstate(divide="ignore", invalid="ignore"):
                                    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
                                cm_pct = np.nan_to_num(cm_pct)

                                annot_labels = np.array([[f"{v:.1f}%" for v in row] for row in cm_pct])

                                fig_cm, ax = plt.subplots()
                                sns.heatmap(
                                    cm_pct,
                                    annot=annot_labels,
                                    fmt="",
                                    cmap="Blues",
                                    vmin=0, vmax=100,
                                    xticklabels=class_names,
                                    yticklabels=class_names,
                                    cbar_kws={"label": "% of true class"}
                                )
                                ax.set_xlabel("Predicted")
                                ax.set_ylabel("True")
                                ax.set_title(f"{model_name} - Confusion Matrix (%)")
                                st.pyplot(fig_cm)

                        # Show metrics summary table
                        results_df = pd.DataFrame(results)
                        st.markdown("### 📊 Model Performance Summary")
                        st.dataframe(results_df.style.background_gradient(cmap='YlGnBu'), use_container_width=True)

                        # Download metrics CSV
                        metrics_csv = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Download Model Metrics CSV", metrics_csv, "model_metrics.csv", "text/csv")

            #################  [New Script - END: 📋 Model Results with Detailed Metric ]  #####################################
            #######################################################################################################################


            ##################################### [ PA Analysis ] ######################################
            if show_pca:
                # Create two columns
                col1, col2 = st.columns([4, 1])  # 4:1 width ratio (adjust as needed)

                with col1:
                    st.subheader("🔬 PCA: Principal Component Analysis")

                # =============================================================================================================

                with st.expander(
                        "PCA (Principal Component Analysis) reduces the high-dimensional FTIR spectral data into 2D and 3D projections to help visualize patterns and separability between classes like WT and OIM.",
                        expanded=False):

                    # ======================================= [ PCA Analysis ] ======================================================================

                    import numpy as np
                    import pandas as pd
                    import streamlit as st
                    from sklearn.decomposition import PCA
                    from sklearn.model_selection import StratifiedKFold, KFold
                    from scipy.signal import savgol_filter
                    from scipy.sparse import diags, eye
                    from scipy.sparse.linalg import spsolve

                    # ---------------- Preprocessing selection (combined) ----------------
                    PREPROCESSING_OPTIONS = ["Mean Center", "Autoscale", "Smoothing", "Normalization", "Baseline",
                                             "Second Derivative"]
                    x_preproc_pca = st.multiselect(
                        "Select Preprocessing Techniques (applied together, in order)",
                        options=PREPROCESSING_OPTIONS,
                        default=["Autoscale"]
                    )

                    # ---- Savitzky–Golay helpers ----
                    def _savgol_params(n_feat: int, window: int, poly: int, deriv: int):
                        w_max_odd = n_feat if (n_feat % 2 == 1) else (n_feat - 1)
                        if w_max_odd < 3:
                            return None
                        w = int(window)
                        if w % 2 == 0: w -= 1
                        if w < 3: w = 3
                        if w > w_max_odd: w = w_max_odd
                        p = int(poly)
                        p = max(p, deriv, 1)
                        if p >= w: p = w - 1
                        return w, p

                    def smooth_savgol(X, window=7, poly=3):
                        Xn = X.values.astype(float) if hasattr(X, "values") else np.asarray(X, dtype=float)
                        Xn = Xn.copy()
                        params = _savgol_params(Xn.shape[1], window, poly, deriv=0)
                        if params is None: return Xn
                        w, p = params
                        return savgol_filter(Xn, window_length=w, polyorder=p, deriv=0, axis=1, mode="interp")


                    def second_derivative_savgol(X, window=7, poly=3):
                        Xn = X.values.astype(float) if hasattr(X, "values") else np.asarray(X, dtype=float)
                        Xn = Xn.copy()
                        params = _savgol_params(Xn.shape[1], window, poly, deriv=2)
                        if params is None: return Xn
                        w, p = params
                        return savgol_filter(Xn, window_length=w, polyorder=p, deriv=2, axis=1, mode="interp")


                    # ---- Baseline (ALS, sparse + fast) ----
                    def als_baseline_sparse(y, lam=1e5, p=0.001, niter=10, eps=1e-12):
                        y = np.asarray(y, dtype=float).ravel()
                        L = y.size
                        if L < 3:
                            return np.zeros_like(y)
                        D = diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L))
                        DTD = lam * (D.T @ D)
                        w = np.ones(L)
                        I = eye(L) * eps
                        for _ in range(niter):
                            W = diags(w, 0, shape=(L, L))
                            z = spsolve(W + DTD + I, w * y)
                            w = p * (y > z) + (1 - p) * (y < z)
                        return z

                    # ---- Fit stats for leakage-safe MC/Autoscale ----
                    def fit_center_scale_stats(X_arr, need_mean: bool, need_std: bool):
                        mu = sd = None
                        if need_mean or need_std:
                            mu = X_arr.mean(axis=0, keepdims=True)
                        if need_std:
                            sd = X_arr.std(axis=0, keepdims=True)
                            sd = sd.copy();
                            sd[sd == 0] = 1.0
                        return mu, sd

                    # ---- Apply the combined preprocessing chain (deterministic steps don't need "fit") ----
                    def apply_preprocessing_chain(
                            X,
                            ops, *,
                            smooth_window=7, smooth_poly=3,
                            normalization_mode="l2",
                            baseline_lam=100_000.0, baseline_p=0.001,
                            mean_=None, std_=None
                    ):
                        Xn = X.values.astype(float) if hasattr(X, "values") else np.asarray(X, dtype=float)
                        Xn = Xn.copy()

                        # 1) Baseline
                        if "Baseline" in ops:
                            Xcorr = np.empty_like(Xn)
                            for i in range(Xn.shape[0]):
                                bl = als_baseline_sparse(Xn[i, :], lam=baseline_lam, p=baseline_p)
                                Xcorr[i, :] = Xn[i, :] - bl
                            Xn = Xcorr

                        # 2) Smoothing / Second Derivative
                        if "Smoothing" in ops and "Second Derivative" in ops:
                            # Prefer derivative if both selected
                            Xn = second_derivative_savgol(Xn, window=smooth_window, poly=smooth_poly)
                        elif "Smoothing" in ops:
                            Xn = smooth_savgol(Xn, window=smooth_window, poly=smooth_poly)
                        elif "Second Derivative" in ops:
                            Xn = second_derivative_savgol(Xn, window=smooth_window, poly=smooth_poly)

                        # 3) Mean Center / Autoscale (column-wise; leakage-safe if mean_/std_ provided)
                        if "Mean Center" in ops or "Autoscale" in ops:
                            mu = mean_ if mean_ is not None else Xn.mean(axis=0, keepdims=True)
                            Xn = Xn - mu
                        if "Autoscale" in ops:
                            sd = std_ if std_ is not None else Xn.std(axis=0, keepdims=True)
                            sd = sd.copy();
                            sd[sd == 0] = 1.0
                            Xn = Xn / sd

                        # 4) Normalization (row-wise)
                        if "Normalization" in ops:
                            mode = normalization_mode.lower()
                            if mode == "l2":
                                denom = np.linalg.norm(Xn, axis=1, keepdims=True)
                            elif mode == "l1":
                                denom = np.sum(np.abs(Xn), axis=1, keepdims=True)
                            else:  # "max"
                                denom = np.max(np.abs(Xn), axis=1, keepdims=True)
                            denom[denom == 0] = 1.0
                            Xn = Xn / denom

                        return Xn

                    # ---- Helper: PCA fit + reconstruction ----
                    def _pca_fit_reconstruct(X_mat, ncomp):
                        pca = PCA(n_components=ncomp, svd_solver="full", random_state=42)
                        T = pca.fit_transform(X_mat)
                        X_hat = pca.inverse_transform(T)
                        return pca, X_hat, T

                    # ========================= Score plots (2D/3D) & controls =========================
                    from sklearn.preprocessing import StandardScaler  # keep if you still need for separate visual plots

                    max_pc = max(2, min(10, X.shape[1]))

                    # (Optional) If you still want a separate plot space using plain StandardScaler:
                    scaler_all_plot = StandardScaler()
                    X_scaled_for_plot = scaler_all_plot.fit_transform(X)
                    ncomp_plot = st.slider("Number of PCs for plotting", 2, max_pc, value=min(5, max_pc), step=1,
                                           key="pca_nplot")
                    pca_plot = PCA(n_components=ncomp_plot, svd_solver="full", random_state=42)
                    T_all = pca_plot.fit_transform(X_scaled_for_plot)

                    # PCA evaluation controls
                    cv_splits_pca = st.slider("CV Splits (PCA)", 3, 15, value=10, step=1)
                    conf_limit_pca = st.selectbox("Confidence Limit (PCA)", [0.90, 0.95, 0.99], index=1)
                    pc_list = st.multiselect(
                        "Ncomp/LVs to evaluate (PCA)",
                        options=list(range(2, max_pc + 1)),
                        default=list(range(2, min(3, max_pc) + 1))
                    )

                    # ========================= CAL & CV with the combined preprocessing =========================
                    rows = []
                    model_id = 1

                    # Build a label for the chosen combination
                    preproc_label = ", ".join(x_preproc_pca) if x_preproc_pca else "None"

                    # ---- Calibration on ALL data using the full chain ----
                    X_all = apply_preprocessing_chain(
                        X, x_preproc_pca,
                        smooth_window=7, smooth_poly=3,
                        baseline_lam=100_000.0, baseline_p=0.001
                    )
                    sst_all = float((X_all ** 2).sum())

                    X_size = f"{X.shape[0]} x {X.shape[1]}"
                    Y_size = "—"

                    for ncomp in pc_list:
                        # CAL
                        pca_all, X_hat_all, _ = _pca_fit_reconstruct(X_all, ncomp)
                        resid_all = X_all - X_hat_all
                        sse_all = float((resid_all ** 2).sum())
                        rmsec = np.sqrt(sse_all / (X.shape[0] * X.shape[1]))
                        r2x_cal = 1.0 - (sse_all / sst_all) if sst_all > 0 else float("nan")
                        bias_cal = float(resid_all.mean())

                        # CV: split and re-apply the same chain with train-only stats for MC/Autoscale
                        if "y" in globals():
                            y_for_split = y
                        elif "y_encoded" in globals():
                            y_for_split = y_encoded
                        else:
                            y_for_split = None

                        if y_for_split is not None:
                            try:
                                folds = StratifiedKFold(n_splits=cv_splits_pca, shuffle=True, random_state=42).split(X,
                                                                                                                     y_for_split)
                            except Exception:
                                folds = KFold(n_splits=cv_splits_pca, shuffle=True, random_state=42).split(X)
                        else:
                            folds = KFold(n_splits=cv_splits_pca, shuffle=True, random_state=42).split(X)

                        sse_cv_total = 0.0
                        sst_cv_total = 0.0
                        bias_cv_accum = 0.0

                        for tr_idx, te_idx in folds:
                            X_tr = X.iloc[tr_idx].values.astype(float)
                            X_te = X.iloc[te_idx].values.astype(float)

                            # Fit stats for MC/Autoscale if selected
                            need_mean = ("Mean Center" in x_preproc_pca) or ("Autoscale" in x_preproc_pca)
                            need_std = ("Autoscale" in x_preproc_pca)
                            mu_tr, sd_tr = fit_center_scale_stats(X_tr, need_mean, need_std)

                            # Apply full chain to train/test with train stats
                            X_tr_s = apply_preprocessing_chain(
                                X_tr, x_preproc_pca,
                                smooth_window=7, smooth_poly=3,
                                baseline_lam=100_000.0, baseline_p=0.001,
                                mean_=mu_tr, std_=sd_tr
                            )
                            X_te_s = apply_preprocessing_chain(
                                X_te, x_preproc_pca,
                                smooth_window=7, smooth_poly=3,
                                baseline_lam=100_000.0, baseline_p=0.001,
                                mean_=mu_tr, std_=sd_tr
                            )

                            pca_cv = PCA(n_components=ncomp, svd_solver="full", random_state=42).fit(X_tr_s)
                            X_te_hat = pca_cv.inverse_transform(pca_cv.transform(X_te_s))

                            resid_te = X_te_s - X_te_hat
                            sse_cv_total += float((resid_te ** 2).sum())
                            sst_cv_total += float((X_te_s ** 2).sum())
                            bias_cv_accum += float(resid_te.mean())

                        rmsecv = np.sqrt(sse_cv_total / (X.shape[0] * X.shape[1]))
                        r2x_cv = 1.0 - (sse_cv_total / sst_cv_total) if sst_cv_total > 0 else float("nan")
                        bias_cv = bias_cv_accum / cv_splits_pca
                        rmse_ratio = rmsecv / rmsec if rmsec > 0 else float("nan")

                        rows.append({
                            "CV Splits": cv_splits_pca,
                            "Confidence Limit": conf_limit_pca,
                            "RMSE Ratio (RMSECV/RMSEC)": rmse_ratio,
                            "RMSEC (Cal)": rmsec,
                            "RMSECV (CV)": rmsecv,
                            "Model Name": f"Model {model_id}",
                            "Ncomp/LVs": ncomp,
                            "X-Preprocessing": preproc_label,
                            "X Include Size": X_size,
                            "Y Include Size": Y_size,
                            "R2C (Cal)": r2x_cal,  # R²X on calibration
                            "R2CV (CV)": r2x_cv,  # R²X on CV
                            "Bias": bias_cal,
                            "Bias (CV)": bias_cv
                        })
                        model_id += 1

                    # ---------------- Table + download ----------------
                    pca_calcv_df = pd.DataFrame(rows, columns=[
                        "CV Splits", "Confidence Limit", "RMSE Ratio (RMSECV/RMSEC)",
                        "RMSEC (Cal)", "RMSECV (CV)", "Model Name", "Ncomp/LVs",
                        "X-Preprocessing", "X Include Size", "Y Include Size",
                        "R2C (Cal)", "R2CV (CV)", "Bias", "Bias (CV)"
                    ])

                    fmt_cols = ["RMSE Ratio (RMSECV/RMSEC)", "RMSEC (Cal)", "RMSECV (CV)", "R2C (Cal)", "R2CV (CV)",
                                "Bias", "Bias (CV)"]

                    st.markdown("### 📋 PCA Analysis Result Summary")
                    st.dataframe(
                        pca_calcv_df.style
                        .format({c: "{:.4f}" for c in fmt_cols})
                        .background_gradient(subset=["R2CV (CV)", "R2C (Cal)"], cmap="YlGn")
                        .background_gradient(subset=["RMSECV (CV)", "RMSEC (Cal)"], cmap="YlOrRd_r"),
                        use_container_width=True
                    )

                    st.download_button(
                        "📥 Download PCA Summary (CSV)",
                        data=pca_calcv_df.to_csv(index=False).encode("utf-8"),
                        file_name="pca_summary_table.csv",
                        mime="text/csv"
                    )

                    ########################################################################################################

                    # ---- Controls matching PLS/PLS-DA style ----
                    from sklearn.decomposition import PCA
                    from sklearn.model_selection import StratifiedKFold
                    from sklearn.preprocessing import StandardScaler

                    max_pc = max(2, min(10, X.shape[1]))

                    # Pre-scale once for calibration pass
                    scaler_all = StandardScaler()
                    X_scaled_all = scaler_all.fit_transform(X)
                    sst_all = float((X_scaled_all ** 2).sum())

                    # ========================= Score plots (2D/3D) =========================
                    # (Removed UI sliders/selects; use defaults)
                    ncomp_plot = min(5, max_pc)
                    pca_plot = PCA(n_components=ncomp_plot)
                    T_all = pca_plot.fit_transform(X_scaled_all)

                    # default confidence limit for ellipses (since UI removed)
                    conf_limit_pca = 0.95

                    # Fit PCA
                    pca = PCA()
                    X_pca_all = pca.fit_transform(X)
                    explained_var = pca.explained_variance_ratio_
                    eigenvalues = pca.explained_variance_

                    # Create PCA dataframe
                    df_pca_all = pd.DataFrame(X_pca_all, columns=[f"PC{i + 1}" for i in range(X_pca_all.shape[1])])
                    df_pca_all['Label'] = y.values

                    # Build plotting DF (include all PCs to pick axes)
                    pc_cols = [f"PC{i + 1}" for i in range(ncomp_plot)]
                    df_pca_all = pd.DataFrame(T_all, columns=pc_cols)
                    df_pca_all["Label"] = y.values

                    # Confidence ellipses
                    chi2_map = {0.90: 4.605, 0.95: 5.991, 0.99: 9.210}
                    chi2_val = chi2_map[conf_limit_pca]


                    def _ellipse_xy(points_2d, chi2_q, npts=200):
                        pts = np.asarray(points_2d)
                        if pts.shape[0] < 3:
                            return None, None
                        mu = pts.mean(axis=0)
                        S = np.cov(pts.T)
                        vals, vecs = np.linalg.eigh(S)
                        order = np.argsort(vals)[::-1]
                        vals, vecs = vals[order], vecs[:, order]
                        radii = np.sqrt(np.maximum(vals, 1e-12) * chi2_q)
                        t = np.linspace(0, 2 * np.pi, npts)
                        circ = np.stack([np.cos(t) * radii[0], np.sin(t) * radii[1]], axis=0)
                        ell = (vecs @ circ).T + mu
                        return ell[:, 0], ell[:, 1]


                    # Axis selectors
                    col_ax1, col_ax2, col_ax3 = st.columns(3)
                    with col_ax1:
                        pc_x = st.selectbox("X-axis (scores)", pc_cols, index=0, key="pca_x")
                    with col_ax2:
                        pc_y = st.selectbox("Y-axis (scores)", pc_cols, index=1, key="pca_y")
                    with col_ax3:
                        pc_z = st.selectbox("Z-axis (scores)", pc_cols, index=min(2, len(pc_cols) - 1), key="pca_z")

                    x_idx = int(pc_x.replace("PC", "")) - 1
                    y_idx = int(pc_y.replace("PC", "")) - 1
                    z_idx = int(pc_z.replace("PC", "")) - 1

                    # 2D plot
                    st.subheader("📉 PCA 2D Projection")
                    fig_2d = px.scatter(df_pca_all, x=pc_x, y=pc_y, color="Label",
                                        title=f"PCA Projection ({pc_x} vs {pc_y})")
                    for lab in df_pca_all["Label"].unique():
                        d = df_pca_all[df_pca_all["Label"] == lab][[pc_x, pc_y]].values
                        ex, ey = _ellipse_xy(d, chi2_val)
                        if ex is not None:
                            fig_2d.add_trace(
                                go.Scatter(x=ex, y=ey, mode="lines",
                                           name=f"{lab} {int(conf_limit_pca * 100)}% ellipse",
                                           line=dict(width=1), showlegend=False)
                            )
                    st.plotly_chart(fig_2d, use_container_width=True)

                    # 3D plot
                    st.subheader("🧭 PCA 3D Visualization")
                    if len(pc_cols) >= 3:
                        fig_3d = px.scatter_3d(df_pca_all, x=pc_x, y=pc_y, z=pc_z, color="Label",
                                               title=f"3D PCA: {pc_x} vs {pc_y} vs {pc_z}",
                                               height=520, width=800)
                        st.plotly_chart(fig_3d, use_container_width=True)
                    else:
                        st.info("Add ≥3 PCs to view the 3D projection.")

                    pca_summary = pd.DataFrame({
                        "PC": [f"PC{i + 1}" for i in range(len(explained_var))],
                        "Eigenvalue of Cov(X)": eigenvalues,
                        "% Variance This PC": explained_var * 100,
                        "% Variance Cumulative": np.cumsum(explained_var) * 100
                    })

                    ################################# display the pca table ######################
                    st.dataframe(pca_summary, use_container_width=True, height=350)

                    ##### PCA Loadings with Bond Annotation #####

                    st.subheader("📊 PCA Loadings")
                    # Compute PCA Loadings
                    # loadings = pd.DataFrame(pca.components_.T, index=selected_columns,
                    #                         columns=[f"PC{i + 1}" for i in range(pca.n_components_)])

                    # --- NEW: get the exact feature names PCA was fit on ---
                    feature_cols = list(X.columns)  # must match n_features of pca.components_
                    selected_columns = feature_cols  # keep your later code that uses selected_columns

                    # Safety check (optional but helpful during debugging)
                    assert pca.components_.shape[1] == len(feature_cols), \
                        f"Mismatch: PCA features={pca.components_.shape[1]}, labels={len(feature_cols)}"

                    # --- Build loadings with the correct index ---
                    loadings = pd.DataFrame(
                        pca.components_.T,
                        index=feature_cols,
                        columns=[f"PC{i + 1}" for i in range(pca.components_.shape[0])]
                    )

                    # User Selects PC to View Loadings
                    selected_pc_loading = st.selectbox("Select PC to view Loadings:", loadings.columns)

                    st.markdown(f"### 🔑 Loadings Plot for {selected_pc_loading}")

                    ###========================== line plot of loading
                    #
                    # fig_load1 = go.Figure()
                    # fig_load1.add_trace(go.Scatter(
                    #     x=selected_columns,
                    #     y=loadings[selected_pc_loading],
                    #     mode="lines",
                    #     line=dict(color="blue"),
                    #     name=f"{selected_pc_loading} Loadings"
                    # ))
                    # fig_load1.add_hline(y=0, line=dict(color="gray", dash="dash"))  # Zero reference line

                    # ===== Line plot of loadings (use feature_cols on x) =====
                    fig_load1 = go.Figure()
                    fig_load1.add_trace(go.Scatter(
                        x=feature_cols,
                        y=loadings[selected_pc_loading],
                        mode="lines",
                        line=dict(color="blue"),
                        name=f"{selected_pc_loading} Loadings"
                    ))
                    fig_load1.add_hline(y=0, line=dict(color="gray", dash="dash"))  # Zero reference line



                    # ✅ Add Band Annotations
                    band_annotations1 = {
                        1740: "C=O Stretch (Lipid)",
                        1650: "Amide I (Protein)",
                        1550: "Amide II (Protein)",
                        1338: "CH₂ (Collagen)",
                        1200: "C–O Stretch (Matrix)",
                        1115: "HPO₄²⁻ (Mineral)",
                        1060: "Sugar Ring C–O",
                        1030: "PO₄³⁻ (Mineral)",
                        875: "CO₃²⁻ Bending",
                        856: "C–S Proteoglycan",
                        7000: "O–H Stretch",
                        6688: "N–H Stretch",
                        5800: "CH₂ Lipid",
                        5200: "O–H Water",
                    }

                    for wn, label in band_annotations1.items():
                        if selected_range[0] <= wn <= selected_range[1]:
                            fig_load1.add_vline(x=wn, line=dict(color="gray", width=1, dash="dash"))
                            fig_load1.add_annotation(x=wn, y=loadings[selected_pc_loading].max(), text=label,
                                                     showarrow=True, arrowhead=1, yshift=10, font=dict(size=9))

                    fig_load1.update_layout(
                        title=f"PCA Loadings for {selected_pc_loading} ({explained_var[int(selected_pc_loading[-1]) - 1] * 100:.2f}% Variance)",
                        xaxis_title="Wavenumber (cm⁻¹)",
                        yaxis_title="Loading Score",
                        height=400
                    )
                    st.plotly_chart(fig_load1, use_container_width=True)

                    ###========================== line plot of loading

                    fig_load = go.Figure()
                    fig_load.add_trace(go.Scatter(
                        x=selected_columns,
                        y=loadings[selected_pc_loading],
                        mode="lines",
                        line=dict(color="blue"),
                        name=f"{selected_pc_loading} Loadings"
                    ))
                    fig_load.add_hline(y=0, line=dict(color="gray", dash="dash"))  # Zero reference line

                    # ✅ Bond Annotation Mapping Table
                    bond_annotations_table = [
                        (8500, "O–H Stretching and Bending (Water)"),
                        (7000, "O–H Stretching (Water)"),
                        (6688, "N–H Stretching (Protein/Collagen)"),
                        (5800, "CH₂ Stretching (Lipid)"),
                        (5200, "O–H Stretching and Bending (Water)"),
                        (4890, "N–H Bending (Protein/Collagen)"),
                        (4610, "C–H Stretching & Deformation (Protein/Collagen)"),
                        (4310, "Sugar Ring Vibrations (Proteoglycan)"),
                        ((3600, 3200), "O–H Stretching (Water/Hydroxyl)"),
                        ((3500, 3300), "N–H Stretching (Proteins)"),
                        ((3000, 2800), "C–H Stretching (Lipids, CH₂)"),
                        ((1750, 1650), "C=O Stretching (Proteins/Lipids)"),
                        (1550, "Amide II (Proteins)"),
                        (1338, "CH₂ Side Chain Bending (Collagen)"),
                        ((1100, 900), "PO₄³⁻ Stretching (Bone Mineral)"),
                        ((890, 850), "CO₃²⁻ Bending (Carbonate)"),
                        (1740, "C=O Stretching (Ester, Lipids)"),
                        (1650, "Amide I (Proteins)"),
                        (1630, "Water O–H Bending (Water)"),
                        ((1200, 1000), "C–O Stretching (Alcohols/Ethers)"),
                        (1115, "HPO₄²⁻ Stretching (Bone mineral)"),
                        (1060, "Sugar Ring C–O Stretch (Carbohydrates)"),
                        (1030, "PO₄³⁻ Stretching (Bone Mineral)"),
                        (875, "CO₃²⁻ Bending (Carbonates)"),
                        (856, "C–S Bending (Proteoglycans)"),
                        ((900, 800), "Aromatic C–H Bending (Fingerprint region)")
                    ]


                    # ✅ Function to Assign Bond Annotation
                    def get_bond_annotation(wavenumber):
                        # Check ranges first
                        for entry in bond_annotations_table:
                            if isinstance(entry[0], tuple):  # Range
                                low, high = entry[0]
                                if high <= wavenumber <= low:
                                    return entry[1]
                        # Exact match
                        for entry in bond_annotations_table:
                            if not isinstance(entry[0], tuple) and entry[0] == int(wavenumber):
                                return entry[1]
                        # Nearest match if no exact or range match
                        all_wavenumbers = [
                            val if isinstance(val, int) else sum(val) // 2 for val, _ in bond_annotations_table
                        ]
                        closest_idx = np.argmin([abs(wavenumber - wn) for wn in all_wavenumbers])
                        return bond_annotations_table[closest_idx][1]


                    ########################################################################################################################

                    # ✅ Variable Importance with Bond Annotations in One Table
                    variable_importance = loadings.abs()[selected_pc_loading].sort_values(ascending=False)
                    importance_df = pd.DataFrame(variable_importance.head(20)).reset_index()
                    importance_df.columns = ["Wavenumber (cm⁻¹)", "Importance"]


                    # ✅ Safely extract numeric value (handle cases like '1030/1240')
                    def extract_first_numeric(value):
                        value_str = str(value).strip()
                        if "/" in value_str:  # Take first part if it's a ratio
                            value_str = value_str.split("/")[0].strip()
                        try:
                            return float(value_str)
                        except ValueError:
                            return None  # In case there's any unexpected text


                    importance_df["Numeric Wavenumber"] = importance_df["Wavenumber (cm⁻¹)"].apply(
                        extract_first_numeric)

                    # ✅ Drop rows where numeric conversion failed (if any)
                    importance_df = importance_df.dropna(subset=["Numeric Wavenumber"])

                    # ✅ Assign bond annotations
                    importance_df["Band Annotation"] = importance_df["Numeric Wavenumber"].apply(
                        get_bond_annotation)

                    # ======================= PCA Loadings — Dynamic Insight & Reasoning (ADD-ON v2) =======================
                    import re
                    import numpy as np
                    import pandas as pd
                    import streamlit as st


                    # ---------- 0) Helpers ----------
                    def _pc_index(name):
                        m = re.search(r"\d+", str(name))
                        return max(int(m.group()) - 1, 0) if m else 0


                    def _num(val):
                        try:
                            return float(str(val).split("/")[0])
                        except Exception:
                            return np.nan


                    def _smooth(y, win=7):
                        # simple moving average, odd window
                        win = int(win)
                        if win < 1: return y
                        if win % 2 == 0: win += 1
                        if win == 1: return y
                        k = np.ones(win, dtype=float) / win
                        pad = win // 2
                        ypad = np.pad(np.asarray(y, dtype=float), (pad, pad), mode="edge")
                        return np.convolve(ypad, k, mode="valid")


                    def _local_extrema(y, order=3, mode="max"):
                        # naive local peaks (mode 'max' for peaks, 'min' for dips)
                        y = np.asarray(y, dtype=float)
                        idxs = []
                        n = len(y)
                        for i in range(order, n - order):
                            seg = y[i - order:i + order + 1]
                            if mode == "max":
                                if y[i] == seg.max() and (y[i] > seg[:order].max()) and (
                                        y[i] > seg[order + 1:].max()):
                                    idxs.append(i)
                            else:  # min
                                if y[i] == seg.min() and (y[i] < seg[:order].min()) and (
                                        y[i] < seg[order + 1:].min()):
                                    idxs.append(i)
                        return idxs


                    def _favor_class(value, class_pos, class_neg):
                        if class_pos is None or class_neg is None:
                            return ""
                        return f"favors **{class_pos}**" if value > 0 else f"favors **{class_neg}**"


                    # ---------- 1) Selected PC, variance, scores ----------
                    pc_idx = _pc_index(selected_pc_loading)
                    try:
                        var_pct = float(explained_var[pc_idx]) * 100.0
                    except Exception:
                        var_pct = np.nan

                    try:
                        pc_scores = X_pca_all[:, pc_idx]
                    except Exception:
                        pc_scores = pca.transform(X)[:, pc_idx]

                    # ---------- 2) Determine which class is on + / − side ----------
                    try:
                        df_scores = pd.DataFrame({"Label": np.asarray(y), "PC_Score": pc_scores})
                        class_means = df_scores.groupby("Label", observed=True)["PC_Score"].mean().sort_values(
                            ascending=False)
                        class_pos = class_means.index[0] if len(class_means) else None
                        class_neg = class_means.index[-1] if len(class_means) else None
                    except Exception:
                        class_pos = class_neg = None

                    # ---------- 3) Create numeric wavenumber axis & focus range ----------
                    try:
                        wn_labels = list(loadings.index)
                    except Exception:
                        wn_labels = list(selected_columns)  # fallback
                    wn_numeric = np.array([_num(v) for v in wn_labels], dtype=float)

                    try:
                        lo, hi = float(selected_range[0]), float(selected_range[1])
                        focus_mask = (wn_numeric >= min(lo, hi)) & (wn_numeric <= max(lo, hi))
                        if not np.any(focus_mask):
                            focus_mask = np.isfinite(wn_numeric)
                    except Exception:
                        focus_mask = np.isfinite(wn_numeric)

                    s_full = pd.to_numeric(loadings[selected_pc_loading], errors="coerce").values
                    s_focus = s_full[focus_mask]
                    wn_focus = wn_numeric[focus_mask]
                    lbl_focus = np.array(wn_labels, dtype=object)[focus_mask]

                    if len(s_focus) < 5 or np.all(~np.isfinite(s_focus)):
                        st.warning("Not enough points in the current range to analyze loadings.")
                        st.stop()

                    # ---------- 4) Controls for detection (defaults) ----------
                    top_k = min(12, len(s_focus))
                    smooth_win = 7
                    half_width_cm = 25

                    y_sm = _smooth(s_focus, win=smooth_win)

                    # ---------- 5) Peak & dip detection ----------
                    order = max(2, smooth_win // 2)
                    pos_idx = _local_extrema(y_sm, order=order, mode="max")
                    neg_idx = _local_extrema(y_sm, order=order, mode="min")

                    cand = [(i, y_sm[i]) for i in pos_idx] + [(i, y_sm[i]) for i in neg_idx]
                    if not cand:
                        cand = list(enumerate(y_sm))
                    cand_sorted = sorted(cand, key=lambda t: abs(t[1]), reverse=True)[:top_k]

                    # ---------- 6) Build "Top drivers" table ----------
                    rows = []
                    for i, amp in cand_sorted:
                        wn = wn_focus[i]
                        label = lbl_focus[i]
                        try:
                            band = get_bond_annotation(float(wn)) if np.isfinite(wn) else ""
                        except Exception:
                            band = ""
                        rows.append({
                            "Wavenumber (cm⁻¹)": str(label),
                            "Numeric (cm⁻¹)": wn if np.isfinite(wn) else np.nan,
                            "Loading": float(s_focus[i]),
                            "|Loading|": float(abs(s_focus[i])),
                            "Band Annotation": band,
                            "Interpretation": _favor_class(s_focus[i], class_pos, class_neg)
                        })

                    drivers_df = (pd.DataFrame(rows)
                                  .sort_values("|Loading|", ascending=False)
                                  .reset_index(drop=True))

                    # ===== 📊 Variable Importance with Bond Annotations + Download (one line) =====
                    col1, col2 = st.columns([4, 1])

                    with col1:
                        st.markdown("### 📊 Variable Importance with Bond Annotations")

                    with col2:
                        var_importance_csv = importance_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "📥 Download CSV",
                            data=var_importance_csv,
                            file_name="variable_importance_with_bands.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                    # ===== Display table =====
                    st.dataframe(importance_df, use_container_width=True)

                    ########################## Most influenced wavenumbers ###############################
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown("### ⭐ Most influential wavenumbers (by |loading|)")
                    with col2:
                        st.download_button(
                            "📥 Download CSV",
                            data=drivers_df[
                                ["Wavenumber (cm⁻¹)", "Loading", "|Loading|", "Band Annotation", "Interpretation"]
                            ].to_csv(index=False).encode("utf-8"),
                            file_name=f"most_influential_wavenumbers_{selected_pc_loading}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                    st.dataframe(
                        drivers_df[
                            ["Wavenumber (cm⁻¹)", "Loading", "|Loading|", "Band Annotation", "Interpretation"]
                        ].style.format({"Loading": "{:.4f}", "|Loading|": "{:.4f}"}),
                        use_container_width=True
                    )

                    ################################################# [  Auto-detected regions ] ###############################################################################

                    # ---------- 7) Auto-detected regions around each peak ----------
                    def _region_for(center_cm1):
                        lo = center_cm1 - half_width_cm
                        hi = center_cm1 + half_width_cm
                        return lo, hi


                    regs = []
                    for _, r in drivers_df.dropna(subset=["Numeric (cm⁻¹)"]).iterrows():
                        c = float(r["Numeric (cm⁻¹)"])
                        lo, hi = _region_for(c)
                        regs.append([lo, hi, r["Wavenumber (cm⁻¹)"], r["Band Annotation"]])

                    regs.sort(key=lambda x: x[0])
                    merged = []
                    for lo, hi, lbl, ann in regs:
                        if not merged or lo > merged[-1][1]:
                            merged.append([lo, hi, [lbl], [ann]])
                        else:
                            merged[-1][1] = max(merged[-1][1], hi)
                            merged[-1][2].append(lbl)
                            merged[-1][3].append(ann)

                    reg_rows = []
                    for lo, hi, lbls, anns in merged:
                        m = (wn_focus >= lo) & (wn_focus <= hi)
                        if not np.any(m):
                            continue
                        net = float(np.nansum(s_focus[m]))
                        mass = float(np.nansum(np.abs(s_focus[m])))
                        lbl_text = ", ".join(map(str, lbls[:3])) + ("..." if len(lbls) > 3 else "")
                        ann_text = ", ".join(sorted(set(a for a in anns if a)))
                        reg_rows.append({
                            "Region (cm⁻¹)": f"{int(lo)}–{int(hi)}",
                            "|Σ loading|": mass,
                            "Net sign": "Positive" if net > 0 else ("Negative" if net < 0 else "≈0"),
                            "Peak labels": lbl_text,
                            "Likely bands": ann_text or "—",
                        })

                    contrib_df = (pd.DataFrame(reg_rows)
                                  .sort_values("|Σ loading|", ascending=False)
                                  .reset_index(drop=True))

                    ###########################################################################################################################################
                    # --- Ensure lv_idx / var_pct / class_pos / class_neg available for the summary ---
                    try:
                        lv_idx
                    except NameError:
                        import re

                        m = re.search(r"\d+", str(selected_pc_loading))
                        lv_idx = max(int(m.group()) - 1, 0) if m else 0

                    try:
                        var_pct
                    except NameError:
                        # explained_x should be a normalized ratio array for PLS LVs (sum≈1). If missing, compute from scores.
                        explained_x_attr = getattr(pls, "x_explained_variance_ratio_", None)
                        if explained_x_attr is not None:
                            explained_x = np.asarray(explained_x_attr, dtype=float)
                        else:
                            T_all = getattr(pls, "x_scores_", None)
                            if T_all is not None and np.sum(np.var(T_all, axis=0)) > 0:
                                raw_var = np.var(T_all, axis=0)
                                explained_x = raw_var / np.sum(raw_var)
                            else:
                                explained_x = np.ones(pls.x_loadings_.shape[1], dtype=float) / pls.x_loadings_.shape[1]
                        var_pct = float(explained_x[min(lv_idx, len(explained_x) - 1)] * 100.0)

                    # If class context wasn’t computed earlier, build it from PLS scores
                    try:
                        class_pos, class_neg
                    except NameError:
                        try:
                            lv_scores = pls.x_scores_[:, lv_idx]
                            y_labels = y.values if hasattr(y, "values") else np.asarray(y)
                            _df_tmp = pd.DataFrame({"Label": y_labels, "LV_Score": lv_scores})
                            class_means = _df_tmp.groupby("Label", observed=True)["LV_Score"].mean().sort_values(
                                ascending=False)
                            class_pos = class_means.index[0] if len(class_means) else None
                            class_neg = class_means.index[-1] if len(class_means) else None
                        except Exception:
                            class_pos = class_neg = None

                    # ---------- Narrative summary (same structure as PCA, PLS wording/vars) ----------
                    pos_mass = float(np.nansum(np.abs(s_focus[s_focus > 0])))
                    neg_mass = float(np.nansum(np.abs(s_focus[s_focus < 0])))
                    contrast = ("Positive-weighted regions dominate"
                                if pos_mass > neg_mass else
                                (
                                    "Negative-weighted regions dominate" if neg_mass > pos_mass else "Balanced positive/negative weights"))

                    summary = []
                    summary.append(f"- **{selected_pc_loading}** explains **{var_pct:.2f}%** of total variance.")
                    summary.append(f"- **Contrast:** {contrast} in the current range.")
                    if class_pos and class_neg:
                        summary.append(
                            f"- **Class context:** Higher LV scores (positive side) align with **{class_pos}**; "
                            f"lower scores (negative side) align with **{class_neg}**."
                        )
                    if len(contrib_df):
                        top_reg = contrib_df.iloc[0]
                        summary.append(
                            f"- **Strongest region:** {top_reg['Region (cm⁻¹)']} "
                            f"({top_reg['Net sign']}, |Σloading|={top_reg['|Σ loading|']:.4f}); "
                            f"likely bands: {top_reg['Likely bands'] or '—'}."
                        )

                    head_n = min(6, len(drivers_df))
                    bullets = []
                    for _, r in drivers_df.head(head_n).iterrows():
                        side = "positive" if r["Loading"] > 0 else "negative"
                        interp = f" — {r['Interpretation']}" if r["Interpretation"] else ""
                        bullets.append(
                            f"  • {r['Wavenumber (cm⁻¹)']}: **{side}** loading ({r['Band Annotation']}){interp}"
                        )

                    st.markdown("### 📝 Auto-summary")
                    st.markdown("\n".join(summary + ["- **Key drivers:**"] + bullets))

                    st.caption(
                        "Notes: (1) PLS LV axis sign is arbitrary; interpretation uses relative +/− sides and class means. "
                        "(2) Smoothing only aids peak detection; it doesn’t change the underlying loadings. "
                        "(3) Preprocessing (e.g., autoscale) affects which bands dominate."
                    )

                    # (Optional correlation block can be added below, identical to your PCA version but using lv_scores if needed.)
                    ###########################################################################################################################################

                    # ---------- 9) Optional correlation ----------
                    try:
                        pos_first = drivers_df.loc[drivers_df["Loading"] > 0].iloc[0]
                        neg_first = drivers_df.loc[drivers_df["Loading"] < 0].iloc[0]
                        pos_col = pos_first["Wavenumber (cm⁻¹)"]
                        neg_col = neg_first["Wavenumber (cm⁻¹)"]

                        all_idx = pd.Series(wn_focus, index=lbl_focus)
                        pos_label = str(pos_col)
                        neg_label = str(neg_col)
                        if hasattr(X, "columns"):
                            if pos_label in X.columns:
                                x_pos = pd.to_numeric(X[pos_label], errors="coerce")
                            else:
                                pn = _num(pos_label)
                                diffs = pd.Series([abs(_num(c) - pn) for c in X.columns], index=X.columns)
                                x_pos = pd.to_numeric(X[diffs.idxmin()], errors="coerce")

                            if neg_label in X.columns:
                                x_neg = pd.to_numeric(X[neg_label], errors="coerce")
                            else:
                                nn = _num(neg_label)
                                diffs = pd.Series([abs(_num(c) - nn) for c in X.columns], index=X.columns)
                                x_neg = pd.to_numeric(X[diffs.idxmin()], errors="coerce")

                            ratio = (x_pos / x_neg).replace([np.inf, -np.inf], np.nan)
                            valid = ratio.notna() & np.isfinite(pc_scores)
                            if valid.any():
                                corr = float(np.corrcoef(ratio[valid], pd.Series(pc_scores)[valid])[0, 1])
                                st.caption(
                                    f"Correlation proxy (PC score vs I({pos_label})/I({neg_label})): **{corr:.3f}** "
                                    f"(expect positive if the positive-weighted region truly drives PC).")
                    except Exception:
                        pass

                    # # ---------- 10) Downloads ----------
                    out_txt = []
                    out_txt.append(f"Selected: {selected_pc_loading}")
                    out_txt.append(f"Explained variance: {var_pct:.2f}%")
                    out_txt.append(f"Class (+): {class_pos} | Class (−): {class_neg}")
                    out_txt.append("Top drivers (by |loading|):")
                    out_txt.append(drivers_df.to_string(index=False))
                    if len(contrib_df):
                        out_txt.append("\nAuto-detected regions:")
                        out_txt.append(contrib_df.to_string(index=False))


                    ############################################[  model performance summary ]#########################################################################
                    st.markdown("### 📊 Model Performance Summary")

                    run_pc_clf = st.checkbox("📊 Model Performance Summary for selected PC", value=True, key="pca_run_selected_pc")

                    # ============================ Run Models on the selected PC ============================
                    if run_pc_clf:
                        # --- choose 1+ models (all selected by default) ---
                        model_choices_pc = ["SVM", "Logistic Regression", "KNN", "Random Forest", "XGBoost"]
                        selected_models_pc = st.multiselect(
                            "Select classification models (trained on the selected PC score)",
                            options=model_choices_pc,
                            default=model_choices_pc,
                            key="pca_selected_pc_models"
                        )

                        # Parse selected PC index (e.g., "PC3" -> 2)
                        sel_pc_idx = int(str(selected_pc_loading).replace("PC", "")) - 1

                        # --- NEW: single vs cumulative toggle ---
                        mode = st.radio(
                            "PC feature mode",
                            ["Single (selected only)", "Cumulative (PC1..selected)"],
                            index=1,
                            key="pc_feature_mode"
                        )

                        # Build feature matrix X_pc based on mode
                        if mode == "Single (selected only)":
                            X_pc = X_pca_all[:, sel_pc_idx].reshape(-1, 1)
                            pc_label = f"{selected_pc_loading}"
                        else:
                            X_pc = X_pca_all[:, :sel_pc_idx + 1]
                            pc_label = f"PC1–PC{sel_pc_idx + 1}"

                        # Encode labels locally to keep ordering consistent for the report & matrix
                        from sklearn.preprocessing import LabelEncoder

                        le_local = LabelEncoder()
                        y_codes_local = le_local.fit_transform(y)

                        # Train/test split (stratified)
                        from sklearn.model_selection import train_test_split

                        X_train_pc, X_test_pc, y_train_pc, y_test_pc = train_test_split(
                            X_pc, y_codes_local, test_size=0.2, stratify=y_codes_local, random_state=42
                        )

                        # Helper to build a model by name
                        from sklearn.svm import SVC
                        from sklearn.linear_model import LogisticRegression
                        from sklearn.neighbors import KNeighborsClassifier
                        from sklearn.ensemble import RandomForestClassifier
                        from xgboost import XGBClassifier
                        from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, \
                            confusion_matrix
                        import numpy as np
                        import pandas as pd
                        import seaborn as sns
                        import matplotlib.pyplot as plt


                        def _make_model_pc(name):
                            if name == "SVM":
                                return SVC()
                            if name == "Logistic Regression":
                                return LogisticRegression(max_iter=1000)
                            if name == "KNN":
                                return KNeighborsClassifier(n_neighbors=3)
                            if name == "Random Forest":
                                return RandomForestClassifier(n_estimators=100)
                            # XGBoost
                            return XGBClassifier(use_label_encoder=False, eval_metric="logloss")


                        # Run all selected models
                        results_pc = []
                        for mdl_name in selected_models_pc:
                            clf_pc = _make_model_pc(mdl_name)
                            clf_pc.fit(X_train_pc, y_train_pc)
                            y_pred_pc = clf_pc.predict(X_test_pc)

                            # Metrics (weighted for multi-class safety)
                            acc_pc = clf_pc.score(X_test_pc, y_test_pc)
                            prec_pc = precision_score(y_test_pc, y_pred_pc, average="weighted", zero_division=0)
                            rec_pc = recall_score(y_test_pc, y_pred_pc, average="weighted", zero_division=0)
                            f1_pc = f1_score(y_test_pc, y_pred_pc, average="weighted", zero_division=0)

                            results_pc.append({
                                "Model": mdl_name,
                                "PC used": pc_label,  # <-- updated label to reflect mode
                                "Accuracy": acc_pc,
                                "Precision (weighted)": prec_pc,
                                "Recall (weighted)": rec_pc,
                                "F1-Score (weighted)": f1_pc
                            })

                        # ===== 📊 Model Performance Summary (selected PC) + Download (one line) =====
                        col1, col2 = st.columns([4, 1])

                        with col1:
                            st.markdown("### 📊 Model Performance Summary (selected PC)")

                        with col2:
                            st.download_button(
                                "📥 Download CSV",
                                data=pd.DataFrame(results_pc).to_csv(index=False).encode("utf-8"),
                                file_name="pca_selected_pc_model_metrics.csv",
                                mime="text/csv",
                                use_container_width=True
                            )

                        results_pc_df = pd.DataFrame(results_pc)

                        st.dataframe(
                            results_pc_df.style
                            .background_gradient(cmap="YlGnBu")
                            .format({
                                "Accuracy": "{:.3f}",
                                "Precision (weighted)": "{:.3f}",
                                "Recall (weighted)": "{:.3f}",
                                "F1-Score (weighted)": "{:.3f}"
                            }),
                            use_container_width=True
                        )

                        # ---- Detailed results per model ----
                        for mdl_name in selected_models_pc:
                            st.markdown(f"### 🔎 Details: {mdl_name} on {pc_label}")  # <-- label reflects mode
                            report_col_pc, matrix_col_pc = st.columns([2, 3])

                            # Rebuild + predict
                            clf_pc = _make_model_pc(mdl_name)
                            clf_pc.fit(X_train_pc, y_train_pc)
                            y_pred_pc = clf_pc.predict(X_test_pc)

                            with report_col_pc:
                                st.markdown("**Classification Report**")
                                report_dict_pc = classification_report(
                                    y_test_pc, y_pred_pc, target_names=list(le_local.classes_), output_dict=True,
                                    zero_division=0
                                )
                                report_df_pc = pd.DataFrame(report_dict_pc).transpose()
                                styled_report_pc = report_df_pc.style.set_table_styles([
                                    {'selector': 'th', 'props': [('border', '1px solid black'), ('padding', '8px'),
                                                                 ('background', '#f2f2f2')]},
                                    {'selector': 'td', 'props': [('border', '1px solid black'), ('padding', '8px')]},
                                    {'selector': 'tr', 'props': [('border', '1px solid black')]}
                                ]).format("{:.2f}")
                                st.write(styled_report_pc)

                            with matrix_col_pc:
                                # Row-normalized confusion matrix -> percentages
                                cm_pc = confusion_matrix(y_test_pc, y_pred_pc, labels=np.arange(len(le_local.classes_)))
                                with np.errstate(divide="ignore", invalid="ignore"):
                                    cm_pct_pc = cm_pc.astype(float) / cm_pc.sum(axis=1, keepdims=True) * 100
                                cm_pct_pc = np.nan_to_num(cm_pct_pc)

                                annot_labels_pc = np.array([[f"{v:.1f}%" for v in row] for row in cm_pct_pc])

                                fig_cm_pc, ax_pc = plt.subplots()
                                sns.heatmap(
                                    cm_pct_pc,
                                    annot=annot_labels_pc,
                                    fmt="",
                                    cmap="Blues",
                                    vmin=0, vmax=100,
                                    xticklabels=list(le_local.classes_),
                                    yticklabels=list(le_local.classes_),
                                    cbar_kws={"label": "% of true class"},
                                    ax=ax_pc
                                )
                                ax_pc.set_xlabel("Predicted")
                                ax_pc.set_ylabel("True")
                                ax_pc.set_title(
                                    f"{mdl_name} — Confusion Matrix (%) on {pc_label}")  # <-- label reflects mode
                                st.pyplot(fig_cm_pc)

                    # ============================== One-file export for the entire "🔬 PCA: Principal Component Projection" ==============================

                    import io, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
                    from sklearn.preprocessing import LabelEncoder
                    from sklearn.model_selection import train_test_split
                    from sklearn.metrics import classification_report, confusion_matrix
                    from sklearn.svm import SVC
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.neighbors import KNeighborsClassifier
                    from sklearn.ensemble import RandomForestClassifier

                    try:
                        from xgboost import XGBClassifier
                        _has_xgb = True
                    except Exception:
                        _has_xgb = False


                    # ---------- helper: auto-fit Excel columns ----------
                    def _autosize_columns(ws, df, extra_pad=2, max_width=60):
                        for i, col in enumerate(df.columns):
                            series = df[col].astype(str)
                            max_len = max([len(str(col))] + [len(s) for s in series.tolist()])
                            ws.set_column(i, i, min(max_len + extra_pad, max_width))


                    def _autosize_range(ws, first_col, last_col, rows, headers=None, extra_pad=2, max_width=60):
                        # Auto-fit arbitrary cell blocks (for Model_Metrics where we write cells)
                        for c in range(first_col, last_col + 1):
                            values = []
                            if headers and c - first_col < len(headers):
                                values.append(str(headers[c - first_col]))
                            for r in rows:
                                v = r.get(c, "")
                                values.append("" if v is None else str(v))
                            width = min(max(len(x) for x in values) + extra_pad, max_width) if values else 10
                            ws.set_column(c, c, width)


                    # ---------- (A) Rebuild model results (AUTO-RUN; independent of any checkbox) ----------
                    _sel_pc_idx = int(str(selected_pc_loading).replace("PC", "")) - 1
                    X_pc_export = X_pca_all[:, :_sel_pc_idx + 1]  # cumulative PC1..selected
                    pc_label_export = f"PC1–PC{_sel_pc_idx + 1}"

                    le_export = LabelEncoder()
                    y_codes_export = le_export.fit_transform(y)

                    X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(
                        X_pc_export, y_codes_export, test_size=0.2, stratify=y_codes_export, random_state=42
                    )

                    models = {
                        "SVM": SVC(),
                        "Logistic Regression": LogisticRegression(max_iter=1000),
                        "KNN": KNeighborsClassifier(n_neighbors=3),
                        "Random Forest": RandomForestClassifier(n_estimators=100),
                    }
                    if _has_xgb:
                        models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric="logloss")

                    results_rows = []
                    confmats_counts = {}
                    confmats_pct = {}
                    confmat_heatmap_png = {}

                    for name, clf in models.items():
                        clf.fit(X_train_e, y_train_e)
                        y_pred = clf.predict(X_test_e)

                        acc = clf.score(X_test_e, y_test_e)
                        rep = classification_report(
                            y_test_e, y_pred, target_names=list(le_export.classes_), output_dict=True, zero_division=0
                        )
                        prec = rep["weighted avg"]["precision"]
                        rec = rep["weighted avg"]["recall"]
                        f1 = rep["weighted avg"]["f1-score"]

                        results_rows.append({
                            "Model": name,
                            "PC used": pc_label_export,
                            "Accuracy": acc,
                            "Precision (weighted)": prec,
                            "Recall (weighted)": rec,
                            "F1-Score (weighted)": f1
                        })

                        # Confusion matrices
                        cm = confusion_matrix(y_test_e, y_pred, labels=np.arange(len(le_export.classes_)))
                        confmats_counts[name] = pd.DataFrame(cm, index=list(le_export.classes_),
                                                             columns=list(le_export.classes_))

                        with np.errstate(divide="ignore", invalid="ignore"):
                            cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100.0
                        cm_pct = np.nan_to_num(cm_pct)
                        confmats_pct[name] = pd.DataFrame(cm_pct, index=list(le_export.classes_),
                                                          columns=list(le_export.classes_))

                        # Heatmap PNG (in-memory)
                        fig, ax = plt.subplots(figsize=(5, 4))
                        sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                                    xticklabels=list(le_export.classes_), yticklabels=list(le_export.classes_),
                                    cbar=True, ax=ax, vmin=0, vmax=100)
                        ax.set_xlabel("Predicted");
                        ax.set_ylabel("True")
                        ax.set_title(f"{name} — Confusion Matrix (%) on {pc_label_export}")
                        _png = io.BytesIO()
                        fig.tight_layout()
                        fig.savefig(_png, format="png", dpi=180)
                        plt.close(fig)
                        _png.seek(0)
                        confmat_heatmap_png[name] = _png

                    results_export_df = pd.DataFrame(results_rows)

                    # ---------- (B) Loadings PNG for 'Loadings Graph' (cleaner visual) ----------
                    try:
                        _pc_num = int(str(selected_pc_loading).replace("PC", ""))
                        _var_pct = float(explained_var[_pc_num - 1]) * 100.0
                    except Exception:
                        _var_pct = np.nan

                    _series_vals = pd.to_numeric(loadings[selected_pc_loading], errors="coerce").values
                    _x_idx = np.arange(len(_series_vals))
                    figL, axL = plt.subplots(figsize=(11, 3.2))
                    axL.plot(_x_idx, _series_vals, linewidth=2.8)
                    axL.axhline(0.0, linestyle="--", color="gray", linewidth=1)
                    axL.set_title(f"{selected_pc_loading} Loadings ({_var_pct:.2f}% Variance)")
                    axL.set_xlabel("Index (wavenumber order)")
                    axL.set_ylabel("Loading")
                    axL.grid(alpha=0.15)
                    _loadings_png = io.BytesIO()
                    figL.tight_layout()
                    figL.savefig(_loadings_png, format="png", dpi=180)
                    plt.close(figL)
                    _loadings_png.seek(0)

                    # ---------- (C) Data for 2D/3D projections ----------
                    proj2d_df = df_pca_all[[pc_x, pc_y, "Label"]].copy()
                    if pc_z in df_pca_all.columns:
                        proj3d_df = df_pca_all[[pc_x, pc_y, pc_z, "Label"]].copy()
                    else:
                        proj3d_df = pd.DataFrame(columns=[pc_x, pc_y, "Z", "Label"])

                    # Optional PNG for 3D visualization (Excel has no native 3D scatter)
                    try:
                        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

                        fig3, ax3 = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(6, 5))
                        for lab in proj3d_df["Label"].unique():
                            d = proj3d_df[proj3d_df["Label"] == lab]
                            ax3.scatter(d[pc_x], d[pc_y], d[pc_z], label=str(lab), s=16)
                        ax3.set_xlabel(pc_x);
                        ax3.set_ylabel(pc_y);
                        ax3.set_zlabel(pc_z)
                        ax3.set_title("PCA 3D Visualization")
                        ax3.legend(loc="best", fontsize=8)
                        _pca3d_png = io.BytesIO()
                        fig3.tight_layout()
                        fig3.savefig(_pca3d_png, format="png", dpi=180)
                        plt.close(fig3)
                        _pca3d_png.seek(0)
                    except Exception:
                        _pca3d_png = None

                    # ---------- (D) Build Excel with all pages & editable charts ----------
                    buffer_full = io.BytesIO()
                    with pd.ExcelWriter(buffer_full, engine="xlsxwriter") as writer:
                        wb = writer.book

                        # --- PCA Summary ---
                        try:
                            pca_summary.to_excel(writer, sheet_name="PCA Summary", index=False)
                            _autosize_columns(writer.sheets["PCA Summary"], pca_summary)
                        except Exception:
                            pass

                        # --- Variable Importance / Drivers / Regions ---
                        try:
                            importance_df.to_excel(writer, sheet_name="Variable Importance", index=False)
                            _autosize_columns(writer.sheets["Variable Importance"], importance_df)
                        except Exception:
                            pass

                        try:
                            _drivers_view = drivers_df[
                                ["Wavenumber (cm⁻¹)", "Loading", "|Loading|", "Band Annotation", "Interpretation"]]
                            _drivers_view.to_excel(writer, sheet_name="Most Influential", index=False)
                            _autosize_columns(writer.sheets["Most Influential"], _drivers_view)
                        except Exception:
                            pass

                        try:
                            contrib_df.to_excel(writer, sheet_name="Regions", index=False)
                            _autosize_columns(writer.sheets["Regions"], contrib_df)
                        except Exception:
                            pass

                        # --- Loadings (table + Excel-editable chart) ---
                        try:
                            def _num_first(val):
                                s = str(val).strip()
                                if "/" in s:
                                    s = s.split("/")[0].strip()
                                try:
                                    return float(s)
                                except Exception:
                                    return np.nan


                            load_tbl = pd.DataFrame({
                                "Wavenumber (cm⁻¹)": list(loadings.index),
                                "Numeric Wavenumber": [_num_first(v) for v in loadings.index],
                                "Index": np.arange(len(loadings.index)),
                                "Loading": pd.to_numeric(loadings[selected_pc_loading], errors="coerce").values
                            })

                            load_tbl.to_excel(writer, sheet_name="Loadings", index=False)
                            ws_load = writer.sheets["Loadings"]
                            _autosize_columns(ws_load, load_tbl)

                            # Editable line chart (cleaner): thicker stroke, legend bottom
                            chart = wb.add_chart({"type": "line"})
                            n_rows = len(load_tbl)
                            chart.add_series({
                                "name": f"{selected_pc_loading}",
                                "categories": ["Loadings", 1, 1, n_rows, 1],  # Numeric Wavenumber
                                "values": ["Loadings", 1, 3, n_rows, 3],  # Loading
                                "line": {"width": 2.25},
                            })
                            chart.set_title({"name": f"Loadings — {selected_pc_loading} ({_var_pct:.2f}% var)"})
                            chart.set_x_axis({"name": "Wavenumber (cm⁻¹)"})
                            chart.set_y_axis({"name": "Loading"})
                            chart.set_legend({"position": "bottom"})
                            ws_load.insert_chart(n_rows + 3, 0, chart)
                        except Exception:
                            pass

                        # --- 📝 Auto-summary ---
                        try:
                            auto_lines = []
                            auto_lines.extend(summary)
                            auto_lines.append("Key drivers:")
                            auto_lines.extend(bullets)
                            auto_df = pd.DataFrame({"Summary": auto_lines})
                            auto_df.to_excel(writer, sheet_name="📝 Auto-summary", index=False)
                            _autosize_columns(writer.sheets["📝 Auto-summary"], auto_df)
                        except Exception:
                            pass

                        # --- 📉 PCA 2D Projection (data + editable scatter chart; markers only, no lines) ---
                        try:
                            sheet2d = "📉 PCA 2D Projection"
                            proj2d_df.to_excel(writer, sheet_name=sheet2d, index=False)
                            ws2d = writer.sheets[sheet2d]
                            _autosize_columns(ws2d, proj2d_df)

                            chart2d = wb.add_chart({"type": "scatter"})
                            cur_row = len(proj2d_df) + 3
                            for lab in proj2d_df["Label"].unique():
                                sub = proj2d_df[proj2d_df["Label"] == lab][[pc_x, pc_y]].reset_index(drop=True)
                                if sub.empty:
                                    continue
                                ws2d.write(cur_row, 0, f"{lab} — {pc_x}")
                                ws2d.write(cur_row, 1, f"{lab} — {pc_y}")
                                ws2d.write_column(cur_row + 1, 0, sub[pc_x].tolist())
                                ws2d.write_column(cur_row + 1, 1, sub[pc_y].tolist())

                                s_start, s_end = cur_row + 1, cur_row + len(sub)
                                chart2d.add_series({
                                    "name": str(lab),
                                    "categories": [sheet2d, s_start, 0, s_end, 0],
                                    "values": [sheet2d, s_start, 1, s_end, 1],
                                    "marker": {"type": "circle", "size": 5},
                                    "line": {"none": True},
                                })
                                cur_row = s_end + 2

                            chart2d.set_title({"name": f"PCA 2D: {pc_x} vs {pc_y}"})
                            chart2d.set_x_axis({"name": pc_x})
                            chart2d.set_y_axis({"name": pc_y})
                            chart2d.set_legend({"position": "bottom"})
                            ws2d.insert_chart(1, 4, chart2d)
                        except Exception:
                            pass

                        # --- 🧭 PCA 3D Visualization (data + PNG) ---
                        try:
                            sheet3d = "🧭 PCA 3D Visualization"
                            proj3d_df.to_excel(writer, sheet_name=sheet3d, index=False)
                            ws3d = writer.sheets[sheet3d]
                            _autosize_columns(ws3d, proj3d_df)
                            if _pca3d_png is not None:
                                ws3d.insert_image(1, 5, "pca3d.png", {"image_data": _pca3d_png})
                        except Exception:
                            pass

                        # --- Model_Metrics (summary + all confusion matrices + heatmaps on one sheet) ---
                        wsM = wb.add_worksheet("Model_Metrics")

                        headers = list(results_export_df.columns)
                        for c, h in enumerate(headers):
                            wsM.write(0, c, h)
                        for r, row in results_export_df.iterrows():
                            for c, h in enumerate(headers):
                                wsM.write(r + 1, c, row[h])
                        _autosize_columns(wsM, results_export_df)

                        start_row = len(results_export_df) + 3

                        for name in models.keys():
                            wsM.write(start_row, 0, f"{name} — Confusion Matrix (counts)")
                            cnt = confmats_counts[name]
                            for j, col in enumerate([""] + cnt.columns.tolist()):
                                wsM.write(start_row + 1, j, col)
                            for i, idx in enumerate(cnt.index.tolist()):
                                wsM.write(start_row + 2 + i, 0, idx)
                                for j, col in enumerate(cnt.columns.tolist()):
                                    wsM.write(start_row + 2 + i, 1 + j, int(cnt.loc[idx, col]))
                            _autosize_range(
                                wsM,
                                first_col=0,
                                last_col=len(cnt.columns),
                                rows=[{k: None for k in range(len(cnt.columns) + 1)} for _ in range(len(cnt.index) + 2)],
                                headers=[""] + cnt.columns.tolist()
                            )
                            start_row += len(cnt) + 3

                            wsM.write(start_row, 0, f"{name} — Confusion Matrix (%)")
                            pct = confmats_pct[name]
                            for j, col in enumerate([""] + pct.columns.tolist()):
                                wsM.write(start_row + 1, j, col)
                            for i, idx in enumerate(pct.index.tolist()):
                                wsM.write(start_row + 2 + i, 0, idx)
                                for j, col in enumerate(pct.columns.tolist()):
                                    wsM.write_number(start_row + 2 + i, 1 + j, float(pct.loc[idx, col]))
                            img_col = pct.shape[1] + 3
                            wsM.insert_image(start_row + 1, img_col, f"{name}_cm.png",
                                             {"image_data": confmat_heatmap_png[name], "x_scale": 1.0, "y_scale": 1.0})
                            start_row += len(pct) + 10

                    buffer_full.seek(0)

                    # ================= Download Complete PCA Analysis =================
                    # st.markdown("### 📥 Download Complete PCA Analysis")

                    colA, colB = st.columns([4, 1])  # adjust ratio if you want button size balance

                    with colA:
                        st.download_button(
                            "📥 Download PCA Full Workbook (editable loadings chart, 2D chart, 3D, summaries, all confusion matrices)",
                            data=buffer_full.getvalue(),
                            file_name="pca_full_export.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )

                    with colB:
                        st.empty()  # placeholder in case you want another button/space

                    # =====================================================================================================================================

            ###############    📊 PLS: Partial Least Squares Regression
            if show_pls:

                col1, col2 = st.columns([4, 1])  # 4:1 width ratio (adjust as needed)

                with col1:
                    st.subheader("📊 PLS: Partial Least Squares Regression")


                with st.expander(
                    "PLS is a supervised projection method. This panel shows a summary table, interactive 2D/3D projections with selectable axes, loadings for a chosen component, and variable-importance with bond annotations.",
                    expanded=False
                ):

                    ############################################ [ PLS Analysis ] ##############################################################################

                    import numpy as np
                    import pandas as pd
                    import streamlit as st
                    from sklearn.cross_decomposition import PLSRegression
                    from sklearn.model_selection import StratifiedKFold, KFold
                    from sklearn.metrics import mean_squared_error, r2_score
                    from scipy.signal import savgol_filter
                    from scipy.sparse import diags, eye
                    from scipy.sparse.linalg import spsolve

                    # ---------------- Preprocessing selection (combined) ----------------
                    PREPROCESSING_OPTIONS = ["Mean Center", "Autoscale", "Smoothing", "Normalization", "Baseline",
                                             "Second Derivative"]
                    x_preproc_pls = st.multiselect(
                        "Select Preprocessing Techniques for PLS (applied together, in order)",
                        options=PREPROCESSING_OPTIONS,
                        default=["Autoscale"]
                    )

                    # ========== Controls ==========
                    max_pls = int(min(10, X.shape[1], len(selected_columns)))
                    n_comp = st.slider("Number of PLS components (LVs)", 2, max_pls, value=min(5, max_pls), step=1)


                    # ===================== helpers: Savitzky–Golay =====================
                    def _savgol_params(n_feat: int, window: int, poly: int, deriv: int):
                        w_max_odd = n_feat if (n_feat % 2 == 1) else (n_feat - 1)
                        if w_max_odd < 3:
                            return None
                        w = int(window)
                        if w % 2 == 0: w -= 1
                        if w < 3: w = 3
                        if w > w_max_odd: w = w_max_odd
                        p = int(poly)
                        p = max(p, deriv, 1)
                        if p >= w: p = w - 1
                        return w, p

                    def smooth_savgol(X, window=7, poly=3):
                        Xn = X.values.astype(float) if hasattr(X, "values") else np.asarray(X, dtype=float)
                        Xn = Xn.copy()
                        params = _savgol_params(Xn.shape[1], window, poly, deriv=0)
                        if params is None: return Xn
                        w, p = params
                        return savgol_filter(Xn, window_length=w, polyorder=p, deriv=0, axis=1, mode="interp")

                    def second_derivative_savgol(X, window=7, poly=3):
                        Xn = X.values.astype(float) if hasattr(X, "values") else np.asarray(X, dtype=float)
                        Xn = Xn.copy()
                        params = _savgol_params(Xn.shape[1], window, poly, deriv=2)
                        if params is None: return Xn
                        w, p = params
                        return savgol_filter(Xn, window_length=w, polyorder=p, deriv=2, axis=1, mode="interp")


                    # ===================== helpers: Baseline (ALS, sparse + fast) =====================
                    def als_baseline_sparse(y, lam=1e5, p=0.001, niter=10, eps=1e-12):
                        y = np.asarray(y, dtype=float).ravel()
                        L = y.size
                        if L < 3:
                            return np.zeros_like(y)
                        D = diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L))
                        DTD = lam * (D.T @ D)
                        w = np.ones(L)
                        I = eye(L) * eps
                        for _ in range(niter):
                            W = diags(w, 0, shape=(L, L))
                            z = spsolve(W + DTD + I, w * y)
                            w = p * (y > z) + (1 - p) * (y < z)
                        return z


                    # ===================== helpers: leakage-safe MC/Autoscale =====================
                    def fit_center_scale_stats(X_arr, need_mean: bool, need_std: bool):
                        mu = sd = None
                        if need_mean or need_std:
                            mu = X_arr.mean(axis=0, keepdims=True)
                        if need_std:
                            sd = X_arr.std(axis=0, keepdims=True)
                            sd = sd.copy();
                            sd[sd == 0] = 1.0
                        return mu, sd


                    def apply_preprocessing_chain(
                            X,
                            ops, *,
                            smooth_window=7, smooth_poly=3,
                            normalization_mode="l2",
                            baseline_lam=100_000.0, baseline_p=0.001,
                            mean_=None, std_=None
                    ):
                        Xn = X.values.astype(float) if hasattr(X, "values") else np.asarray(X, dtype=float)
                        Xn = Xn.copy()

                        # 1) Baseline
                        if "Baseline" in ops:
                            Xcorr = np.empty_like(Xn)
                            for i in range(Xn.shape[0]):
                                bl = als_baseline_sparse(Xn[i, :], lam=baseline_lam, p=baseline_p)
                                Xcorr[i, :] = Xn[i, :] - bl
                            Xn = Xcorr

                        # 2) Smoothing / Second Derivative
                        if "Smoothing" in ops and "Second Derivative" in ops:
                            # If both checked, prefer derivative (common chemometrics convention)
                            Xn = second_derivative_savgol(Xn, window=smooth_window, poly=smooth_poly)
                        elif "Smoothing" in ops:
                            Xn = smooth_savgol(Xn, window=smooth_window, poly=smooth_poly)
                        elif "Second Derivative" in ops:
                            Xn = second_derivative_savgol(Xn, window=smooth_window, poly=smooth_poly)

                        # 3) Mean Center / Autoscale
                        if "Mean Center" in ops or "Autoscale" in ops:
                            mu = mean_ if mean_ is not None else Xn.mean(axis=0, keepdims=True)
                            Xn = Xn - mu
                        if "Autoscale" in ops:
                            sd = std_ if std_ is not None else Xn.std(axis=0, keepdims=True)
                            sd = sd.copy();
                            sd[sd == 0] = 1.0
                            Xn = Xn / sd

                        # 4) Normalization (row-wise)
                        if "Normalization" in ops:
                            mode = normalization_mode.lower()
                            if mode == "l2":
                                denom = np.linalg.norm(Xn, axis=1, keepdims=True)
                            elif mode == "l1":
                                denom = np.sum(np.abs(Xn), axis=1, keepdims=True)
                            else:
                                denom = np.max(np.abs(Xn), axis=1, keepdims=True)
                            denom[denom == 0] = 1.0
                            Xn = Xn / denom

                        return Xn


                    # ===================== helpers: Y shape & metrics =====================
                    def _get_y_matrix():
                        """Fetch y from globals and return as (n_samples, n_targets) float array."""
                        y_name = None
                        if "y_encoded" in globals():
                            y_val = globals()["y_encoded"];
                            y_name = "y_encoded"
                        elif "y" in globals():
                            y_val = globals()["y"];
                            y_name = "y"
                        else:
                            st.error("PLS requires a target vector `y` or `y_encoded` to be defined.")
                            raise RuntimeError("Missing y")

                        y_arr = y_val.values if hasattr(y_val, "values") else y_val
                        y_arr = np.asarray(y_arr)
                        if y_arr.ndim == 1:
                            y_arr = y_arr.reshape(-1, 1)
                        return y_arr, y_name


                    def _rmse_r2_bias(y_true, y_pred):
                        """RMSE / R2 / Bias averaged across targets if multi-output."""
                        y_true = np.asarray(y_true, dtype=float)
                        y_pred = np.asarray(y_pred, dtype=float)
                        if y_true.ndim == 1: y_true = y_true.reshape(-1, 1)
                        if y_pred.ndim == 1: y_pred = y_pred.reshape(-1, 1)
                        # RMSE averaged over targets
                        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
                        # R2 averaged over targets
                        try:
                            r2_vals = []
                            for j in range(y_true.shape[1]):
                                r2_vals.append(r2_score(y_true[:, j], y_pred[:, j]))
                            r2 = float(np.mean(r2_vals))
                        except Exception:
                            r2 = float("nan")
                        bias = float(np.mean(y_pred - y_true))
                        return rmse, r2, bias


                    # ===================== Controls (PLS) =====================
                    max_pls = max(2, min(10, X.shape[1]))

                    cv_splits_pls = st.slider("CV Splits (PLS)", 3, 15, value=10, step=1)
                    conf_limit_pls = st.selectbox("Confidence Limit (PLS)", [0.90, 0.95, 0.99], index=1)
                    lv_list_pls = st.multiselect(
                        "Ncomp/LVs to evaluate (PLS)",
                        options=list(range(2, max_pls + 1)),
                        default=list(range(2, min(3, max_pls) + 1))
                    )

                    # ===================== CAL & CV with the combined preprocessing =====================
                    try:
                        y_mat, y_label_used = _get_y_matrix()
                    except RuntimeError:
                        st.stop()

                    rows_pls = []
                    model_id = 1

                    # Label for the chosen combination
                    preproc_label_pls = ", ".join(x_preproc_pls) if x_preproc_pls else "None"

                    # ---- Calibration data: apply full chain to ALL X ----
                    X_all = apply_preprocessing_chain(
                        X, x_preproc_pls,
                        smooth_window=7, smooth_poly=3,
                        baseline_lam=100_000.0, baseline_p=0.001
                    )

                    X_size_pls = f"{X.shape[0]} x {X.shape[1]}"
                    Y_size_pls = f"{y_mat.shape[0]} x {y_mat.shape[1]}"

                    for ncomp in lv_list_pls:
                        # ===== Calibration on ALL data =====
                        pls_all = PLSRegression(n_components=ncomp)
                        pls_all.fit(X_all, y_mat)
                        y_hat_all = pls_all.predict(X_all)

                        rmsec_pls, r2c_pls, bias_c_pls = _rmse_r2_bias(y_mat, y_hat_all)

                        # ===== CV =====
                        # Prefer StratifiedKFold when y is 1D discrete-like, otherwise KFold
                        y_for_split_1d = y_mat.ravel()  # flatten for stratification attempt
                        try:
                            folds = StratifiedKFold(n_splits=cv_splits_pls, shuffle=True, random_state=42).split(X,
                                                                                                                 y_for_split_1d)
                        except Exception:
                            folds = KFold(n_splits=cv_splits_pls, shuffle=True, random_state=42).split(X)

                        y_cv_pred = np.zeros_like(y_mat, dtype=float)

                        for tr_idx, te_idx in folds:
                            X_tr = X.iloc[tr_idx].values.astype(float)
                            X_te = X.iloc[te_idx].values.astype(float)
                            y_tr = y_mat[tr_idx, :]

                            # Train-only stats for MC/Autoscale if selected
                            need_mean = ("Mean Center" in x_preproc_pls) or ("Autoscale" in x_preproc_pls)
                            need_std = ("Autoscale" in x_preproc_pls)
                            mu_tr, sd_tr = fit_center_scale_stats(X_tr, need_mean, need_std)

                            X_tr_s = apply_preprocessing_chain(
                                X_tr, x_preproc_pls,
                                smooth_window=7, smooth_poly=3,
                                baseline_lam=100_000.0, baseline_p=0.001,
                                mean_=mu_tr, std_=sd_tr
                            )
                            X_te_s = apply_preprocessing_chain(
                                X_te, x_preproc_pls,
                                smooth_window=7, smooth_poly=3,
                                baseline_lam=100_000.0, baseline_p=0.001,
                                mean_=mu_tr, std_=sd_tr
                            )

                            pls_cv = PLSRegression(n_components=ncomp)
                            pls_cv.fit(X_tr_s, y_tr)
                            y_cv_pred[te_idx, :] = pls_cv.predict(X_te_s)

                        rmsecv_pls, r2cv_pls, bias_cv_pls = _rmse_r2_bias(y_mat, y_cv_pred)
                        rmse_ratio_pls = rmsecv_pls / rmsec_pls if rmsec_pls > 0 else float("nan")

                        rows_pls.append({
                            "CV Splits": cv_splits_pls,
                            "Confidence Limit": conf_limit_pls,
                            "RMSE Ratio (RMSECV/RMSEC)": rmse_ratio_pls,
                            "RMSEC (Cal)": rmsec_pls,
                            "RMSECV (CV)": rmsecv_pls,
                            "Model Name": f"Model {model_id}",
                            "Ncomp/LVs": ncomp,
                            "X-Preprocessing": preproc_label_pls,
                            "X Include Size": X_size_pls,
                            "Y Include Size": Y_size_pls,
                            "R2C (Cal)": r2c_pls,
                            "R2CV (CV)": r2cv_pls,
                            "Bias": bias_c_pls,
                            "Bias (CV)": bias_cv_pls
                        })
                        model_id += 1

                    # ---------------- Table + download ----------------
                    pls_calcv_df = pd.DataFrame(rows_pls, columns=[
                        "CV Splits", "Confidence Limit", "RMSE Ratio (RMSECV/RMSEC)",
                        "RMSEC (Cal)", "RMSECV (CV)", "Model Name", "Ncomp/LVs",
                        "X-Preprocessing", "X Include Size", "Y Include Size",
                        "R2C (Cal)", "R2CV (CV)", "Bias", "Bias (CV)"
                    ])

                    fmt_cols_pls = ["RMSE Ratio (RMSECV/RMSEC)", "RMSEC (Cal)", "RMSECV (CV)", "R2C (Cal)", "R2CV (CV)",
                                    "Bias", "Bias (CV)"]

                    st.markdown("### 📋 PLS Analysis Result Summary")
                    st.dataframe(
                        pls_calcv_df.style
                        .format({c: "{:.4f}" for c in fmt_cols_pls})
                        .background_gradient(subset=["R2CV (CV)", "R2C (Cal)"], cmap="YlGn")
                        .background_gradient(subset=["RMSECV (CV)", "RMSEC (Cal)"], cmap="YlOrRd_r"),
                        use_container_width=True
                    )

                    st.download_button(
                        "📥 Download PLS Summary (CSV)",
                        data=pls_calcv_df.to_csv(index=False).encode("utf-8"),
                        file_name="pls_summary_table.csv",
                        mime="text/csv"
                    )

                    ##########################################################################################################################

                    ##### Display plots

                    # Scale X (recommended for PLS on spectroscopy)
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    # y must be continuous for PLSRegression; for classification we use the encoded labels
                    pls = PLSRegression(n_components=n_comp)
                    pls.fit(X_scaled, y_encoded)

                    # Scores / Loadings / Weights
                    T = pls.x_scores_                    # (n_samples, n_comp)
                    U = pls.y_scores_                    # (n_samples, n_comp)
                    P = pls.x_loadings_                  # (n_features, n_comp)
                    W = pls.x_weights_                   # (n_features, n_comp)
                    Q = pls.y_loadings_                  # (1 or n_classes, n_comp) depending on encoding

                    # ========== Axis Selection for 2D/3D ==========
                    all_pls_labels = [f"PLS{i+1}" for i in range(n_comp)]
                    col_ax1, col_ax2, col_ax3 = st.columns(3)
                    with col_ax1:
                        pls_x_label = st.selectbox("X-axis", all_pls_labels, index=0, key="pls_x_axis")
                    with col_ax2:
                        pls_y_label = st.selectbox("Y-axis (2D/3D)", all_pls_labels, index=min(1, n_comp-1), key="pls_y_axis")
                    with col_ax3:
                        pls_z_label = st.selectbox("Z-axis (3D)", all_pls_labels, index=min(2, n_comp-1), key="pls_z_axis")

                    x_idx = int(pls_x_label.replace("PLS","")) - 1
                    y_idx = int(pls_y_label.replace("PLS","")) - 1
                    z_idx = int(pls_z_label.replace("PLS","")) - 1

                    # Build a DataFrame of selected scores
                    df_pls = pd.DataFrame({
                        "Label": y.values,
                        "PLS1": T[:,0] if n_comp >= 1 else np.nan,
                        "PLS2": T[:,1] if n_comp >= 2 else np.nan,
                        "PLS3": T[:,2] if n_comp >= 3 else np.nan
                    })
                    # Include arbitrary components by their chosen indexes
                    for i in range(n_comp):
                        df_pls[f"PLS{i+1}"] = T[:, i]

                    # 2D Plot
                    st.subheader("📉 PLS 2D Projection")
                    xcol = f"PLS{x_idx + 1}"
                    ycol = f"PLS{y_idx + 1}"

                    fig_pls_2d = px.scatter(
                        df_pls,
                        x=xcol,
                        y=ycol,
                        color="Label",
                        title=f"PLS Projection ({pls_x_label} vs {pls_y_label})",
                        labels={xcol: pls_x_label, ycol: pls_y_label}
                    )

                    # ---- Add class ellipses (same approach as PCA) ----
                    chi2_map = {0.90: 4.605, 0.95: 5.991, 0.99: 9.210}
                    chi2_val = chi2_map.get(conf_limit_pls, 5.991)  # default 95%


                    def _ellipse_xy(points_2d, chi2_q, npts=200):
                        pts = np.asarray(points_2d)
                        if pts.shape[0] < 3:
                            return None, None
                        mu = pts.mean(axis=0)
                        S = np.cov(pts.T)
                        vals, vecs = np.linalg.eigh(S)
                        order = np.argsort(vals)[::-1]
                        vals, vecs = vals[order], vecs[:, order]
                        radii = np.sqrt(np.maximum(vals, 1e-12) * chi2_q)
                        t = np.linspace(0, 2 * np.pi, npts)
                        circ = np.stack([np.cos(t) * radii[0], np.sin(t) * radii[1]], axis=0)
                        ell = (vecs @ circ).T + mu
                        return ell[:, 0], ell[:, 1]


                    for lab in df_pls["Label"].unique():
                        d = df_pls[df_pls["Label"] == lab][[xcol, ycol]].values
                        ex, ey = _ellipse_xy(d, chi2_val)
                        if ex is not None:
                            fig_pls_2d.add_trace(
                                go.Scatter(
                                    x=ex, y=ey, mode="lines",
                                    name=f"{lab} {int(conf_limit_pls * 100)}% ellipse",
                                    line=dict(width=1),
                                    showlegend=False
                                )
                            )

                    st.plotly_chart(fig_pls_2d, use_container_width=True)

                    # 3D Plot (only if at least 3 comps)
                    if n_comp >= 3:
                        st.subheader("🧭 PLS 3D Visualization")
                        fig_pls_3d = px.scatter_3d(
                            df_pls,
                            x=f"PLS{x_idx+1}",
                            y=f"PLS{y_idx+1}",
                            z=f"PLS{z_idx+1}",
                            color="Label",
                            title=f"3D PLS: {pls_x_label} vs {pls_y_label} vs {pls_z_label}",
                            height=520, width=800
                        )
                        st.plotly_chart(fig_pls_3d, use_container_width=True)
                    else:
                        st.info("Add ≥3 components to view the 3D projection.")

                    #### ======================= PLS Loadings with Bond Annotation =======================

                    # -------------------- Helpers --------------------
                    def _num_first(val):
                        s = str(val).strip()
                        if "/" in s:
                            s = s.split("/")[0].strip()
                        try:
                            return float(s)
                        except Exception:
                            return np.nan


                    # -------------------- Prepare X (numeric) --------------------
                    if isinstance(X, pd.DataFrame):
                        X_num = X.apply(pd.to_numeric, errors="coerce").values
                        selected_columns = list(X.columns)
                    else:
                        X_num = np.asarray(X, dtype=float)
                        selected_columns = [str(c) for c in range(X_num.shape[1])]

                    # fill NaNs if any
                    if np.isnan(X_num).any():
                        col_means = np.nanmean(X_num, axis=0)
                        ii = np.where(np.isnan(X_num))
                        X_num[ii] = np.take(col_means, ii[1])

                    # -------------------- Prepare Y (numeric / PLS-DA) --------------------
                    if isinstance(y, pd.Series):
                        y_arr = y.values
                    elif isinstance(y, pd.DataFrame):
                        y_arr = y.values
                    else:
                        y_arr = np.asarray(y)

                    if y_arr.ndim == 1 and not np.issubdtype(y_arr.dtype, np.number):
                        _le = LabelEncoder()
                        y_codes = _le.fit_transform(y_arr)
                        if len(np.unique(y_codes)) == 2:
                            Y_matrix = y_codes.reshape(-1, 1).astype(float)
                        else:
                            Y_matrix = np.eye(len(np.unique(y_codes)), dtype=float)[y_codes]
                    elif y_arr.ndim == 1:
                        Y_matrix = y_arr.reshape(-1, 1).astype(float)
                    else:
                        Y_matrix = y_arr.astype(float)

                    # -------------------- Fit PLS (if not already) & explained X-variance --------------------
                    n_samples, p = X_num.shape
                    q = 1 if Y_matrix.ndim == 1 else Y_matrix.shape[1]
                    max_lv_safe = max(1, min(n_samples - 1, p, max(1, q)))
                    ncomp_pls = min(5, max_lv_safe)

                    try:
                        pls  # reuse if already fitted above
                        T_all = pls.x_scores_
                    except Exception:
                        pls = PLSRegression(n_components=ncomp_pls, scale=False).fit(X_num, Y_matrix)
                        T_all = pls.x_scores_

                    explained_x = getattr(pls, "x_explained_variance_ratio_", None)
                    if explained_x is None:
                        X_tot = float(np.var(X_num, axis=0).sum()) or 1.0
                        explained_x = np.var(T_all, axis=0) / X_tot
                    explained_x = np.asarray(explained_x, dtype=float)

                    # -------------------- Compute PLS Loadings (X-loadings) --------------------
                    loadings = pd.DataFrame(
                        pls.x_loadings_,
                        index=selected_columns,
                        columns=[f"PLS{i + 1}" for i in range(pls.x_loadings_.shape[1])]
                    )

                    # -------------------- Normalize explained X-variance for nice percentages --------------------
                    # Prefer sklearn's attribute if present; otherwise compute from scores and normalize to sum=1
                    T_all = getattr(pls, "x_scores_", None)
                    explained_x_attr = getattr(pls, "x_explained_variance_ratio_", None)
                    if explained_x_attr is not None:
                        explained_x = np.asarray(explained_x_attr, dtype=float)
                    elif T_all is not None:
                        raw_var = np.var(T_all, axis=0)
                        explained_x = raw_var / np.sum(raw_var) if np.sum(raw_var) > 0 else np.zeros_like(raw_var)
                    else:
                        # Fallback: equal share to avoid crashes (should rarely happen)
                        explained_x = np.ones(pls.x_loadings_.shape[1], dtype=float) / pls.x_loadings_.shape[1]

                    def _lv_index(name: str) -> int:
                        # Parse "PLS3" -> 2; be defensive
                        import re
                        m = re.search(r"\d+", str(name))
                        idx = int(m.group()) - 1 if m else 0
                        return max(0, min(idx, len(explained_x) - 1))

                    # -------------------- User Selects LV to View Loadings --------------------
                    st.subheader("📊 PLS Loadings")
                    selected_lv_loading = st.selectbox("Select PLS to view Loadings:", loadings.columns, key="pls_loadings_select")

                    lv_idx = _lv_index(selected_lv_loading)
                    var_pct = float(explained_x[lv_idx] * 100.0)

                    # Clean, consistent title line (like PCA)
                    st.markdown(
                        f"### 🔑 Loadings Plot for {selected_lv_loading} &nbsp;&nbsp;—&nbsp;&nbsp;**{var_pct:.2f}%** of total variance")

                    # ===== line plot of loading (with quick band annotations like your PCA block) =====
                    fig_load1 = go.Figure()
                    fig_load1.add_trace(go.Scatter(
                        x=selected_columns,
                        y=loadings[selected_lv_loading],
                        mode="lines",
                        line=dict(color="blue"),
                        name=f"{selected_lv_loading} Loadings"
                    ))
                    fig_load1.add_hline(y=0, line=dict(color="gray", dash="dash"))  # Zero reference line

                    # ✅ Add Band Annotations (same set as PCA)
                    band_annotations1 = {
                        1740: "C=O Stretch (Lipid)",
                        1650: "Amide I (Protein)",
                        1550: "Amide II (Protein)",
                        1338: "CH₂ (Collagen)",
                        1200: "C–O Stretch (Matrix)",
                        1115: "HPO₄²⁻ (Mineral)",
                        1060: "Sugar Ring C–O",
                        1030: "PO₄³⁻ (Mineral)",
                        875: "CO₃²⁻ Bending",
                        856: "C–S Proteoglycan",
                        7000: "O–H Stretch",
                        6688: "N–H Stretch",
                        5800: "CH₂ Lipid",
                        5200: "O–H Water",
                    }
                    try:
                        _lv_idx_for_title = int(str(selected_lv_loading).replace("LV", "")) - 1
                    except Exception:
                        _lv_idx_for_title = 0

                    try:
                        # use selected_range if present (from your global UI)
                        _lo, _hi = float(selected_range[0]), float(selected_range[1])
                    except Exception:
                        # infer a full range from numeric column names
                        wn_numeric_all = np.array([_num_first(c) for c in selected_columns], dtype=float)
                        mask_valid = np.isfinite(wn_numeric_all)
                        if mask_valid.any():
                            _lo, _hi = float(np.nanmin(wn_numeric_all[mask_valid])), float(
                                np.nanmax(wn_numeric_all[mask_valid]))
                        else:
                            _lo, _hi = -np.inf, np.inf

                    for wn, label in band_annotations1.items():
                        if _lo <= wn <= _hi:
                            fig_load1.add_vline(x=wn, line=dict(color="gray", width=1, dash="dash"))
                            fig_load1.add_annotation(
                                x=wn, y=loadings[selected_lv_loading].max(), text=label,
                                showarrow=True, arrowhead=1, yshift=10, font=dict(size=9)
                            )

                    fig_load1.update_layout(
                        title=f"PLS Loadings for {selected_lv_loading} ({(explained_x[_lv_idx_for_title] * 100.0 if _lv_idx_for_title < len(explained_x) else float('nan')):.2f}% X-Variance)",
                        xaxis_title="Wavenumber (cm⁻¹)",
                        yaxis_title="Loading Score",
                        height=400
                    )
                    st.plotly_chart(fig_load1, use_container_width=True)

                    ############################## [  📊 Variable Importance with Bond Annotations (PLS) ] ############################################################

                    import numpy as np
                    import pandas as pd
                    import streamlit as st

                    # --- Bond Annotation Mapping Table (same as your PCA section) ---
                    bond_annotations_table = [
                        (8500, "O–H Stretching and Bending (Water)"),
                        (7000, "O–H Stretching (Water)"),
                        (6688, "N–H Stretching (Protein/Collagen)"),
                        (5800, "CH₂ Stretching (Lipid)"),
                        (5200, "O–H Stretching and Bending (Water)"),
                        (4890, "N–H Bending (Protein/Collagen)"),
                        (4610, "C–H Stretching & Deformation (Protein/Collagen)"),
                        (4310, "Sugar Ring Vibrations (Proteoglycan)"),
                        ((3600, 3200), "O–H Stretching (Water/Hydroxyl)"),
                        ((3500, 3300), "N–H Stretching (Proteins)"),
                        ((3000, 2800), "C–H Stretching (Lipids, CH₂)"),
                        ((1750, 1650), "C=O Stretching (Proteins/Lipids)"),
                        (1550, "Amide II (Proteins)"),
                        (1338, "CH₂ Side Chain Bending (Collagen)"),
                        ((1100, 900), "PO₄³⁻ Stretching (Bone Mineral)"),
                        ((890, 850), "CO₃²⁻ Bending (Carbonate)"),
                        (1740, "C=O Stretching (Ester, Lipids)"),
                        (1650, "Amide I (Proteins)"),
                        (1630, "Water O–H Bending (Water)"),
                        ((1200, 1000), "C–O Stretching (Alcohols/Ethers)"),
                        (1115, "HPO₄²⁻ Stretching (Bone mineral)"),
                        (1060, "Sugar Ring C–O Stretch (Carbohydrates)"),
                        (1030, "PO₄³⁻ Stretching (Bone Mineral)"),
                        (875, "CO₃²⁻ Bending (Carbonates)"),
                        (856, "C–S Bending (Proteoglycans)"),
                        ((900, 800), "Aromatic C–H Bending (Fingerprint region)")
                    ]


                    def get_bond_annotation(wavenumber: float) -> str:
                        # Check ranges first
                        for entry in bond_annotations_table:
                            if isinstance(entry[0], tuple):
                                low, high = entry[0]
                                if high <= wavenumber <= low:
                                    return entry[1]
                        # Exact match
                        for entry in bond_annotations_table:
                            if not isinstance(entry[0], tuple) and entry[0] == int(wavenumber):
                                return entry[1]
                        # Nearest match if no exact/range match
                        all_wavenumbers = [val if isinstance(val, int) else (val[0] + val[1]) // 2 for val, _ in
                                           bond_annotations_table]
                        closest_idx = int(np.argmin([abs(wavenumber - wn) for wn in all_wavenumbers]))
                        return bond_annotations_table[closest_idx][1]


                    # --- Variable Importance by |loadings| (top 20) ---
                    variable_importance = loadings.abs()[selected_lv_loading].sort_values(ascending=False)
                    importance_df = pd.DataFrame(variable_importance.head(20)).reset_index()
                    importance_df.columns = ["Wavenumber (cm⁻¹)", "Importance"]


                    # Safely extract numeric value (handles labels like '1030/1240')
                    def extract_first_numeric(value):
                        s = str(value).strip()
                        if "/" in s:
                            s = s.split("/")[0].strip()
                        try:
                            return float(s)
                        except ValueError:
                            return None


                    importance_df["Numeric Wavenumber"] = importance_df["Wavenumber (cm⁻¹)"].apply(
                        extract_first_numeric)
                    importance_df = importance_df.dropna(subset=["Numeric Wavenumber"])
                    importance_df["Band Annotation"] = importance_df["Numeric Wavenumber"].apply(get_bond_annotation)

                    ############################## [ 📊 Variable Importance with Bond Annotations - PLS ] ############################################################

                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown("### 📊 Variable Importance with Bond Annotations")
                    with col2:
                        var_importance_csv = importance_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "📥 Download CSV",
                            data=var_importance_csv,
                            file_name="variable_importance_with_bands.csv",
                            mime="text/csv",
                            use_container_width=True,
                            key = "varimp_csv_btn"
                        )

                    # ===== Display table =====
                    st.dataframe(importance_df, use_container_width=True)

                    ##########################################[  ⭐ Most influential wavenumbers (by |loading|)]]################################################
                    # ========================= ⭐ Most influential wavenumbers (by |loading|) — PLS =========================

                    import numpy as np
                    import pandas as pd
                    import streamlit as st

                    # --- helpers (only defined if missing) ---
                    if "_num_first" not in globals():
                        def _num_first(val):
                            s = str(val).strip()
                            if "/" in s:
                                s = s.split("/")[0].strip()
                            try:
                                return float(s)
                            except Exception:
                                return np.nan

                    if "get_bond_annotation" not in globals():
                        bond_annotations_table = [
                            (8500, "O–H Stretching and Bending (Water)"),
                            (7000, "O–H Stretching (Water)"),
                            (6688, "N–H Stretching (Protein/Collagen)"),
                            (5800, "CH₂ Stretching (Lipid)"),
                            (5200, "O–H Stretching and Bending (Water)"),
                            (4890, "N–H Bending (Protein/Collagen)"),
                            (4610, "C–H Stretching & Deformation (Protein/Collagen)"),
                            (4310, "Sugar Ring Vibrations (Proteoglycan)"),
                            ((3600, 3200), "O–H Stretching (Water/Hydroxyl)"),
                            ((3500, 3300), "N–H Stretching (Proteins)"),
                            ((3000, 2800), "C–H Stretching (Lipids, CH₂)"),
                            ((1750, 1650), "C=O Stretching (Proteins/Lipids)"),
                            (1550, "Amide II (Proteins)"),
                            (1338, "CH₂ Side Chain Bending (Collagen)"),
                            ((1100, 900), "PO₄³⁻ Stretching (Bone Mineral)"),
                            ((890, 850), "CO₃²⁻ Bending (Carbonate)"),
                            (1740, "C=O Stretching (Ester, Lipids)"),
                            (1650, "Amide I (Proteins)"),
                            (1630, "Water O–H Bending (Water)"),
                            ((1200, 1000), "C–O Stretching (Alcohols/Ethers)"),
                            (1115, "HPO₄²⁻ Stretching (Bone mineral)"),
                            (1060, "Sugar Ring C–O Stretch (Carbohydrates)"),
                            (1030, "PO₄³⁻ Stretching (Bone Mineral)"),
                            (875, "CO₃²⁻ Bending (Carbonates)"),
                            (856, "C–S Bending (Proteoglycans)"),
                            ((900, 800), "Aromatic C–H Bending (Fingerprint region)")
                        ]


                        def get_bond_annotation(wavenumber: float) -> str:
                            for entry in bond_annotations_table:
                                if isinstance(entry[0], tuple):
                                    low, high = entry[0]
                                    if high <= wavenumber <= low:
                                        return entry[1]
                            for entry in bond_annotations_table:
                                if not isinstance(entry[0], tuple) and entry[0] == int(wavenumber):
                                    return entry[1]
                            mids = [v if isinstance(v, int) else (v[0] + v[1]) // 2 for v, _ in bond_annotations_table]
                            idx = int(np.argmin([abs(wavenumber - m) for m in mids]))
                            return bond_annotations_table[idx][1]

                    if "_favor_class" not in globals():
                        def _favor_class(value, class_pos, class_neg):
                            if class_pos is None or class_neg is None:
                                return ""
                            return f"favors **{class_pos}**" if value > 0 else f"favors **{class_neg}**"

                    # --- compute LV index + X-variance explained for title & summary ---
                    try:
                        lv_idx = int(str(selected_lv_loading).replace("PLS", "")) - 1
                    except Exception:
                        lv_idx = 0
                    try:
                        var_pct = float(explained_x[lv_idx] * 100.0)
                    except Exception:
                        var_pct = float("nan")

                    # --- numeric axis from feature labels ---
                    wn_labels = list(loadings.index)
                    wn_numeric = np.array([_num_first(v) for v in wn_labels], dtype=float)

                    # --- focus window from selected_range if present ---
                    try:
                        lo, hi = float(selected_range[0]), float(selected_range[1])
                        focus_mask = (wn_numeric >= min(lo, hi)) & (wn_numeric <= max(lo, hi))
                        if not np.any(focus_mask):
                            focus_mask = np.isfinite(wn_numeric)
                    except Exception:
                        focus_mask = np.isfinite(wn_numeric)

                    # --- loadings for selected LV ---
                    s_full = pd.to_numeric(loadings[selected_lv_loading], errors="coerce").values
                    s_focus = s_full[focus_mask]
                    wn_focus = wn_numeric[focus_mask]
                    lbl_focus = np.array(wn_labels, dtype=object)[focus_mask]

                    # --- determine class orientation (+/− side) from LV scores (if available) ---
                    class_pos = class_neg = None
                    try:
                        df_scores_pls = pd.DataFrame({"Label": np.asarray(y), "LV_Score": T[:, lv_idx]})
                        class_means_pls = df_scores_pls.groupby("Label", observed=True)["LV_Score"].mean().sort_values(
                            ascending=False)
                        if len(class_means_pls):
                            class_pos = class_means_pls.index[0]
                            class_neg = class_means_pls.index[-1]
                    except Exception:
                        pass

                    # --- build drivers_df (with Band Annotation + Interpretation filled) ---
                    rows = []
                    for lbl, wn, val in zip(lbl_focus, wn_focus, s_focus):
                        band = get_bond_annotation(float(wn)) if np.isfinite(wn) else ""
                        interp = _favor_class(float(val), class_pos, class_neg) if np.isfinite(val) else ""
                        rows.append({
                            "Wavenumber (cm⁻¹)": str(lbl),
                            "Numeric (cm⁻¹)": wn if np.isfinite(wn) else np.nan,
                            "Loading": float(val) if np.isfinite(val) else np.nan,
                            "|Loading|": float(abs(val)) if np.isfinite(val) else np.nan,
                            "Band Annotation": band,
                            "Interpretation": interp
                        })

                    drivers_df = (pd.DataFrame(rows)
                                  .dropna(subset=["Numeric (cm⁻¹)", "Loading"])
                                  .sort_values("|Loading|", ascending=False)
                                  .reset_index(drop=True))

                    # --- UI: Most influential table + CSV ---
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown("### ⭐ Most influential wavenumbers (by |loading|)")
                    with col2:
                        st.download_button(
                            "📥 Download CSV",
                            data=drivers_df[
                                ["Wavenumber (cm⁻¹)", "Loading", "|Loading|", "Band Annotation", "Interpretation"]
                            ].to_csv(index=False).encode("utf-8"),
                            file_name=f"most_influential_wavenumbers_{selected_lv_loading}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                    st.dataframe(
                        drivers_df[
                            ["Wavenumber (cm⁻¹)", "Loading", "|Loading|", "Band Annotation", "Interpretation"]
                        ].style.format({"Loading": "{:.4f}", "|Loading|": "{:.4f}"}),
                        use_container_width=True
                    )

                    ############################## [ 🧩 Auto-detected regions (aggregated contribution) — PLS ] ############################################################

                    # Window half-width for region detection (cm⁻¹)
                    half_width_cm = 25

                    def _region_for(center_cm1):
                        lo = center_cm1 - half_width_cm
                        hi = center_cm1 + half_width_cm
                        return lo, hi

                    regs = []
                    for _, r in drivers_df.dropna(subset=["Numeric (cm⁻¹)"]).iterrows():
                        c = float(r["Numeric (cm⁻¹)"])
                        lo_r, hi_r = _region_for(c)
                        regs.append([lo_r, hi_r, r["Wavenumber (cm⁻¹)"], r["Band Annotation"]])

                    regs.sort(key=lambda x: x[0])
                    merged = []
                    for lo_r, hi_r, lbl, ann in regs:
                        if not merged or lo_r > merged[-1][1]:
                            merged.append([lo_r, hi_r, [lbl], [ann]])
                        else:
                            merged[-1][1] = max(merged[-1][1], hi_r)
                            merged[-1][2].append(lbl)
                            merged[-1][3].append(ann)

                    # Compute region contributions using the focused loadings vector
                    reg_rows = []
                    for lo_r, hi_r, lbls, anns in merged:
                        m = (wn_focus >= lo_r) & (wn_focus <= hi_r)
                        if not np.any(m):
                            continue
                        net = float(np.nansum(s_focus[m]))
                        mass = float(np.nansum(np.abs(s_focus[m])))
                        lbl_text = ", ".join(map(str, lbls[:3])) + ("..." if len(lbls) > 3 else "")
                        ann_text = ", ".join(sorted(set(a for a in anns if a)))
                        reg_rows.append({
                            "Region (cm⁻¹)": f"{int(lo_r)}–{int(hi_r)}",
                            "|Σ loading|": mass,
                            "Net sign": "Positive" if net > 0 else ("Negative" if net < 0 else "≈0"),
                            "Peak labels": lbl_text,
                            "Likely bands": ann_text or "—",
                        })

                    contrib_df = (pd.DataFrame(reg_rows)
                                  .sort_values("|Σ loading|", ascending=False)
                                  .reset_index(drop=True))

                    ################## [ 🧩 Auto-detected regions (aggregated contribution) ] ###############
                    ###### removed script from here


                    ############################## [📝 Auto-summary - PLS ] ############################################################

                    pos_mass = float(np.nansum(np.abs(s_focus[s_focus > 0])))
                    neg_mass = float(np.nansum(np.abs(s_focus[s_focus < 0])))
                    contrast = ("Positive-weighted regions dominate"
                                if pos_mass > neg_mass else
                                ("Negative-weighted regions dominate" if neg_mass > pos_mass else "Balanced positive/negative weights"))

                    summary_lines = []
                    summary_lines.append(f"- **{selected_lv_loading}** explains **{var_pct:.2f}%** of X-variance.")
                    summary_lines.append(f"- **Contrast:** {contrast} in the current range.")
                    if class_pos and class_neg:
                        summary_lines.append(
                            f"- **Class context:** Higher LV scores (positive side) align with **{class_pos}**; "
                            f"lower scores (negative side) align with **{class_neg}**."
                        )
                    if len(contrib_df):
                        top_reg = contrib_df.iloc[0]
                        summary_lines.append(
                            f"- **Strongest region:** {top_reg['Region (cm⁻¹)']} "
                            f"({top_reg['Net sign']}, |Σloading|={top_reg['|Σ loading|']:.4f}); "
                            f"likely bands: {top_reg['Likely bands'] or '—'}."
                        )

                    head_n = min(6, len(drivers_df))
                    bullets = []
                    for _, r in drivers_df.head(head_n).iterrows():
                        side = "positive" if r["Loading"] > 0 else "negative"
                        interp = f" — {r['Interpretation']}" if r["Interpretation"] else ""
                        bullets.append(
                            f"  • {r['Wavenumber (cm⁻¹)']}: **{side}** loading ({r['Band Annotation']}){interp}")

                    st.markdown("### 📝 Auto-summary")

                    st.markdown("\n".join(summary_lines + ["- **Key drivers:**"] + bullets))
                    st.caption(
                        "Notes: (1) PLS loading sign is relative to LV orientation; interpretation uses class-wise mean scores. "
                        "(2) Region aggregation uses |loading| within ±half-width around top wavenumbers. "
                        "(3) Preprocessing (e.g., autoscale/derivative) affects which bands dominate."
                    )
                    ############################## [ 📊 Model Performance Summary (selected PLS) ] ############################################################
                    ############################################[  model performance summary (PLS) ]#########################################################################
                    st.markdown("### 📊 Model Performance Summary")

                    run_pls_clf = st.checkbox("📊 Model Performance Summary for selected LV", value=True,key="pls_run_selected_lv")

                    if run_pls_clf:
                        # --- choose 1+ models (all selected by default) ---
                        model_choices_pls = ["SVM", "Logistic Regression", "KNN", "Random Forest", "XGBoost"]
                        selected_models_pls = st.multiselect(
                            "Select classification models (trained on the selected LV score)",
                            options=model_choices_pls,
                            default=model_choices_pls,
                            key="pls_selected_lv_models"
                        )

                        # Parse selected LV index (e.g., "PLS3" -> 2)
                        try:
                            sel_lv_idx = int(str(selected_lv_loading).replace("PLS", "")) - 1
                        except Exception:
                            sel_lv_idx = 0

                        # --- single vs cumulative toggle (PLS) ---
                        mode_pls = st.radio(
                            "LV feature mode",
                            ["Single (selected only)", "Cumulative (PLS1..selected)"],
                            index=1,
                            key="pls_feature_mode"
                        )

                        # --- Build feature matrix from PLS scores ---
                        # Ensure we have PLS scores matrix T (n_samples x n_comp)
                        try:
                            T_scores = pls.x_scores_
                        except Exception:
                            # fallback: recompute from fitted pls
                            T_scores = pls.transform(X)  # assumes `pls` already fitted

                        if mode_pls == "Single (selected only)":
                            X_pls_feat = T_scores[:, sel_lv_idx].reshape(-1, 1)
                            lv_label = f"{selected_lv_loading}"
                        else:
                            X_pls_feat = T_scores[:, :sel_lv_idx + 1]
                            lv_label = f"PLS1–PLS{sel_lv_idx + 1}"

                        # --- Encode labels locally (use raw y; fallback to y_encoded if needed) ---
                        from sklearn.preprocessing import LabelEncoder

                        if "y" in globals():
                            y_labels_src = y
                        elif "y_encoded" in globals():
                            y_labels_src = y_encoded
                        else:
                            st.error("Targets `y` (labels) or `y_encoded` not found for classification.")
                            st.stop()

                        le_local_pls = LabelEncoder()
                        y_codes_pls = le_local_pls.fit_transform(
                            y_labels_src.values if hasattr(y_labels_src, "values") else y_labels_src
                        )

                        # --- Train/test split (stratified) ---
                        from sklearn.model_selection import train_test_split

                        X_train_pls, X_test_pls, y_train_pls, y_test_pls = train_test_split(
                            X_pls_feat, y_codes_pls, test_size=0.2, stratify=y_codes_pls, random_state=42
                        )

                        # --- Helper to build a model by name (same as PCA block) ---
                        from sklearn.svm import SVC
                        from sklearn.linear_model import LogisticRegression
                        from sklearn.neighbors import KNeighborsClassifier
                        from sklearn.ensemble import RandomForestClassifier
                        from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, \
                            confusion_matrix
                        import numpy as np
                        import pandas as pd
                        import seaborn as sns
                        import matplotlib.pyplot as plt

                        try:
                            from xgboost import XGBClassifier

                            _has_xgb_pls = True
                        except Exception:
                            _has_xgb_pls = False


                        def _make_model_pls(name):
                            if name == "SVM":
                                return SVC()
                            if name == "Logistic Regression":
                                return LogisticRegression(max_iter=1000)
                            if name == "KNN":
                                return KNeighborsClassifier(n_neighbors=3)
                            if name == "Random Forest":
                                return RandomForestClassifier(n_estimators=100)
                            # XGBoost
                            if _has_xgb_pls:
                                return XGBClassifier(use_label_encoder=False, eval_metric="logloss")
                            # graceful fallback if xgboost not available
                            return RandomForestClassifier(n_estimators=100)


                        # --- Run all selected models ---
                        results_pls = []
                        for mdl_name in selected_models_pls:
                            clf_pls = _make_model_pls(mdl_name)
                            clf_pls.fit(X_train_pls, y_train_pls)
                            y_pred_pls = clf_pls.predict(X_test_pls)

                            # Metrics (weighted for multi-class safety)
                            acc_pls = clf_pls.score(X_test_pls, y_test_pls)
                            prec_pls = precision_score(y_test_pls, y_pred_pls, average="weighted", zero_division=0)
                            rec_pls = recall_score(y_test_pls, y_pred_pls, average="weighted", zero_division=0)
                            f1_pls = f1_score(y_test_pls, y_pred_pls, average="weighted", zero_division=0)

                            results_pls.append({
                                "Model": mdl_name,
                                "LV used": lv_label,  # mirrors PCA's "PC used"
                                "Accuracy": acc_pls,
                                "Precision (weighted)": prec_pls,
                                "Recall (weighted)": rec_pls,
                                "F1-Score (weighted)": f1_pls
                            })

                        # ===== 📊 Model Performance Summary (selected LV) + Download (one line) =====
                        col1, col2 = st.columns([4, 1])

                        with col1:
                            st.markdown("### 📊 Model Performance Summary (selected PLS)")

                        with col2:
                            st.download_button(
                                "📥 Download CSV",
                                data=pd.DataFrame(results_pls).to_csv(index=False).encode("utf-8"),
                                file_name="pls_selected_lv_model_metrics.csv",
                                mime="text/csv",
                                use_container_width=True
                            )

                        results_pls_df = pd.DataFrame(results_pls)
                        st.dataframe(
                            results_pls_df.style
                            .background_gradient(cmap="YlGnBu")
                            .format({
                                "Accuracy": "{:.3f}",
                                "Precision (weighted)": "{:.3f}",
                                "Recall (weighted)": "{:.3f}",
                                "F1-Score (weighted)": "{:.3f}"
                            }),
                            use_container_width=True
                        )

                        # ---- Detailed results per model ----
                        for mdl_name in selected_models_pls:
                            st.markdown(f"### 🔎 Details: {mdl_name} on {lv_label}")
                            report_col_pls, matrix_col_pls = st.columns([2, 3])

                            clf_pls = _make_model_pls(mdl_name)
                            clf_pls.fit(X_train_pls, y_train_pls)
                            y_pred_pls = clf_pls.predict(X_test_pls)

                            with report_col_pls:
                                st.markdown("**Classification Report**")
                                report_dict_pls = classification_report(
                                    y_test_pls, y_pred_pls, target_names=list(le_local_pls.classes_), output_dict=True,
                                    zero_division=0
                                )
                                report_df_pls = pd.DataFrame(report_dict_pls).transpose()
                                styled_report_pls = report_df_pls.style.set_table_styles([
                                    {'selector': 'th', 'props': [('border', '1px solid black'), ('padding', '8px'),
                                                                 ('background', '#f2f2f2')]},
                                    {'selector': 'td', 'props': [('border', '1px solid black'), ('padding', '8px')]},
                                    {'selector': 'tr', 'props': [('border', '1px solid black')]}
                                ]).format("{:.2f}")
                                st.write(styled_report_pls)

                            with matrix_col_pls:
                                # Row-normalized confusion matrix -> percentages
                                cm_pls = confusion_matrix(y_test_pls, y_pred_pls,
                                                          labels=np.arange(len(le_local_pls.classes_)))
                                with np.errstate(divide="ignore", invalid="ignore"):
                                    cm_pct_pls = cm_pls.astype(float) / cm_pls.sum(axis=1, keepdims=True) * 100
                                cm_pct_pls = np.nan_to_num(cm_pct_pls)

                                annot_labels_pls = np.array([[f"{v:.1f}%" for v in row] for row in cm_pct_pls])

                                fig_cm_pls, ax_pls = plt.subplots()
                                sns.heatmap(
                                    cm_pct_pls,
                                    annot=annot_labels_pls,
                                    fmt="",
                                    cmap="Blues",
                                    vmin=0, vmax=100,
                                    xticklabels=list(le_local_pls.classes_),
                                    yticklabels=list(le_local_pls.classes_),
                                    cbar_kws={"label": "% of true class"},
                                    ax=ax_pls
                                )
                                ax_pls.set_xlabel("Predicted")
                                ax_pls.set_ylabel("True")
                                ax_pls.set_title(f"{mdl_name} — Confusion Matrix (%) on {lv_label}")
                                st.pyplot(fig_cm_pls)

                    ############################## [  ] ############################################################
                    # ============================== 📥 Download PLS Full Workbook (editable loadings chart, 2D chart, 3D, summaries, all confusion matrices) ==============================

                    import io, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
                    # ============================== One-file export for the entire "🔬 PLS: Partial Least Squares" ==============================

                    import io, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
                    from sklearn.preprocessing import LabelEncoder
                    from sklearn.model_selection import train_test_split
                    from sklearn.metrics import classification_report, confusion_matrix
                    from sklearn.svm import SVC
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.neighbors import KNeighborsClassifier
                    from sklearn.ensemble import RandomForestClassifier

                    try:
                        from xgboost import XGBClassifier

                        _has_xgb = True
                    except Exception:
                        _has_xgb = False


                    # ---------- helper: auto-fit Excel columns ----------
                    def _autosize_columns(ws, df, extra_pad=2, max_width=60):
                        for i, col in enumerate(df.columns):
                            series = df[col].astype(str)
                            max_len = max([len(str(col))] + [len(s) for s in series.tolist()])
                            ws.set_column(i, i, min(max_len + extra_pad, max_width))


                    def _autosize_range(ws, first_col, last_col, rows, headers=None, extra_pad=2, max_width=60):
                        # Auto-fit arbitrary cell blocks (for Model_Metrics where we write cells)
                        for c in range(first_col, last_col + 1):
                            values = []
                            if headers and c - first_col < len(headers):
                                values.append(str(headers[c - first_col]))
                            for r in rows:
                                v = r.get(c, "")
                                values.append("" if v is None else str(v))
                            width = min(max(len(x) for x in values) + extra_pad, max_width) if values else 10
                            ws.set_column(c, c, width)


                    # ---------- (A) Rebuild model results (AUTO-RUN; independent of any checkbox) ----------
                    # Use cumulative PLS1..selected for export
                    _sel_lv_idx = int(str(selected_lv_loading).replace("PLS", "")) - 1

                    # Get PLS scores (T); if not present, compute via pls.transform(X_num)
                    try:
                        T_all = pls.x_scores_
                    except Exception:
                        # Fallback if scores not cached; assumes X_num (numeric X) exists, else try X
                        X_for_tr = X_num if 'X_num' in globals() else (
                            X.values if hasattr(X, "values") else np.asarray(X, dtype=float))
                        T_all = pls.transform(X_for_tr)

                    X_pls_export = T_all[:, :_sel_lv_idx + 1]  # cumulative PLS1..selected
                    lv_label_export = f"PLS1–PLS{_sel_lv_idx + 1}"

                    # Labels
                    le_export = LabelEncoder()
                    y_codes_export = le_export.fit_transform(y if not hasattr(y, "values") else y.values)

                    # Split
                    X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(
                        X_pls_export, y_codes_export, test_size=0.2, stratify=y_codes_export, random_state=42
                    )

                    # Models
                    models = {
                        "SVM": SVC(),
                        "Logistic Regression": LogisticRegression(max_iter=1000),
                        "KNN": KNeighborsClassifier(n_neighbors=3),
                        "Random Forest": RandomForestClassifier(n_estimators=100),
                    }
                    if _has_xgb:
                        models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric="logloss")

                    results_rows = []
                    confmats_counts = {}
                    confmats_pct = {}
                    confmat_heatmap_png = {}

                    for name, clf in models.items():
                        clf.fit(X_train_e, y_train_e)
                        y_pred = clf.predict(X_test_e)

                        acc = clf.score(X_test_e, y_test_e)
                        rep = classification_report(
                            y_test_e, y_pred, target_names=list(le_export.classes_), output_dict=True, zero_division=0
                        )
                        prec = rep["weighted avg"]["precision"]
                        rec = rep["weighted avg"]["recall"]
                        f1 = rep["weighted avg"]["f1-score"]

                        results_rows.append({
                            "Model": name,
                            "LV used": lv_label_export,
                            "Accuracy": acc,
                            "Precision (weighted)": prec,
                            "Recall (weighted)": rec,
                            "F1-Score (weighted)": f1
                        })

                        # Confusion matrices
                        cm = confusion_matrix(y_test_e, y_pred, labels=np.arange(len(le_export.classes_)))
                        confmats_counts[name] = pd.DataFrame(cm, index=list(le_export.classes_),
                                                             columns=list(le_export.classes_))

                        with np.errstate(divide="ignore", invalid="ignore"):
                            cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100.0
                        cm_pct = np.nan_to_num(cm_pct)
                        confmats_pct[name] = pd.DataFrame(cm_pct, index=list(le_export.classes_),
                                                          columns=list(le_export.classes_))

                        # Heatmap PNG (in-memory)
                        fig, ax = plt.subplots(figsize=(5, 4))
                        sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                                    xticklabels=list(le_export.classes_), yticklabels=list(le_export.classes_),
                                    cbar=True, ax=ax, vmin=0, vmax=100)
                        ax.set_xlabel("Predicted");
                        ax.set_ylabel("True")
                        ax.set_title(f"{name} — Confusion Matrix (%) on {lv_label_export}")
                        _png = io.BytesIO()
                        fig.tight_layout()
                        fig.savefig(_png, format="png", dpi=180)
                        plt.close(fig)
                        _png.seek(0)
                        confmat_heatmap_png[name] = _png

                    results_export_df = pd.DataFrame(results_rows)

                    # ---------- (B) Loadings PNG for 'Loadings Graph' (cleaner visual) ----------
                    try:
                        _lv_num = int(str(selected_lv_loading).replace("PLS", ""))
                        _var_pct = float(
                            explained_x[_lv_num - 1]) * 100.0  # explained_x should be PLS X-variance ratio per LV
                    except Exception:
                        _var_pct = np.nan

                    _series_vals = pd.to_numeric(loadings[selected_lv_loading], errors="coerce").values
                    _x_idx = np.arange(len(_series_vals))
                    figL, axL = plt.subplots(figsize=(11, 3.2))
                    axL.plot(_x_idx, _series_vals, linewidth=2.8)
                    axL.axhline(0.0, linestyle="--", color="gray", linewidth=1)
                    axL.set_title(f"{selected_lv_loading} Loadings ({_var_pct:.2f}% X-Variance)")
                    axL.set_xlabel("Index (wavenumber order)")
                    axL.set_ylabel("Loading")
                    axL.grid(alpha=0.15)
                    _loadings_png = io.BytesIO()
                    figL.tight_layout()
                    figL.savefig(_loadings_png, format="png", dpi=180)
                    plt.close(figL)
                    _loadings_png.seek(0)

                    # ---------- (C) Data for 2D/3D projections ----------
                    # Build a scores DataFrame with labels for export
                    labels_arr = y if not hasattr(y, "values") else y.values
                    pls_cols = [f"PLS{i + 1}" for i in range(T_all.shape[1])]
                    df_pls_all = pd.DataFrame(T_all, columns=pls_cols)
                    df_pls_all["Label"] = labels_arr

                    # Default axis names (you can pass your UI selections if you have them)
                    lv_x = pls_cols[0] if len(pls_cols) > 0 else "PLS1"
                    lv_y = pls_cols[1] if len(pls_cols) > 1 else lv_x
                    lv_z = pls_cols[2] if len(pls_cols) > 2 else None

                    proj2d_df = df_pls_all[[lv_x, lv_y, "Label"]].copy()
                    if lv_z and lv_z in df_pls_all.columns:
                        proj3d_df = df_pls_all[[lv_x, lv_y, lv_z, "Label"]].copy()
                    else:
                        proj3d_df = pd.DataFrame(columns=[lv_x, lv_y, "Z", "Label"])

                    # Optional PNG for 3D visualization (Excel has no native 3D scatter)
                    try:
                        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

                        if lv_z and lv_z in df_pls_all.columns:
                            fig3, ax3 = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(6, 5))
                            for lab in proj3d_df["Label"].unique():
                                d = proj3d_df[proj3d_df["Label"] == lab]
                                ax3.scatter(d[lv_x], d[lv_y], d[lv_z], label=str(lab), s=16)
                            ax3.set_xlabel(lv_x);
                            ax3.set_ylabel(lv_y);
                            ax3.set_zlabel(lv_z)
                            ax3.set_title("PLS 3D Visualization")
                            ax3.legend(loc="best", fontsize=8)
                            _pls3d_png = io.BytesIO()
                            fig3.tight_layout()
                            fig3.savefig(_pls3d_png, format="png", dpi=180)
                            plt.close(fig3)
                            _pls3d_png.seek(0)
                        else:
                            _pls3d_png = None
                    except Exception:
                        _pls3d_png = None

                    # ---------- (D) Build Excel with all pages & editable charts ----------
                    buffer_full = io.BytesIO()
                    with pd.ExcelWriter(buffer_full, engine="xlsxwriter") as writer:
                        wb = writer.book

                        # --- PLS Summary (if you built one) ---
                        try:
                            pls_calcv_df.to_excel(writer, sheet_name="PLS Summary", index=False)
                            _autosize_columns(writer.sheets["PLS Summary"], pls_calcv_df)
                        except Exception:
                            pass

                        # --- Variable Importance / Drivers / Regions ---
                        try:
                            importance_df.to_excel(writer, sheet_name="Variable Importance", index=False)
                            _autosize_columns(writer.sheets["Variable Importance"], importance_df)
                        except Exception:
                            pass

                        try:
                            _drivers_view = drivers_df[
                                ["Wavenumber (cm⁻¹)", "Loading", "|Loading|", "Band Annotation", "Interpretation"]]
                            _drivers_view.to_excel(writer, sheet_name="Most Influential", index=False)
                            _autosize_columns(writer.sheets["Most Influential"], _drivers_view)
                        except Exception:
                            pass

                        # try:
                        #     contrib_df.to_excel(writer, sheet_name="Regions", index=False)
                        #     _autosize_columns(writer.sheets["Regions"], contrib_df)
                        # except Exception:
                        #     pass

                        # --- 📝 Auto-summary (PLS) -> Excel sheet ---
                        try:
                            # Fallbacks so the sheet is always populated
                            if "summary_lines" not in globals() or not isinstance(summary_lines, list) or len(
                                    summary_lines) == 0:
                                try:
                                    _lv_num = int(str(selected_lv_loading).replace("PLS", ""))
                                except Exception:
                                    _lv_num = 1
                                try:
                                    _var_pct_safe = float(explained_x[_lv_num - 1]) * 100.0
                                except Exception:
                                    _var_pct_safe = float("nan")
                                summary_lines = [
                                    f"- **{selected_lv_loading}** explains **{_var_pct_safe:.2f}%** of X-variance."]

                            if "bullets" not in globals() or not isinstance(bullets, list):
                                bullets = []

                            auto_df_pls = pd.DataFrame({"Summary": summary_lines + ["- **Key drivers:**"] + bullets})
                            auto_df_pls.to_excel(writer, sheet_name="📝 Auto-summary", index=False)

                            # Auto-fit columns if helper is available
                            try:
                                _autosize_columns(writer.sheets["📝 Auto-summary"], auto_df_pls)
                            except Exception:
                                pass
                        except Exception:
                            # Keep exporter resilient
                            pass

                        # --- Loadings (table + Excel-editable chart) ---
                        try:
                            def _num_first(val):
                                s = str(val).strip()
                                if "/" in s:
                                    s = s.split("/")[0].strip()
                                try:
                                    return float(s)
                                except Exception:
                                    return np.nan


                            load_tbl = pd.DataFrame({
                                "Wavenumber (cm⁻¹)": list(loadings.index),
                                "Numeric Wavenumber": [_num_first(v) for v in loadings.index],
                                "Index": np.arange(len(loadings.index)),
                                "Loading": pd.to_numeric(loadings[selected_lv_loading], errors="coerce").values
                            })

                            load_tbl.to_excel(writer, sheet_name="Loadings", index=False)
                            ws_load = writer.sheets["Loadings"]
                            _autosize_columns(ws_load, load_tbl)

                            # Editable line chart (cleaner): thicker stroke, legend bottom
                            chart = wb.add_chart({"type": "line"})
                            n_rows = len(load_tbl)
                            chart.add_series({
                                "name": f"{selected_lv_loading}",
                                "categories": ["Loadings", 1, 1, n_rows, 1],  # Numeric Wavenumber
                                "values": ["Loadings", 1, 3, n_rows, 3],  # Loading
                                "line": {"width": 2.25},
                            })
                            chart.set_title({"name": f"Loadings — {selected_lv_loading} ({_var_pct:.2f}% X-var)"})
                            chart.set_x_axis({"name": "Wavenumber (cm⁻¹)"})
                            chart.set_y_axis({"name": "Loading"})
                            chart.set_legend({"position": "bottom"})
                            ws_load.insert_chart(n_rows + 3, 0, chart)
                        except Exception:
                            pass

                        # --- 📝 Auto-summary (if available) ---
                        try:
                            auto_lines = []
                            auto_lines.extend(summary)
                            auto_lines.append("Key drivers:")
                            auto_lines.extend(bullets)
                            auto_df = pd.DataFrame({"Summary": auto_lines})
                            auto_df.to_excel(writer, sheet_name="📝 Auto-summary", index=False)
                            _autosize_columns(writer.sheets["📝 Auto-summary"], auto_df)
                        except Exception:
                            pass

                        # --- 📉 PLS 2D Projection (data + editable scatter chart; markers only, no lines) ---
                        try:
                            sheet2d = "📉 PLS 2D Projection"
                            proj2d_df.to_excel(writer, sheet_name=sheet2d, index=False)
                            ws2d = writer.sheets[sheet2d]
                            _autosize_columns(ws2d, proj2d_df)

                            chart2d = wb.add_chart({"type": "scatter"})
                            cur_row = len(proj2d_df) + 3
                            for lab in proj2d_df["Label"].unique():
                                sub = proj2d_df[proj2d_df["Label"] == lab][[lv_x, lv_y]].reset_index(drop=True)
                                if sub.empty:
                                    continue
                                ws2d.write(cur_row, 0, f"{lab} — {lv_x}")
                                ws2d.write(cur_row, 1, f"{lab} — {lv_y}")
                                ws2d.write_column(cur_row + 1, 0, sub[lv_x].tolist())
                                ws2d.write_column(cur_row + 1, 1, sub[lv_y].tolist())

                                s_start, s_end = cur_row + 1, cur_row + len(sub)
                                chart2d.add_series({
                                    "name": str(lab),
                                    "categories": [sheet2d, s_start, 0, s_end, 0],
                                    "values": [sheet2d, s_start, 1, s_end, 1],
                                    "marker": {"type": "circle", "size": 5},
                                    "line": {"none": True},
                                })
                                cur_row = s_end + 2

                            chart2d.set_title({"name": f"PLS 2D: {lv_x} vs {lv_y}"})
                            chart2d.set_x_axis({"name": lv_x})
                            chart2d.set_y_axis({"name": lv_y})
                            chart2d.set_legend({"position": "bottom"})
                            ws2d.insert_chart(1, 4, chart2d)
                        except Exception:
                            pass

                        # --- 🧭 PLS 3D Visualization (data + PNG) ---
                        try:
                            sheet3d = "🧭 PLS 3D Visualization"
                            proj3d_df.to_excel(writer, sheet_name=sheet3d, index=False)
                            ws3d = writer.sheets[sheet3d]
                            _autosize_columns(ws3d, proj3d_df)
                            if _pls3d_png is not None:
                                ws3d.insert_image(1, 5, "pls3d.png", {"image_data": _pls3d_png})
                        except Exception:
                            pass

                        # --- Model_Metrics (summary + all confusion matrices + heatmaps on one sheet) ---
                        wsM = wb.add_worksheet("Model_Metrics")

                        headers = list(results_export_df.columns)
                        for c, h in enumerate(headers):
                            wsM.write(0, c, h)
                        for r, row in results_export_df.iterrows():
                            for c, h in enumerate(headers):
                                wsM.write(r + 1, c, row[h])
                        _autosize_columns(wsM, results_export_df)

                        start_row = len(results_export_df) + 3

                        for name in models.keys():
                            wsM.write(start_row, 0, f"{name} — Confusion Matrix (counts)")
                            cnt = confmats_counts[name]
                            for j, col in enumerate([""] + cnt.columns.tolist()):
                                wsM.write(start_row + 1, j, col)
                            for i, idx in enumerate(cnt.index.tolist()):
                                wsM.write(start_row + 2 + i, 0, idx)
                                for j, col in enumerate(cnt.columns.tolist()):
                                    wsM.write(start_row + 2 + i, 1 + j, int(cnt.loc[idx, col]))
                            _autosize_range(
                                wsM,
                                first_col=0,
                                last_col=len(cnt.columns),
                                rows=[{k: None for k in range(len(cnt.columns) + 1)} for _ in
                                      range(len(cnt.index) + 2)],
                                headers=[""] + cnt.columns.tolist()
                            )
                            start_row += len(cnt) + 3

                            wsM.write(start_row, 0, f"{name} — Confusion Matrix (%)")
                            pct = confmats_pct[name]
                            for j, col in enumerate([""] + pct.columns.tolist()):
                                wsM.write(start_row + 1, j, col)
                            for i, idx in enumerate(pct.index.tolist()):
                                wsM.write(start_row + 2 + i, 0, idx)
                                for j, col in enumerate(pct.columns.tolist()):
                                    wsM.write_number(start_row + 2 + i, 1 + j, float(pct.loc[idx, col]))
                            img_col = pct.shape[1] + 3
                            wsM.insert_image(start_row + 1, img_col, f"{name}_cm.png",
                                             {"image_data": confmat_heatmap_png[name], "x_scale": 1.0, "y_scale": 1.0})
                            start_row += len(pct) + 10

                    buffer_full.seek(0)

                    # ================= Download Complete PLS Analysis =================
                    colA, colB = st.columns([4, 1])

                    with colA:
                        st.download_button(
                            "📥 Download PLS Full Workbook (editable loadings chart, 2D chart, 3D, summaries, all confusion matrices)",
                            data=buffer_full.getvalue(),
                            file_name="pls_full_export.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )

                    with colB:
                        st.empty()

            #####################################################################################################################
            ##############    📊 PLS-DA: Partial Least Squares Discriminant Analysis

            if show_plsda:

                # ============================
                # 📊 PLS-DA: Partial Least Squares Discriminant Analysis
                # ============================

                col1, col2 = st.columns([4, 1])  # 4:1 width ratio (adjust as needed)

                with col1:
                    st.subheader("📊 PLS-DA: Partial Least Squares Discriminant Analysis")

                with st.expander(
                        "PLS-DA is a supervised projection method that maximizes separation across labeled groups. This panel provides a summary table, selectable 2D/3D projections, component-wise loadings with bond annotations, and VIP with bond annotations.",
                        expanded=True
                ):

                    ############################################ [ PLS-DA Analysis ] ##############################################################################
                    import numpy as np
                    import pandas as pd
                    import streamlit as st
                    from sklearn.cross_decomposition import PLSRegression
                    from sklearn.model_selection import StratifiedKFold, KFold
                    from sklearn.preprocessing import LabelEncoder
                    from sklearn.metrics import r2_score

                    # ---------- Preprocessing selection (combined) ----------
                    PREPROCESSING_OPTIONS = ["Mean Center", "Autoscale", "Smoothing", "Normalization", "Baseline",
                                             "Second Derivative"]
                    x_preproc_plsda = st.multiselect(
                        "Select Preprocessing Techniques for PLS-DA (applied together, in order)",
                        options=PREPROCESSING_OPTIONS,
                        default=["Autoscale"],
                        key="plsda_preproc"
                    )

                    # ========== Controls (give every widget a unique key) ==========
                    max_plsda = max(2, min(10, X.shape[1]))

                    # Slider: number of PLS-DA components
                    n_comp_da = st.slider(
                        "Number of PLS-DA components (LVs)",
                        2,
                        max_plsda,
                        value=min(5, max_plsda),
                        step=1,
                        key="plsda_n_comp"
                    )

                    cv_splits_plsda = st.slider("CV Splits (PLS-DA)", 3, 15, value=10, step=1, key="plsda_cv_splits")
                    conf_limit_plsda = st.selectbox("Confidence Limit (PLS-DA)", [0.90, 0.95, 0.99], index=1,
                                                    key="plsda_conf_limit")
                    lv_list_plsda = st.multiselect(
                        "Ncomp/LVs to evaluate (PLS-DA)",
                        options=list(range(2, max_plsda + 1)),
                        default=[2, 3],
                        key="plsda_lv_list"
                    )

                    # ===================== helpers: Savitzky–Golay =====================
                    from scipy.signal import savgol_filter


                    def _savgol_params(n_feat: int, window: int, poly: int, deriv: int):
                        w_max_odd = n_feat if (n_feat % 2 == 1) else (n_feat - 1)
                        if w_max_odd < 3:
                            return None
                        w = int(window)
                        if w % 2 == 0: w -= 1
                        if w < 3: w = 3
                        if w > w_max_odd: w = w_max_odd
                        p = int(poly)
                        p = max(p, deriv, 1)
                        if p >= w: p = w - 1
                        return w, p


                    def smooth_savgol(X, window=7, poly=3):
                        Xn = X.values.astype(float) if hasattr(X, "values") else np.asarray(X, dtype=float)
                        Xn = Xn.copy()
                        params = _savgol_params(Xn.shape[1], window, poly, deriv=0)
                        if params is None: return Xn
                        w, p = params
                        return savgol_filter(Xn, window_length=w, polyorder=p, deriv=0, axis=1, mode="interp")

                    def second_derivative_savgol(X, window=7, poly=3):
                        Xn = X.values.astype(float) if hasattr(X, "values") else np.asarray(X, dtype=float)
                        Xn = Xn.copy()
                        params = _savgol_params(Xn.shape[1], window, poly, deriv=2)
                        if params is None: return Xn
                        w, p = params
                        return savgol_filter(Xn, window_length=w, polyorder=p, deriv=2, axis=1, mode="interp")

                    # ===================== helpers: Baseline (ALS, sparse + fast) =====================
                    from scipy.sparse import diags, eye
                    from scipy.sparse.linalg import spsolve


                    def als_baseline_sparse(y, lam=1e5, p=0.001, niter=10, eps=1e-12):
                        y = np.asarray(y, dtype=float).ravel()
                        L = y.size
                        if L < 3:
                            return np.zeros_like(y)
                        D = diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L))
                        DTD = lam * (D.T @ D)
                        w = np.ones(L)
                        I = eye(L) * eps
                        for _ in range(niter):
                            W = diags(w, 0, shape=(L, L))
                            z = spsolve(W + DTD + I, w * y)
                            w = p * (y > z) + (1 - p) * (y < z)
                        return z

                    # ===================== helpers: leakage-safe MC/Autoscale =====================
                    def fit_center_scale_stats(X_arr, need_mean: bool, need_std: bool):
                        mu = sd = None
                        if need_mean or need_std:
                            mu = X_arr.mean(axis=0, keepdims=True)
                        if need_std:
                            sd = X_arr.std(axis=0, keepdims=True)
                            sd = sd.copy();
                            sd[sd == 0] = 1.0
                        return mu, sd

                    def apply_preprocessing_chain(
                            X,
                            ops, *,
                            smooth_window=7, smooth_poly=3,
                            normalization_mode="l2",
                            baseline_lam=100_000.0, baseline_p=0.001,
                            mean_=None, std_=None
                    ):
                        Xn = X.values.astype(float) if hasattr(X, "values") else np.asarray(X, dtype=float)
                        Xn = Xn.copy()

                        # 1) Baseline
                        if "Baseline" in ops:
                            Xcorr = np.empty_like(Xn)
                            for i in range(Xn.shape[0]):
                                bl = als_baseline_sparse(Xn[i, :], lam=baseline_lam, p=baseline_p)
                                Xcorr[i, :] = Xn[i, :] - bl
                            Xn = Xcorr

                        # 2) Smoothing / Second Derivative
                        if "Smoothing" in ops and "Second Derivative" in ops:
                            Xn = second_derivative_savgol(Xn, window=smooth_window, poly=smooth_poly)
                        elif "Smoothing" in ops:
                            Xn = smooth_savgol(Xn, window=smooth_window, poly=smooth_poly)
                        elif "Second Derivative" in ops:
                            Xn = second_derivative_savgol(Xn, window=smooth_window, poly=smooth_poly)

                        # 3) Mean Center / Autoscale
                        if "Mean Center" in ops or "Autoscale" in ops:
                            mu = mean_ if mean_ is not None else Xn.mean(axis=0, keepdims=True)
                            Xn = Xn - mu
                        if "Autoscale" in ops:
                            sd = std_ if std_ is not None else Xn.std(axis=0, keepdims=True)
                            sd = sd.copy();
                            sd[sd == 0] = 1.0
                            Xn = Xn / sd

                        # 4) Normalization (row-wise)
                        if "Normalization" in ops:
                            mode = normalization_mode.lower()
                            if mode == "l2":
                                denom = np.linalg.norm(Xn, axis=1, keepdims=True)
                            elif mode == "l1":
                                denom = np.sum(np.abs(Xn), axis=1, keepdims=True)
                            else:
                                denom = np.max(np.abs(Xn), axis=1, keepdims=True)
                            denom[denom == 0] = 1.0
                            Xn = Xn / denom

                        return Xn

                    # ===================== helpers: labels → one-hot & metrics =====================
                    def _labels_and_onehot():
                        if "y_encoded" in globals():
                            y_raw = globals()["y_encoded"]
                        elif "y" in globals():
                            y_raw = globals()["y"]
                        else:
                            st.error("PLS-DA requires a target vector `y` or `y_encoded` to be defined.")
                            raise RuntimeError("Missing y")

                        y_arr = y_raw.values if hasattr(y_raw, "values") else y_raw
                        y_arr = np.asarray(y_arr)
                        # integer-encode if needed
                        if np.issubdtype(y_arr.dtype, np.integer):
                            y_labels = y_arr.astype(int)
                        else:
                            y_labels = LabelEncoder().fit_transform(y_arr)
                        n_classes = int(np.max(y_labels)) + 1
                        Y_onehot = np.eye(n_classes)[y_labels]
                        return y_labels, Y_onehot

                    def _rmse_r2_bias_matrix(Y_true, Y_pred):
                        Y_true = np.asarray(Y_true, dtype=float)
                        Y_pred = np.asarray(Y_pred, dtype=float)
                        if Y_true.ndim == 1: Y_true = Y_true.reshape(-1, 1)
                        if Y_pred.ndim == 1: Y_pred = Y_pred.reshape(-1, 1)
                        rmse = float(np.sqrt(np.mean((Y_true - Y_pred) ** 2)))
                        r2_vals = []
                        for j in range(Y_true.shape[1]):
                            try:
                                r2_vals.append(r2_score(Y_true[:, j], Y_pred[:, j]))
                            except Exception:
                                r2_vals.append(np.nan)
                        r2_mean = float(np.nanmean(r2_vals))
                        bias = float(np.mean(Y_pred - Y_true))
                        return rmse, r2_mean, bias

                    # ===================== CAL & CV (PLS-DA with PLS-style metrics) =====================
                    try:
                        y_labels, Y_onehot = _labels_and_onehot()
                    except RuntimeError:
                        st.stop()

                    rows_plsda = []
                    model_id = 1
                    preproc_label_plsda = ", ".join(x_preproc_plsda) if x_preproc_plsda else "None"
                    X_size_da = f"{X.shape[0]} x {X.shape[1]}"
                    Y_size_da = f"{Y_onehot.shape[0]} x {Y_onehot.shape[1]}"

                    # Preprocess ALL data for CAL pass
                    X_all = apply_preprocessing_chain(
                        X, x_preproc_plsda,
                        smooth_window=7, smooth_poly=3,
                        baseline_lam=100_000.0, baseline_p=0.001
                    )

                    for ncomp in lv_list_plsda:
                        # ===== CAL =====
                        pls_all = PLSRegression(n_components=ncomp)
                        pls_all.fit(X_all, Y_onehot)
                        Yhat_all = pls_all.predict(X_all)
                        rmsec, r2c, bias_c = _rmse_r2_bias_matrix(Y_onehot, Yhat_all)

                        # ===== CV (leakage-safe preprocessing per fold) =====
                        try:
                            fold_iter = StratifiedKFold(n_splits=cv_splits_plsda, shuffle=True, random_state=42).split(
                                X, y_labels)
                        except Exception:
                            fold_iter = KFold(n_splits=cv_splits_plsda, shuffle=True, random_state=42).split(X)

                        Y_pred_cv = np.zeros_like(Y_onehot, dtype=float)

                        for tr_idx, te_idx in fold_iter:
                            X_tr = X.iloc[tr_idx].values.astype(float)
                            X_te = X.iloc[te_idx].values.astype(float)
                            Y_tr = Y_onehot[tr_idx, :]

                            need_mean = ("Mean Center" in x_preproc_plsda) or ("Autoscale" in x_preproc_plsda)
                            need_std = ("Autoscale" in x_preproc_plsda)
                            mu_tr, sd_tr = fit_center_scale_stats(X_tr, need_mean, need_std)

                            X_tr_s = apply_preprocessing_chain(
                                X_tr, x_preproc_plsda,
                                smooth_window=7, smooth_poly=3,
                                baseline_lam=100_000.0, baseline_p=0.001,
                                mean_=mu_tr, std_=sd_tr
                            )
                            X_te_s = apply_preprocessing_chain(
                                X_te, x_preproc_plsda,
                                smooth_window=7, smooth_poly=3,
                                baseline_lam=100_000.0, baseline_p=0.001,
                                mean_=mu_tr, std_=sd_tr
                            )

                            pls_cv = PLSRegression(n_components=ncomp)
                            pls_cv.fit(X_tr_s, Y_tr)
                            Y_pred_cv[te_idx, :] = pls_cv.predict(X_te_s)

                        rmsecv, r2cv, bias_cv = _rmse_r2_bias_matrix(Y_onehot, Y_pred_cv)
                        rmse_ratio = rmsecv / rmsec if rmsec > 0 else float("nan")

                        rows_plsda.append({
                            "CV Splits": cv_splits_plsda,
                            "Confidence Limit": conf_limit_plsda,
                            "RMSE Ratio (RMSECV/RMSEC)": rmse_ratio,
                            "RMSEC (Cal)": rmsec,
                            "RMSECV (CV)": rmsecv,
                            "Model Name": f"Model {model_id}",
                            "Ncomp/LVs": ncomp,
                            "X-Preprocessing": preproc_label_plsda,
                            "X Include Size": X_size_da,
                            "Y Include Size": Y_size_da,
                            "R2C (Cal)": r2c,
                            "R2CV (CV)": r2cv,
                            "Bias": bias_c,
                            "Bias (CV)": bias_cv
                        })
                        model_id += 1

                    # ---------------- Table + download ----------------
                    plsda_calcv_df = pd.DataFrame(rows_plsda, columns=[
                        "CV Splits", "Confidence Limit", "RMSE Ratio (RMSECV/RMSEC)",
                        "RMSEC (Cal)", "RMSECV (CV)", "Model Name", "Ncomp/LVs",
                        "X-Preprocessing", "X Include Size", "Y Include Size",
                        "R2C (Cal)", "R2CV (CV)", "Bias", "Bias (CV)"
                    ])

                    fmt_cols_da = ["RMSE Ratio (RMSECV/RMSEC)", "RMSEC (Cal)", "RMSECV (CV)", "R2C (Cal)", "R2CV (CV)",
                                   "Bias", "Bias (CV)"]

                    st.markdown("### 📋 PLS-DA Analysis Result Summary (PLS-style metrics)")
                    st.dataframe(
                        plsda_calcv_df.style
                        .format({c: "{:.4f}" for c in fmt_cols_da if c in plsda_calcv_df.columns})
                        .background_gradient(
                            subset=[c for c in ["R2CV (CV)", "R2C (Cal)"] if c in plsda_calcv_df.columns], cmap="YlGn")
                        .background_gradient(
                            subset=[c for c in ["RMSECV (CV)", "RMSEC (Cal)"] if c in plsda_calcv_df.columns],
                            cmap="YlOrRd_r"),
                        use_container_width=True
                    )

                    st.download_button(
                        "📥 Download PLS-DA Summary (CSV)",
                        data=plsda_calcv_df.to_csv(index=False).encode("utf-8"),
                        file_name="plsda_summary_table.csv",
                        mime="text/csv",
                        key="plsda_download"
                    )

                    ################################################################################################################################################

                    # ===== Prepare X, y (one-hot) =====
                    scaler_da = StandardScaler()
                    X_scaled_da = scaler_da.fit_transform(X)

                    le_da = LabelEncoder()
                    y_codes_all = le_da.fit_transform(y)
                    y_onehot = pd.get_dummies(y_codes_all)  # (n_samples, n_classes)

                    # ===== Fit PLS-DA (PLSRegression with one-hot Y) =====
                    pls_da = PLSRegression(n_components=n_comp_da)
                    pls_da.fit(X_scaled_da, y_onehot.values)

                    # Matrices
                    T_da = pls_da.x_scores_  # (n_samples, n_comp_da)
                    U_da = pls_da.y_scores_  # (n_samples, n_comp_da)
                    P_da = pls_da.x_loadings_  # (n_features, n_comp_da)
                    W_da = pls_da.x_weights_  # (n_features, n_comp_da)
                    Q_da = pls_da.y_loadings_  # (n_classes, n_comp_da)

                    # ===== Axis Selection for 2D/3D =====
                    all_da_labels = [f"PLSDA{i + 1}" for i in range(n_comp_da)]
                    colx, coly, colz = st.columns(3)
                    with colx:
                        da_x_label = st.selectbox("X-axis", all_da_labels, index=0, key="plsda_x_axis")
                    with coly:
                        da_y_label = st.selectbox("Y-axis (2D/3D)", all_da_labels, index=min(1, n_comp_da - 1),
                                                  key="plsda_y_axis")
                    with colz:
                        da_z_label = st.selectbox("Z-axis (3D)", all_da_labels, index=min(2, n_comp_da - 1),
                                                  key="plsda_z_axis")

                    da_x_idx = int(da_x_label.replace("PLSDA", "")) - 1
                    da_y_idx = int(da_y_label.replace("PLSDA", "")) - 1
                    da_z_idx = int(da_z_label.replace("PLSDA", "")) - 1

                    # Build projection DF (include all comps so selections work)
                    df_plsda = pd.DataFrame({"Label": y.values})
                    for i in range(n_comp_da):
                        df_plsda[f"PLSDA{i + 1}"] = T_da[:, i]

                    # # ===== 2D Projection =====

                    st.subheader("📉 PLS-DA 2D Projection")
                    xcol = f"PLSDA{da_x_idx + 1}"
                    ycol = f"PLSDA{da_y_idx + 1}"

                    fig_plsda_2d = px.scatter(
                        df_plsda,
                        x=xcol,
                        y=ycol,
                        color="Label",
                        title=f"PLS-DA Projection ({da_x_label} vs {da_y_label})",
                        labels={xcol: da_x_label, ycol: da_y_label}
                    )

                    # ---- Add class ellipses (same approach as PCA/PLS) ----
                    chi2_map = {0.90: 4.605, 0.95: 5.991, 0.99: 9.210}
                    chi2_val = chi2_map.get(conf_limit_plsda, 5.991)  # default 95%


                    def _ellipse_xy(points_2d, chi2_q, npts=200):
                        pts = np.asarray(points_2d)
                        if pts.shape[0] < 3:
                            return None, None
                        mu = pts.mean(axis=0)
                        S = np.cov(pts.T)
                        vals, vecs = np.linalg.eigh(S)
                        order = np.argsort(vals)[::-1]
                        vals, vecs = vals[order], vecs[:, order]
                        radii = np.sqrt(np.maximum(vals, 1e-12) * chi2_q)
                        t = np.linspace(0, 2 * np.pi, npts)
                        circ = np.stack([np.cos(t) * radii[0], np.sin(t) * radii[1]], axis=0)
                        ell = (vecs @ circ).T + mu
                        return ell[:, 0], ell[:, 1]


                    for lab in df_plsda["Label"].unique():
                        d = df_plsda[df_plsda["Label"] == lab][[xcol, ycol]].values
                        ex, ey = _ellipse_xy(d, chi2_val)
                        if ex is not None:
                            fig_plsda_2d.add_trace(
                                go.Scatter(
                                    x=ex, y=ey, mode="lines",
                                    name=f"{lab} {int(conf_limit_plsda * 100)}% ellipse",
                                    line=dict(width=1),
                                    showlegend=False
                                )
                            )

                    st.plotly_chart(fig_plsda_2d, use_container_width=True)

                    # ===== 3D Projection =====
                    if n_comp_da >= 3:
                        st.subheader("🧭 PLS-DA 3D Visualization")
                        fig_plsda_3d = px.scatter_3d(
                            df_plsda,
                            x=f"PLSDA{da_x_idx + 1}",
                            y=f"PLSDA{da_y_idx + 1}",
                            z=f"PLSDA{da_z_idx + 1}",
                            color="Label",
                            title=f"3D PLS-DA: {da_x_label} vs {da_y_label} vs {da_z_label}",
                            height=520, width=800
                        )
                        st.plotly_chart(fig_plsda_3d, use_container_width=True)
                    else:
                        st.info("Add ≥3 components to view the 3D projection.")

                    # ===== Loadings for Selected PLS-DA Component =====
                    st.subheader("📊 PLS-DA Loadings")
                    selected_da_loading = st.selectbox(
                        "Select a component to view its loadings:",
                        all_da_labels, index=0, key="plsda_loading_select"
                    )
                    da_load_idx = int(selected_da_loading.replace("PLSDA", "")) - 1

                    #####################################################################################################

                    fig_da_load = go.Figure()
                    fig_da_load.add_trace(go.Scatter(
                        x=selected_columns,
                        y=P_da[:, da_load_idx],
                        mode="lines",
                        name=f"{selected_da_loading} Loadings"
                    ))
                    fig_da_load.add_hline(y=0, line=dict(dash="dash"))

                    # Bond annotations (filtered by selected range)
                    band_annotations_da = {
                        1740: "C=O Stretch (Lipid)",
                        1650: "Amide I (Protein)",
                        1550: "Amide II (Protein)",
                        1338: "CH₂ (Collagen)",
                        1200: "C–O Stretch (Matrix)",
                        1115: "HPO₄²⁻ (Mineral)",
                        1060: "Sugar Ring C–O",
                        1030: "PO₄³⁻ (Mineral)",
                        875: "CO₃²⁻ Bending",
                        856: "C–S Proteoglycan",
                        7000: "O–H Stretch",
                        6688: "N–H Stretch",
                        5800: "CH₂ Lipid",
                        5200: "O–H Water",
                    }
                    y_top_da = float(np.nanmax(np.abs(P_da[:, da_load_idx])))
                    for wn, lbl in band_annotations_da.items():
                        if selected_range[0] <= wn <= selected_range[1]:
                            fig_da_load.add_vline(x=wn, line=dict(width=1, dash="dash"))
                            fig_da_load.add_annotation(x=wn, y=y_top_da, text=lbl, showarrow=True, yshift=10,
                                                       font=dict(size=9))

                    fig_da_load.update_layout(
                        title=f"Loadings for {selected_da_loading}",
                        xaxis_title="Wavenumber (cm⁻¹)",
                        yaxis_title="Loading",
                        height=400
                    )
                    st.plotly_chart(fig_da_load, use_container_width=True)

                    # ===== VIP (Variable Importance in Projection) with Bond Annotations =====

                    def compute_vip_multiresponse(pls_model):
                        T = pls_model.x_scores_  # (n_samples, A)
                        W = pls_model.x_weights_  # (p, A)
                        Q = pls_model.y_loadings_  # (m, A) for m classes
                        p, A = W.shape

                        ssq_y_comp = []
                        for a in range(A):
                            ss_t = np.sum(T[:, a] ** 2)
                            ss_q = np.sum(Q[:, a] ** 2)
                            ssq_y_comp.append(ss_t * ss_q)
                        ssq_y_comp = np.array(ssq_y_comp)
                        total_ssy = np.sum(ssq_y_comp)

                        vip = np.zeros(p)
                        denom_w = np.sum(W ** 2, axis=0)  # (A,)
                        for j in range(p):
                            weight = (W[j, :] ** 2) / denom_w  # (A,)
                            vip[j] = np.sqrt(p * np.sum(ssq_y_comp * weight) / total_ssy)
                        return vip


                    vip_da = compute_vip_multiresponse(pls_da)

                    vip_df_da = pd.DataFrame({
                        "Wavenumber (cm⁻¹)": selected_columns,
                        "VIP": vip_da,
                        "Abs Loading (selected)": np.abs(P_da[:, da_load_idx])
                    })

                    # ---- Bond annotation utilities ----
                    bond_annotations_table = [
                        (8500, "O–H Stretching and Bending (Water)"),
                        (7000, "O–H Stretching (Water)"),
                        (6688, "N–H Stretching (Protein/Collagen)"),
                        (5800, "CH₂ Stretching (Lipid)"),
                        (5200, "O–H Stretching and Bending (Water)"),
                        (4890, "N–H Bending (Protein/Collagen)"),
                        (4610, "C–H Stretching & Deformation (Protein/Collagen)"),
                        (4310, "Sugar Ring Vibrations (Proteoglycan)"),
                        ((3600, 3200), "O–H Stretching (Water/Hydroxyl)"),
                        ((3500, 3300), "N–H Stretching (Proteins)"),
                        ((3000, 2800), "C–H Stretching (Lipids, CH₂)"),
                        ((1750, 1650), "C=O Stretching (Proteins/Lipids)"),
                        (1550, "Amide II (Proteins)"),
                        (1338, "CH₂ Side Chain Bending (Collagen)"),
                        ((1100, 900), "PO₄³⁻ Stretching (Bone Mineral)"),
                        ((890, 850), "CO₃²⁻ Bending (Carbonate)"),
                        (1740, "C=O Stretching (Ester, Lipids)"),
                        (1650, "Amide I (Proteins)"),
                        (1630, "Water O–H Bending (Water)"),
                        ((1200, 1000), "C–O Stretching (Alcohols/Ethers)"),
                        (1115, "HPO₄²⁻ Stretching (Bone mineral)"),
                        (1060, "Sugar Ring C–O Stretch (Carbohydrates)"),
                        (1030, "PO₄³⁻ Stretching (Bone Mineral)"),
                        (875, "CO₃²⁻ Bending (Carbonates)"),
                        (856, "C–S Bending (Proteoglycans)"),
                        ((900, 800), "Aromatic C–H Bending (Fingerprint region)")
                    ]


                    def get_bond_annotation(wavenumber):
                        for entry in bond_annotations_table:
                            if isinstance(entry[0], tuple):
                                low, high = entry[0]
                                if high <= wavenumber <= low:
                                    return entry[1]
                        for entry in bond_annotations_table:
                            if not isinstance(entry[0], tuple) and entry[0] == int(wavenumber):
                                return entry[1]
                        all_wavenumbers = [val if isinstance(val, int) else (val[0] + val[1]) // 2 for val, _ in
                                           bond_annotations_table]
                        closest_idx = int(np.argmin([abs(wavenumber - wn) for wn in all_wavenumbers]))
                        return bond_annotations_table[closest_idx][1]


                    def extract_first_numeric(value):
                        value_str = str(value).strip()
                        if "/" in value_str:
                            value_str = value_str.split("/")[0].strip()
                        try:
                            return float(value_str)
                        except ValueError:
                            return None

                    vip_df_da["Numeric Wavenumber"] = vip_df_da["Wavenumber (cm⁻¹)"].apply(extract_first_numeric)
                    vip_df_da = vip_df_da.dropna(subset=["Numeric Wavenumber"])
                    vip_df_da["Band Annotation"] = vip_df_da["Numeric Wavenumber"].apply(get_bond_annotation)

                    vip_top_da = vip_df_da.sort_values("VIP", ascending=False).head(20).reset_index(drop=True)

                    # ===== 📊 Variable Importance with Bond Annotations + Download (one line) PLSDA =====
                    # ===== Header + Download =====
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown("### ⭐ Variable Importance with Bond Annotations")
                    with col2:
                        st.download_button(
                            "📥 Download CSV",
                            data=vip_top_da.to_csv(index=False).encode("utf-8"),
                            file_name="plsda_vip_with_bands.csv",
                            mime="text/csv"
                        )
                    # ===== 📊 Display Table =====
                    st.dataframe(
                        vip_top_da.style.format({
                            "VIP": "{:.3f}",
                            "Abs Loading (selected)": "{:.3f}"
                        }),
                        use_container_width=True
                    )

                    ############### PLSDA Most Influential wavenumber
                    # ========================= ⭐ Most influential wavenumbers (by |loading|) — PLS-DA =========================

                    # --- tiny helpers (use your existing ones if already defined) ---
                    def _extract_first_numeric(v):
                        s = str(v).strip()
                        if "/" in s:
                            s = s.split("/")[0].strip()
                        try:
                            return float(s)
                        except Exception:
                            return np.nan


                    # If you already defined `get_bond_annotation` earlier in PLS-DA, you can remove this block
                    bond_annotations_table_da = [
                        (8500, "O–H Stretching and Bending (Water)"),
                        (7000, "O–H Stretching (Water)"),
                        (6688, "N–H Stretching (Protein/Collagen)"),
                        (5800, "CH₂ Stretching (Lipid)"),
                        (5200, "O–H Stretching and Bending (Water)"),
                        (4890, "N–H Bending (Protein/Collagen)"),
                        (4610, "C–H Stretching & Deformation (Protein/Collagen)"),
                        (4310, "Sugar Ring Vibrations (Proteoglycan)"),
                        ((3600, 3200), "O–H Stretching (Water/Hydroxyl)"),
                        ((3500, 3300), "N–H Stretching (Proteins)"),
                        ((3000, 2800), "C–H Stretching (Lipids, CH₂)"),
                        ((1750, 1650), "C=O Stretching (Proteins/Lipids)"),
                        (1550, "Amide II (Proteins)"),
                        (1338, "CH₂ Side Chain Bending (Collagen)"),
                        ((1100, 900), "PO₄³⁻ Stretching (Bone Mineral)"),
                        ((890, 850), "CO₃²⁻ Bending (Carbonate)"),
                        (1740, "C=O Stretching (Ester, Lipids)"),
                        (1650, "Amide I (Proteins)"),
                        (1630, "Water O–H Bending (Water)"),
                        ((1200, 1000), "C–O Stretching (Alcohols/Ethers)"),
                        (1115, "HPO₄²⁻ Stretching (Bone mineral)"),
                        (1060, "Sugar Ring C–O Stretch (Carbohydrates)"),
                        (1030, "PO₄³⁻ Stretching (Bone Mineral)"),
                        (875, "CO₃²⁻ Bending (Carbonates)"),
                        (856, "C–S Bending (Proteoglycans)"),
                        ((900, 800), "Aromatic C–H Bending (Fingerprint region)")
                    ]


                    def get_bond_annotation_da(wavenumber: float) -> str:
                        # ranges first
                        for entry in bond_annotations_table_da:
                            if isinstance(entry[0], tuple):
                                lo, hi = entry[0]
                                if hi <= wavenumber <= lo:
                                    return entry[1]
                        # exact
                        for entry in bond_annotations_table_da:
                            if not isinstance(entry[0], tuple) and entry[0] == int(wavenumber):
                                return entry[1]
                        # nearest
                        centers = [v if isinstance(v, int) else (v[0] + v[1]) / 2 for v, _ in bond_annotations_table_da]
                        idx = int(np.argmin([abs(wavenumber - c) for c in centers]))
                        return bond_annotations_table_da[idx][1]


                    # --- compute class context on the selected LV scores (T_da) ---
                    try:
                        df_scores_da = pd.DataFrame({
                            "Label": np.asarray(y),
                            "LV_Score": T_da[:, da_load_idx]
                        })
                        class_means_da = df_scores_da.groupby("Label", observed=True)["LV_Score"].mean().sort_values(
                            ascending=False)
                        class_pos_da = class_means_da.index[0] if len(class_means_da) else None
                        class_neg_da = class_means_da.index[-1] if len(class_means_da) else None
                    except Exception:
                        class_pos_da = class_neg_da = None


                    def _favor_class_da(value):
                        if class_pos_da is None or class_neg_da is None:
                            return ""
                        return f"favors **{class_pos_da}**" if value > 0 else f"favors **{class_neg_da}**"


                    # --- build drivers_df for the selected PLS-DA loading vector ---
                    load_vec_da = P_da[:, da_load_idx]  # shape: (n_features,)
                    drivers_rows_da = []
                    for label, val in zip(selected_columns, load_vec_da):
                        num = _extract_first_numeric(label)
                        band = get_bond_annotation_da(num) if np.isfinite(num) else ""
                        interp = _favor_class_da(val)
                        drivers_rows_da.append({
                            "Wavenumber (cm⁻¹)": str(label),
                            "Numeric (cm⁻¹)": float(num) if np.isfinite(num) else np.nan,
                            "Loading": float(val) if np.isfinite(val) else np.nan,
                            "|Loading|": float(abs(val)) if np.isfinite(val) else np.nan,
                            "Band Annotation": band,
                            "Interpretation": interp
                        })

                    drivers_df_da = (pd.DataFrame(drivers_rows_da)
                                     .sort_values("|Loading|", ascending=False)
                                     .reset_index(drop=True))

                    # For downstream code that expects `drivers_df`, mirror the variable name (safe alias).
                    drivers_df = drivers_df_da.copy()

                    # ===== Header + Download =====
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown("### ⭐ Most influential wavenumbers (by |loading|)")
                    with col2:
                        st.download_button(
                            "📥 Download CSV",
                            data=drivers_df_da[
                                ["Wavenumber (cm⁻¹)", "Loading", "|Loading|", "Band Annotation", "Interpretation"]]
                            .to_csv(index=False).encode("utf-8"),
                            file_name=f"most_influential_wavenumbers_{selected_da_loading}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                    # ===== Table =====
                    st.dataframe(
                        drivers_df_da[
                            ["Wavenumber (cm⁻¹)", "Loading", "|Loading|", "Band Annotation", "Interpretation"]]
                        .style.format({"Loading": "{:.4f}", "|Loading|": "{:.4f}"}),
                        use_container_width=True
                    )

                    ############## new summary table to run all model on selected pls-da###############

                    ############################# [  Narrative summary (PLS-DA style, parallel to PCA/PLS) ] #######################################################

                    ### ---------- Narrative summary (PLS-DA style, parallel to PCA/PLS) ----------
                    try:
                        var_pct_da = float(explained_x[da_load_idx]) * 100.0
                    except Exception:
                        var_pct_da = np.nan

                    # Positive vs negative contribution (aggregated in current focus)
                    pos_mass_da = float(np.nansum(np.abs(s_focus[s_focus > 0]))) if "s_focus" in globals() else np.nan
                    neg_mass_da = float(np.nansum(np.abs(s_focus[s_focus < 0]))) if "s_focus" in globals() else np.nan

                    contrast_da = (
                        "Positive-weighted regions dominate"
                        if pos_mass_da > neg_mass_da else
                        (
                            "Negative-weighted regions dominate" if neg_mass_da > pos_mass_da else "Balanced positive/negative weights")
                    )

                    summary_da = []

                    # summary_da.append(f"- **{selected_da_loading}** explains **{var_pct_da:.2f}%** of X-variance.")
                    # --- Compute explained X-variance ratio for PLS-DA manually ---
                    try:
                        X_residual = X_scaled_da.copy()
                        ss_total = np.sum(np.var(X_residual, axis=0, ddof=1))  # total variance in X
                        ss_comp = []

                        for a in range(pls_da.x_scores_.shape[1]):
                            t = pls_da.x_scores_[:, [a]]  # LV score
                            p = pls_da.x_loadings_[:, [a]].T  # LV loading
                            X_hat = t @ p  # variance explained by this LV
                            ss_comp.append(np.sum(np.var(X_hat, axis=0, ddof=1)))
                            X_residual = X_residual - X_hat  # subtract variance explained

                        explained_x_da = np.array(ss_comp) / ss_total
                    except Exception as e:
                        explained_x_da = np.full(pls_da.x_scores_.shape[1], np.nan)

                    # --- Use this in summary ---
                    try:
                        var_pct_da = float(explained_x_da[da_load_idx]) * 100.0
                    except Exception:
                        var_pct_da = np.nan

                    summary_da.append(f"- **{selected_da_loading}** explains **{var_pct_da:.2f}%** of X-variance.")

                    summary_da.append(f"- **Contrast:** {contrast_da} in the current range.")
                    if class_pos_da and class_neg_da:
                        summary_da.append(
                            f"- **Class context:** Higher LV scores (positive side) align with **{class_pos_da}**; "
                            f"lower scores (negative side) align with **{class_neg_da}**."
                        )
                    if "contrib_df" in globals() and len(contrib_df):
                        top_reg_da = contrib_df.iloc[0]
                        summary_da.append(
                            f"- **Strongest region:** {top_reg_da['Region (cm⁻¹)']} "
                            f"({top_reg_da['Net sign']}, |Σloading|={top_reg_da['|Σ loading|']:.4f}); "
                            f"likely bands: {top_reg_da['Likely bands'] or '—'}."
                        )

                    # Collect top driver bullets
                    head_n_da = min(6, len(drivers_df_da))
                    bullets_da = []
                    for _, r in drivers_df_da.head(head_n_da).iterrows():
                        side = "positive" if r["Loading"] > 0 else "negative"
                        interp = f" — {r['Interpretation']}" if r["Interpretation"] else ""
                        bullets_da.append(
                            f"  • {r['Wavenumber (cm⁻¹)']}: **{side}** loading ({r['Band Annotation']}){interp}"
                        )

                    # ===== Render summary in Streamlit =====
                    st.markdown("### 📝 Auto-summary (PLS-DA)")
                    st.markdown("\n".join(summary_da + ["- **Key drivers:**"] + bullets_da))

                    st.caption(
                        "Notes: (1) PLS-DA LV axis sign is arbitrary; interpretation uses relative +/− sides and class means. "
                        "(2) Region aggregation uses |loading| within ±half-width around top wavenumbers. "
                        "(3) Preprocessing (e.g., autoscale, derivatives) affects which bands dominate."
                    )

                    # ======================= TOGGLE: Run models on selected PLS-DA LV =======================
                    run_plsda_clf = st.checkbox("📊 Model Performance Summary for selected PLS-DA", value=True,key="plsda_run_selected_lv")

                    if run_plsda_clf:
                        # --- choose 1+ models (all selected by default) ---
                        model_choices_da = ["SVM", "Logistic Regression", "KNN", "Random Forest", "XGBoost"]
                        selected_models_da = st.multiselect(
                            "Select classification models (trained on the selected PLS-DA score)",
                            options=model_choices_da,
                            default=model_choices_da,
                            key="plsda_selected_lv_models"
                        )

                        # parse LV index, e.g. "PLSDA3" -> 2
                        sel_da_idx = int(str(selected_da_loading).replace("PLSDA", "")) - 1

                        # 1D feature = selected PLS-DA X-score
                        X_da = T_da[:, sel_da_idx].reshape(-1, 1)

                        # encode labels (keeps names for reports)
                        from sklearn.preprocessing import LabelEncoder

                        le_local_da = LabelEncoder()
                        y_codes_local_da = le_local_da.fit_transform(y)

                        # stratified split
                        from sklearn.model_selection import train_test_split

                        X_train_da, X_test_da, y_train_da, y_test_da = train_test_split(
                            X_da, y_codes_local_da, test_size=0.2, stratify=y_codes_local_da, random_state=42
                        )

                        # helper to build a model by name
                        def _make_model(name):
                            if name == "SVM":
                                return SVC()
                            if name == "Logistic Regression":
                                return LogisticRegression(max_iter=1000)
                            if name == "KNN":
                                return KNeighborsClassifier(n_neighbors=3)
                            if name == "Random Forest":
                                return RandomForestClassifier(n_estimators=100)
                            # XGBoost
                            return XGBClassifier(use_label_encoder=False, eval_metric="logloss")

                        # run all selected models
                        results_da = []
                        for mdl_name in selected_models_da:
                            clf_da = _make_model(mdl_name)
                            clf_da.fit(X_train_da, y_train_da)
                            y_pred_da = clf_da.predict(X_test_da)

                            # metrics (weighted for multi-class safety)
                            acc_da = clf_da.score(X_test_da, y_test_da)
                            prec_da = precision_score(y_test_da, y_pred_da, average="weighted", zero_division=0)
                            rec_da = recall_score(y_test_da, y_pred_da, average="weighted", zero_division=0)
                            f1_da = f1_score(y_test_da, y_pred_da, average="weighted", zero_division=0)

                            results_da.append({
                                "Model": mdl_name,
                                "LV used": selected_da_loading,
                                "Accuracy": acc_da,
                                "Precision (weighted)": prec_da,
                                "Recall (weighted)": rec_da,
                                "F1-Score (weighted)": f1_da
                            })

                        # ====== Colorful combined summary table + optional bar chart ======

                        results_da_df = pd.DataFrame(results_da)

                        # ===== 📊 Model Performance Summary (selected PC) + Download (one line) =====
                        col1, col2 = st.columns([4, 1])

                        with col1:
                            st.markdown("### 📊 Model Performance Summary (selected PLS-DA LV)")

                        with col2:
                            st.download_button(
                                "📥 Download CSV ", results_da_df.to_csv(index=False).encode("utf-8"),
                            file_name="plsda_selected_lv_model_metrics.csv",
                            mime="text/csv"
                        )

                        st.dataframe(
                            results_da_df.style
                            .background_gradient(cmap="YlGnBu")
                            .format({
                                "Accuracy": "{:.3f}",
                                "Precision (weighted)": "{:.3f}",
                                "Recall (weighted)": "{:.3f}",
                                "F1-Score (weighted)": "{:.3f}"
                            }),
                            use_container_width=True
                        )


                        # detailed outputs per model
                        for mdl_name in selected_models_da:
                            st.markdown(f"### 🔎 Details: {mdl_name} on {selected_da_loading}")
                            col_rep, col_cm = st.columns([2, 3])

                            # rebuild + predict (keeps code simple; small 1D fit)
                            clf_da = _make_model(mdl_name)
                            clf_da.fit(X_train_da, y_train_da)
                            y_pred_da = clf_da.predict(X_test_da)

                            with col_rep:
                                st.markdown("**Classification Report**")
                                rep_dict = classification_report(
                                    y_test_da, y_pred_da,
                                    target_names=list(le_local_da.classes_),
                                    output_dict=True, zero_division=0
                                )
                                rep_df = pd.DataFrame(rep_dict).transpose()
                                rep_styled = rep_df.style.set_table_styles([
                                    {'selector': 'th', 'props': [('border', '1px solid black'), ('padding', '8px'),
                                                                 ('background', '#f2f2f2')]},
                                    {'selector': 'td', 'props': [('border', '1px solid black'), ('padding', '8px')]},
                                    {'selector': 'tr', 'props': [('border', '1px solid black')]}
                                ]).format("{:.2f}")
                                st.write(rep_styled)

                            with col_cm:
                                st.markdown("**Confusion Matrix (% by true class)**")
                                cm = confusion_matrix(y_test_da, y_pred_da, labels=np.arange(len(le_local_da.classes_)))
                                with np.errstate(divide="ignore", invalid="ignore"):
                                    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
                                cm_pct = np.nan_to_num(cm_pct)
                                annot = np.array([[f"{v:.1f}%" for v in row] for row in cm_pct])

                                fig_cm, ax = plt.subplots()
                                sns.heatmap(
                                    cm_pct, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=100,
                                    xticklabels=list(le_local_da.classes_),
                                    yticklabels=list(le_local_da.classes_),
                                    cbar_kws={"label": "% of true class"},
                                    ax=ax
                                )
                                ax.set_xlabel("Predicted")
                                ax.set_ylabel("True")
                                ax.set_title(f"{mdl_name} — Confusion Matrix (%) on {selected_da_loading}")
                                st.pyplot(fig_cm)
                    # ===================================================================================================



                    ############################## [  ] ############################################################
                    # ============================== 📥 Download PLSDA Full Workbook (editable loadings chart, 2D chart, 3D, summaries, all confusion matrices) ==============================

                    import io, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
                    # ============================== One-file export for the entire "🔬 PLS: Partial Least Squares" ==============================
                    # ============================== One-file export for the entire "📈 PLS-DA: Partial Least Squares Discriminant Analysis" ==============================

                    import io, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
                    from sklearn.preprocessing import LabelEncoder
                    from sklearn.model_selection import train_test_split
                    from sklearn.metrics import classification_report, confusion_matrix
                    from sklearn.svm import SVC
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.neighbors import KNeighborsClassifier
                    from sklearn.ensemble import RandomForestClassifier

                    try:
                        from xgboost import XGBClassifier

                        _has_xgb = True
                    except Exception:
                        _has_xgb = False


                    # ---------- helper: auto-fit Excel columns ----------
                    def _autosize_columns(ws, df, extra_pad=2, max_width=60):
                        for i, col in enumerate(df.columns):
                            series = df[col].astype(str)
                            max_len = max([len(str(col))] + [len(s) for s in series.tolist()])
                            ws.set_column(i, i, min(max_len + extra_pad, max_width))


                    def _autosize_range(ws, first_col, last_col, rows, headers=None, extra_pad=2, max_width=60):
                        for c in range(first_col, last_col + 1):
                            values = []
                            if headers and c - first_col < len(headers):
                                values.append(str(headers[c - first_col]))
                            for r in rows:
                                v = r.get(c, "")
                                values.append("" if v is None else str(v))
                            width = min(max(len(x) for x in values) + extra_pad, max_width) if values else 10
                            ws.set_column(c, c, width)


                    # ---------- (A) Rebuild model results (AUTO-RUN; cumulative PLS-DA scores up to selected) ----------
                    # Parse k from "PLSDAk"
                    try:
                        _sel_da_idx = int(str(selected_da_loading).replace("PLSDA", "")) - 1
                    except Exception:
                        _sel_da_idx = 0

                    # Cumulative feature matrix: PLSDA1..selected
                    # (T_da shape: (n_samples, n_comp_da))
                    X_da_export = T_da[:, :_sel_da_idx + 1]
                    da_label_export = f"PLSDA1–PLSDA{_sel_da_idx + 1}"

                    # Encode labels
                    le_export_da = LabelEncoder()
                    y_codes_export_da = le_export_da.fit_transform(
                        df_plsda["Label"] if "Label" in df_plsda.columns else y)

                    # Split
                    X_train_e_da, X_test_e_da, y_train_e_da, y_test_e_da = train_test_split(
                        X_da_export, y_codes_export_da, test_size=0.2, stratify=y_codes_export_da, random_state=42
                    )

                    models_da = {
                        "SVM": SVC(),
                        "Logistic Regression": LogisticRegression(max_iter=1000),
                        "KNN": KNeighborsClassifier(n_neighbors=3),
                        "Random Forest": RandomForestClassifier(n_estimators=100),
                    }
                    if _has_xgb:
                        models_da["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric="logloss")

                    results_rows_da = []
                    confmats_counts_da = {}
                    confmats_pct_da = {}
                    confmat_heatmap_png_da = {}

                    for name, clf in models_da.items():
                        clf.fit(X_train_e_da, y_train_e_da)
                        y_pred = clf.predict(X_test_e_da)

                        acc = clf.score(X_test_e_da, y_test_e_da)
                        rep = classification_report(
                            y_test_e_da, y_pred, target_names=list(le_export_da.classes_), output_dict=True,
                            zero_division=0
                        )
                        prec = rep["weighted avg"]["precision"]
                        rec = rep["weighted avg"]["recall"]
                        f1 = rep["weighted avg"]["f1-score"]

                        results_rows_da.append({
                            "Model": name,
                            "LVs used": da_label_export,
                            "Accuracy": acc,
                            "Precision (weighted)": prec,
                            "Recall (weighted)": rec,
                            "F1-Score (weighted)": f1
                        })

                        # Confusion matrices
                        cm = confusion_matrix(y_test_e_da, y_pred, labels=np.arange(len(le_export_da.classes_)))
                        confmats_counts_da[name] = pd.DataFrame(cm, index=list(le_export_da.classes_),
                                                                columns=list(le_export_da.classes_))

                        with np.errstate(divide="ignore", invalid="ignore"):
                            cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100.0
                        cm_pct = np.nan_to_num(cm_pct)
                        confmats_pct_da[name] = pd.DataFrame(cm_pct, index=list(le_export_da.classes_),
                                                             columns=list(le_export_da.classes_))

                        # Heatmap PNG (in-memory)
                        fig, ax = plt.subplots(figsize=(5, 4))
                        sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                                    xticklabels=list(le_export_da.classes_), yticklabels=list(le_export_da.classes_),
                                    cbar=True, ax=ax, vmin=0, vmax=100)
                        ax.set_xlabel("Predicted");
                        ax.set_ylabel("True")
                        ax.set_title(f"{name} — Confusion Matrix (%) on {da_label_export}")
                        _png = io.BytesIO()
                        fig.tight_layout()
                        fig.savefig(_png, format="png", dpi=180)
                        plt.close(fig)
                        _png.seek(0)
                        confmat_heatmap_png_da[name] = _png

                    results_export_da = pd.DataFrame(results_rows_da)

                    # ---------- (B) Loadings PNG for 'Loadings Graph' ----------
                    # Compute %X-variance for selected PLS-DA LV (fallback if attribute missing)
                    try:
                        _da_num = int(str(selected_da_loading).replace("PLSDA", ""))
                    except Exception:
                        _da_num = 1

                    try:
                        _xexp = getattr(pls_da, "x_explained_variance_ratio_", None)
                        if _xexp is not None:
                            _var_pct_da = float(_xexp[_da_num - 1]) * 100.0
                        else:
                            # fallback: variance of scores relative to total X variance
                            X_num_all = X.values.astype(float) if hasattr(X, "values") else np.asarray(X, dtype=float)
                            X_tot = float(np.var(X_num_all, axis=0).sum()) or 1.0
                            _var_pct_da = float(np.var(T_da[:, _da_num - 1])) / X_tot * 100.0
                    except Exception:
                        _var_pct_da = np.nan

                    # Build a simple index-based loadings series for the selected LV
                    _da_load_idx = _da_num - 1
                    _series_vals_da = pd.to_numeric(pd.Series(P_da[:, _da_load_idx]), errors="coerce").values
                    _x_idx_da = np.arange(len(_series_vals_da))
                    figLd, axLd = plt.subplots(figsize=(11, 3.2))
                    axLd.plot(_x_idx_da, _series_vals_da, linewidth=2.8)
                    axLd.axhline(0.0, linestyle="--", color="gray", linewidth=1)
                    axLd.set_title(f"{selected_da_loading} Loadings ({_var_pct_da:.2f}% X-Variance)")
                    axLd.set_xlabel("Index (wavenumber order)")
                    axLd.set_ylabel("Loading")
                    axLd.grid(alpha=0.15)
                    _loadings_png_da = io.BytesIO()
                    figLd.tight_layout()
                    figLd.savefig(_loadings_png_da, format="png", dpi=180)
                    plt.close(figLd)
                    _loadings_png_da.seek(0)

                    # ---------- (C) Data for 2D/3D projections (use df_plsda) ----------
                    # Assume df_plsda has columns "PLSDA1", "PLSDA2", "PLSDA3", ..., and "Label"
                    proj2d_da_df = df_plsda[
                        [c for c in df_plsda.columns if c.startswith("PLSDA")][:2] + ["Label"]].copy()
                    # If 3D available:
                    _plsda_cols = [c for c in df_plsda.columns if c.startswith("PLSDA")]
                    if len(_plsda_cols) >= 3:
                        proj3d_da_df = df_plsda[_plsda_cols[:3] + ["Label"]].copy()
                    else:
                        proj3d_da_df = pd.DataFrame(columns=_plsda_cols[:2] + ["Z", "Label"])

                    # ---------- (D) Build Excel with all pages & editable charts ----------
                    buffer_plsda = io.BytesIO()
                    with pd.ExcelWriter(buffer_plsda, engine="xlsxwriter") as writer:
                        wb = writer.book

                        # --- PLS-DA Summary (if you built a summary table like plsda_calcv_df) ---
                        try:
                            plsda_calcv_df.to_excel(writer, sheet_name="PLS-DA Summary", index=False)
                            _autosize_columns(writer.sheets["PLS-DA Summary"], plsda_calcv_df)
                        except Exception:
                            pass

                        # --- VIP / Drivers / Regions (if available) ---
                        try:
                            vip_top_da.to_excel(writer, sheet_name="VIP (Top N)", index=False)
                            _autosize_columns(writer.sheets["VIP (Top N)"], vip_top_da)
                        except Exception:
                            pass

                        try:
                            _drivers_view_da = drivers_df[
                                ["Wavenumber (cm⁻¹)", "Loading", "|Loading|", "Band Annotation", "Interpretation"]]
                            _drivers_view_da.to_excel(writer, sheet_name="Most Influential", index=False)
                            _autosize_columns(writer.sheets["Most Influential"], _drivers_view_da)
                        except Exception:
                            pass

                        try:
                            contrib_df.to_excel(writer, sheet_name="Regions", index=False)
                            _autosize_columns(writer.sheets["Regions"], contrib_df)
                        except Exception:
                            pass

                        # --- Loadings (table + Excel-editable chart) ---
                        try:
                            def _num_first(val):
                                s = str(val).strip()
                                if "/" in s:
                                    s = s.split("/")[0].strip()
                                try:
                                    return float(s)
                                except Exception:
                                    return np.nan


                            load_tbl_da = pd.DataFrame({
                                "Wavenumber (cm⁻¹)": list(selected_columns),
                                "Numeric Wavenumber": [_num_first(v) for v in selected_columns],
                                "Index": np.arange(len(selected_columns)),
                                "Loading": pd.to_numeric(P_da[:, _da_load_idx], errors="coerce")
                            })

                            load_tbl_da.to_excel(writer, sheet_name="Loadings", index=False)
                            ws_load_da = writer.sheets["Loadings"]
                            _autosize_columns(ws_load_da, load_tbl_da)

                            chart_da = wb.add_chart({"type": "line"})
                            n_rows_da = len(load_tbl_da)
                            chart_da.add_series({
                                "name": f"{selected_da_loading}",
                                "categories": ["Loadings", 1, 1, n_rows_da, 1],  # Numeric Wavenumber
                                "values": ["Loadings", 1, 3, n_rows_da, 3],  # Loading
                                "line": {"width": 2.25},
                            })
                            chart_da.set_title({"name": f"Loadings — {selected_da_loading} ({_var_pct_da:.2f}% X-var)"})
                            chart_da.set_x_axis({"name": "Wavenumber (cm⁻¹)"})
                            chart_da.set_y_axis({"name": "Loading"})
                            chart_da.set_legend({"position": "bottom"})
                            ws_load_da.insert_chart(n_rows_da + 3, 0, chart_da)
                        except Exception:
                            pass

                        # --- 📝 Auto-summary ---
                        try:
                            auto_lines_da = []
                            auto_lines_da.extend(summary)  # list of bullet lines built earlier
                            auto_lines_da.append("Key drivers:")
                            auto_lines_da.extend(bullets)
                            auto_df_da = pd.DataFrame({"Summary": auto_lines_da})
                            auto_df_da.to_excel(writer, sheet_name="📝 Auto-summary", index=False)
                            _autosize_columns(writer.sheets["📝 Auto-summary"], auto_df_da)
                        except Exception:
                            pass

                        # --- 📉 PLS-DA 2D Projection (editable scatter) ---
                        try:
                            sheet2d_da = "📉 PLS-DA 2D Projection"
                            proj2d_da_df.to_excel(writer, sheet_name=sheet2d_da, index=False)
                            ws2d_da = writer.sheets[sheet2d_da]
                            _autosize_columns(ws2d_da, proj2d_da_df)

                            # guess first two PLSDA axes
                            da_x, da_y = [c for c in proj2d_da_df.columns if c.startswith("PLSDA")][:2]

                            chart2d_da = wb.add_chart({"type": "scatter"})
                            cur_row = len(proj2d_da_df) + 3
                            for lab in proj2d_da_df["Label"].unique():
                                sub = proj2d_da_df[proj2d_da_df["Label"] == lab][[da_x, da_y]].reset_index(drop=True)
                                if sub.empty:
                                    continue
                                ws2d_da.write(cur_row, 0, f"{lab} — {da_x}")
                                ws2d_da.write(cur_row, 1, f"{lab} — {da_y}")
                                ws2d_da.write_column(cur_row + 1, 0, sub[da_x].tolist())
                                ws2d_da.write_column(cur_row + 1, 1, sub[da_y].tolist())

                                s_start, s_end = cur_row + 1, cur_row + len(sub)
                                chart2d_da.add_series({
                                    "name": str(lab),
                                    "categories": [sheet2d_da, s_start, 0, s_end, 0],
                                    "values": [sheet2d_da, s_start, 1, s_end, 1],
                                    "marker": {"type": "circle", "size": 5},
                                    "line": {"none": True},
                                })
                                cur_row = s_end + 2

                            chart2d_da.set_title({"name": f"PLS-DA 2D: {da_x} vs {da_y}"})
                            chart2d_da.set_x_axis({"name": da_x})
                            chart2d_da.set_y_axis({"name": da_y})
                            chart2d_da.set_legend({"position": "bottom"})
                            ws2d_da.insert_chart(1, 4, chart2d_da)
                        except Exception:
                            pass

                        # --- 🧭 PLS-DA 3D Visualization (PNG fallback for Excel) ---
                        try:
                            sheet3d_da = "🧭 PLS-DA 3D Visualization"
                            proj3d_da_df.to_excel(writer, sheet_name=sheet3d_da, index=False)
                            ws3d_da = writer.sheets[sheet3d_da]
                            _autosize_columns(ws3d_da, proj3d_da_df)

                            _plsda_cols = [c for c in proj3d_da_df.columns if c.startswith("PLSDA")]
                            if len(_plsda_cols) >= 3:
                                from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

                                fig3d_da = plt.figure(figsize=(6, 5))
                                ax3d_da = fig3d_da.add_subplot(111, projection="3d")
                                for lab in proj3d_da_df["Label"].unique():
                                    d = proj3d_da_df[proj3d_da_df["Label"] == lab]
                                    ax3d_da.scatter(d[_plsda_cols[0]], d[_plsda_cols[1]], d[_plsda_cols[2]],
                                                    label=str(lab), s=16)
                                ax3d_da.set_xlabel(_plsda_cols[0]);
                                ax3d_da.set_ylabel(_plsda_cols[1]);
                                ax3d_da.set_zlabel(_plsda_cols[2])
                                ax3d_da.set_title("PLS-DA 3D Visualization")
                                ax3d_da.legend(loc="best", fontsize=8)
                                _plsda3d_png = io.BytesIO()
                                fig3d_da.tight_layout()
                                fig3d_da.savefig(_plsda3d_png, format="png", dpi=180)
                                plt.close(fig3d_da)
                                _plsda3d_png.seek(0)
                                ws3d_da.insert_image(1, 5, "plsda3d.png", {"image_data": _plsda3d_png})
                        except Exception:
                            pass

                        # --- Model_Metrics (summary + all confusion matrices + heatmaps) ---
                        wsM_da = wb.add_worksheet("Model_Metrics")
                        headers_da = list(results_export_da.columns)
                        for c, h in enumerate(headers_da):
                            wsM_da.write(0, c, h)
                        for r, row in results_export_da.iterrows():
                            for c, h in enumerate(headers_da):
                                wsM_da.write(r + 1, c, row[h])
                        _autosize_columns(wsM_da, results_export_da)

                        start_row = len(results_export_da) + 3
                        for name in models_da.keys():
                            # counts
                            wsM_da.write(start_row, 0, f"{name} — Confusion Matrix (counts)")
                            cnt = confmats_counts_da[name]
                            for j, col in enumerate([""] + cnt.columns.tolist()):
                                wsM_da.write(start_row + 1, j, col)
                            for i, idx in enumerate(cnt.index.tolist()):
                                wsM_da.write(start_row + 2 + i, 0, idx)
                                for j, col in enumerate(cnt.columns.tolist()):
                                    wsM_da.write(start_row + 2 + i, 1 + j, int(cnt.loc[idx, col]))
                            _autosize_range(
                                wsM_da,
                                first_col=0,
                                last_col=len(cnt.columns),
                                rows=[{k: None for k in range(len(cnt.columns) + 1)} for _ in
                                      range(len(cnt.index) + 2)],
                                headers=[""] + cnt.columns.tolist()
                            )
                            start_row += len(cnt) + 3

                            # percentages + heatmap image
                            wsM_da.write(start_row, 0, f"{name} — Confusion Matrix (%)")
                            pct = confmats_pct_da[name]
                            for j, col in enumerate([""] + pct.columns.tolist()):
                                wsM_da.write(start_row + 1, j, col)
                            for i, idx in enumerate(pct.index.tolist()):
                                wsM_da.write(start_row + 2 + i, 0, idx)
                                for j, col in enumerate(pct.columns.tolist()):
                                    wsM_da.write_number(start_row + 2 + i, 1 + j, float(pct.loc[idx, col]))
                            img_col = pct.shape[1] + 3
                            wsM_da.insert_image(start_row + 1, img_col, f"{name}_cm.png",
                                                {"image_data": confmat_heatmap_png_da[name], "x_scale": 1.0,
                                                 "y_scale": 1.0})
                            start_row += len(pct) + 10

                    buffer_plsda.seek(0)

                    # ================= Download Complete PLS-DA Analysis =================
                    colA_da, colB_da = st.columns([4, 1])
                    with colA_da:
                        st.download_button(
                            "📥 Download PLS-DA Full Workbook (editable loadings chart, 2D chart, 3D, summaries, all confusion matrices)",
                            data=buffer_plsda.getvalue(),
                            file_name="plsda_full_export.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                    with colB_da:
                        st.empty()
                    # =====================================================================================================================================

            ###############################################################################################################

            # ============================
            # 📊 PCA • PLS • PLSDA (Side-by-Side) with Class Ellipses
            # ============================
            st.subheader("📊 PCA • PLS • PLSDA")

            with st.expander("📊 PCA • PLS • PLSDA — side-by-side comparison", expanded=True):

                if show_plsda and show_pls and show_pca:

                    from plotly.subplots import make_subplots
                    from sklearn.preprocessing import StandardScaler, LabelEncoder
                    from sklearn.decomposition import PCA
                    from sklearn.cross_decomposition import PLSRegression

                    # ---------- helpers ----------
                    def _class_ellipse_xy(xy, nsig=2.0, npts=200):
                        """Return x,y for an nsig covariance ellipse of 2D points."""
                        xy = np.asarray(xy)
                        if xy.shape[0] < 3:
                            return None, None
                        mu = xy.mean(axis=0)
                        S = np.cov(xy.T)
                        vals, vecs = np.linalg.eigh(S)
                        order = np.argsort(vals)[::-1]
                        vals, vecs = vals[order], vecs[:, order]
                        radii = nsig * np.sqrt(np.maximum(vals, 1e-12))
                        ang = np.linspace(0, 2 * np.pi, npts)
                        circ = np.stack([np.cos(ang) * radii[0], np.sin(ang) * radii[1]], axis=0)
                        ell = (vecs @ circ).T + mu
                        return ell[:, 0], ell[:, 1]

                    def _add_class_ellipses(fig, df, xcol, ycol, label_col, subplot_col, nsig=2.0):
                        for lab in df[label_col].unique():
                            d = df[df[label_col] == lab][[xcol, ycol]].values
                            ex, ey = _class_ellipse_xy(d, nsig=nsig)
                            if ex is None:
                                continue
                            fig.add_trace(
                                go.Scatter(x=ex, y=ey, mode="lines", name=f"{lab} ellipse",
                                           line=dict(width=1), showlegend=(subplot_col == 1)),
                                row=1, col=subplot_col
                            )

                    # ---------- prep ----------
                    scaler = StandardScaler()
                    Xz = scaler.fit_transform(X)

                    # ---------- PCA (2D) ----------
                    pca2 = PCA(n_components=2).fit(Xz)
                    T_pca = pca2.transform(Xz)
                    df_pca = pd.DataFrame(T_pca, columns=["PC1", "PC2"])
                    df_pca["Label"] = y.values

                    # ---------- PLS (2D, regression to encoded labels) ----------
                    y_enc = LabelEncoder().fit_transform(y)
                    pls2 = PLSRegression(n_components=2).fit(Xz, y_enc)
                    T_pls = pls2.x_scores_[:, :2]
                    df_pls = pd.DataFrame(T_pls, columns=["PLS1", "PLS2"])
                    df_pls["Label"] = y.values

                    # ---------- PLS-DA (multiresponse PLS on one-hot Y) ----------
                    y_codes = LabelEncoder().fit_transform(y)
                    y_onehot = pd.get_dummies(y_codes)  # (n_samples, n_classes)
                    plsda2 = PLSRegression(n_components=2).fit(Xz, y_onehot.values)
                    T_da = plsda2.x_scores_[:, :2]
                    df_da = pd.DataFrame(T_da, columns=["PLSDA1", "PLSDA2"])
                    df_da["Label"] = y.values

                    # ---------- figure ----------
                    fig = make_subplots(
                        rows=1, cols=3,
                        subplot_titles=("PCA", "PLS", "PLS-DA"),
                        horizontal_spacing=0.07
                    )

                    # PCA panel
                    for lab in df_pca["Label"].unique():
                        d = df_pca[df_pca["Label"] == lab]
                        fig.add_trace(go.Scatter(x=d["PC1"], y=d["PC2"], mode="markers",
                                                 name=str(lab), showlegend=True),
                                      row=1, col=1)
                    _add_class_ellipses(fig, df_pca, "PC1", "PC2", "Label", subplot_col=1)

                    # PLS panel
                    for lab in df_pls["Label"].unique():
                        d = df_pls[df_pls["Label"] == lab]
                        fig.add_trace(go.Scatter(x=d["PLS1"], y=d["PLS2"], mode="markers",
                                                 name=str(lab), showlegend=False),
                                      row=1, col=2)
                    _add_class_ellipses(fig, df_pls, "PLS1", "PLS2", "Label", subplot_col=2)

                    # PLS-DA panel
                    for lab in df_da["Label"].unique():
                        d = df_da[df_da["Label"] == lab]
                        fig.add_trace(go.Scatter(x=d["PLSDA1"], y=d["PLSDA2"], mode="markers",
                                                 name=str(lab), showlegend=False),
                                      row=1, col=3)
                    _add_class_ellipses(fig, df_da, "PLSDA1", "PLSDA2", "Label", subplot_col=3)

                    # axes + layout
                    fig.update_xaxes(title_text="PC1", row=1, col=1)
                    fig.update_yaxes(title_text="PC2", row=1, col=1)
                    fig.update_xaxes(title_text="PLS1", row=1, col=2)
                    fig.update_yaxes(title_text="PLS2", row=1, col=2)
                    fig.update_xaxes(title_text="PLSDA1", row=1, col=3)
                    fig.update_yaxes(title_text="PLSDA2", row=1, col=3)

                    fig.update_layout(
                        height=420,
                        margin=dict(l=30, r=30, t=60, b=10),
                        legend_title_text="Class"
                    )
                    st.plotly_chart(fig, use_container_width=True)

            #####################################################################################################################
            #####################################################################################################################

            ##############    📉 Standard Deviation of FTIR Spectra
            if show_stddev:

                st.subheader("📉 Standard Deviation of FTIR Spectra")
                with st.expander(
                        "Standard Deviation of FTIR Spectra reveals variability in absorbance at each wavenumber, helping identify spectral regions with significant differences across sample classes.",
                        expanded=True):
                    # Compute standard deviation for each class dynamically
                    std_df = X.groupby(y).std()

                    fig_std = go.Figure()

                    # Add a line for each class
                    for label in std_df.index:
                        fig_std.add_trace(go.Scatter(
                            x=selected_columns,
                            y=std_df.loc[label],
                            mode='lines',
                            name=f'{label} Std Dev'
                        ))

                    # Define annotation wavenumbers and corresponding labels
                    annotations = {
                        1740: "C=O Stretching (Ester)",
                        1650: "Amide I (C=O Stretch)",
                        1630: "Water O–H Bending",
                        1550: "Amide II (C–N stretch + N–H bend)",
                        1338: "CH₂ Side Chain Vibration",
                        1200: "C–O Stretching",
                        1115: "HPO₄²⁻ Stretching",
                        1060: "Sugar Ring C–O Stretch",
                        1030: "PO₄³⁻ Stretching",
                        875: "CO₃²⁻ Bending",
                        856: "C–S Bending",
                        850: "CO₃²⁻ Bending",
                        7000: "O–H Stretching",
                        6688: "N–H Stretching",
                        5800: "CH₂ Stretching",
                        5200: "O–H Stretching and Bending",
                        4890: "N–H Bending",
                        4610: "C–H Stretching & Deformation",
                        4310: "Sugar Ring Vibrations"
                    }

                    # Add vertical lines and annotations if in selected range
                    for wavenumber, label in annotations.items():
                        if selected_range[0] <= wavenumber <= selected_range[1]:
                            fig_std.add_vline(x=wavenumber, line=dict(color="gray", width=1, dash="dash"))
                            # Compute max Y value from the std_df for proper placement of annotation
                            max_y = std_df.max().max()
                            fig_std.add_annotation(x=wavenumber, y=max_y, text=label,
                                                   showarrow=True, arrowhead=1, yshift=10,
                                                   font=dict(size=9))

                    fig_std.update_layout(title="Standard Deviation of FTIR Spectra with Band Annotations",
                                          xaxis_title="Wavenumber (cm⁻¹)",
                                          yaxis_title="Standard Deviation",
                                          legend_title="Group",
                                          height=400,
                                          margin=dict(l=40, r=40, t=40, b=40))

                    st.plotly_chart(fig_std, use_container_width=True)

                    st.markdown(
                        "This plot compares the standard deviation of absorbance values across the FTIR spectrum between groups. Higher deviation may indicate greater molecular variability or structural heterogeneity in those regions.")

            #####################################################################################################################
            ##############    📈 Mean of FTIR Spectra

            if show_mean:

                st.subheader("📈 Mean of FTIR Spectra")
                with st.expander(
                        "Average FTIR Spectra with Band Annotations highlights the typical absorbance patterns for each class, with key vibrational bands labeled to aid in molecular interpretation.",
                        expanded=True):
                    # Compute mean for each class dynamically
                    mean_df = X.groupby(y).mean()

                    fig_mean = go.Figure()

                    # Add a line for each class
                    for label in mean_df.index:
                        fig_mean.add_trace(go.Scatter(
                            x=selected_columns,
                            y=mean_df.loc[label],
                            mode='lines',
                            name=f'{label} Mean'
                        ))

                    # Define annotation wavenumbers and corresponding labels
                    annotations = {
                        1740: "C=O Stretching (Ester)",
                        1650: "Amide I (C=O Stretch)",
                        1630: "Water O–H Bending",
                        1550: "Amide II (C–N stretch + N–H bend)",
                        1338: "CH₂ Side Chain Vibration",
                        1200: "C–O Stretching",
                        1115: "HPO₄²⁻ Stretching",
                        1060: "Sugar Ring C–O Stretch",
                        1030: "PO₄³⁻ Stretching",
                        875: "CO₃²⁻ Bending",
                        856: "C–S Bending",
                        850: "CO₃²⁻ Bending",
                        7000: "O–H Stretching",
                        6688: "N–H Stretching",
                        5800: "CH₂ Stretching",
                        5200: "O–H Stretching and Bending",
                        4890: "N–H Bending",
                        4610: "C–H Stretching & Deformation",
                        4310: "Sugar Ring Vibrations"
                    }

                    # Add vertical lines and annotations if in selected range
                    for wavenumber, label in annotations.items():
                        if selected_range[0] <= wavenumber <= selected_range[1]:
                            fig_mean.add_vline(x=wavenumber, line=dict(color="gray", width=1, dash="dash"))
                            # Compute max Y value from the mean_df for proper placement of annotation
                            max_y = mean_df.max().max()
                            fig_mean.add_annotation(x=wavenumber, y=max_y, text=label,
                                                    showarrow=True, arrowhead=1, yshift=10,
                                                    font=dict(size=9))

                    fig_mean.update_layout(title="Mean FTIR Spectra with Band Annotations",
                                           xaxis_title="Wavenumber (cm⁻¹)",
                                           yaxis_title="Mean Absorbance",
                                           legend_title="Group",
                                           height=400,
                                           margin=dict(l=40, r=40, t=40, b=40))

                    st.plotly_chart(fig_mean, use_container_width=True)

                    st.markdown(
                        "This plot displays the average absorbance spectrum for each class, allowing comparison of characteristic molecular features and structural composition across samples.")

            #####################################################################################################################
            ##############    📊 Mean+StdDev of FTIR Spectra

            if show_mean_stddev:

                st.subheader("📊 Mean + Standard Deviation of FTIR Spectra")
                with st.expander(
                        "Mean+Standard Deviation highlights areas of high absorbance and variability, indicating dynamic molecular activity across classes.",
                        expanded=True):
                    mean_df = X.groupby(y).mean()
                    std_df = X.groupby(y).std()

                    fig_mean_std = go.Figure()

                    for label in mean_df.index:
                        mean_plus_std = mean_df.loc[label] + std_df.loc[label]
                        fig_mean_std.add_trace(go.Scatter(
                            x=selected_columns,
                            y=mean_plus_std,
                            mode='lines',
                            name=f'{label} Mean + StdDev'
                        ))

                    annotations = {
                        1740: "C=O Stretching (Ester)",
                        1650: "Amide I (C=O Stretch)",
                        1630: "Water O–H Bending",
                        1550: "Amide II (C–N stretch + N–H bend)",
                        1338: "CH₂ Side Chain Vibration",
                        1200: "C–O Stretching",
                        1115: "HPO₄²⁻ Stretching",
                        1060: "Sugar Ring C–O Stretch",
                        1030: "PO₄³⁻ Stretching",
                        875: "CO₃²⁻ Bending",
                        856: "C–S Bending",
                        850: "CO₃²⁻ Bending",
                        7000: "O–H Stretching",
                        6688: "N–H Stretching",
                        5800: "CH₂ Stretching",
                        5200: "O–H Stretching and Bending",
                        4890: "N–H Bending",
                        4610: "C–H Stretching & Deformation",
                        4310: "Sugar Ring Vibrations"
                    }

                    for wavenumber, label_text in annotations.items():
                        if selected_range[0] <= wavenumber <= selected_range[1]:
                            fig_mean_std.add_vline(x=wavenumber, line=dict(color="gray", width=1, dash="dash"))
                            max_y = (mean_df + std_df).max().max()
                            fig_mean_std.add_annotation(x=wavenumber, y=max_y, text=label_text,
                                                        showarrow=True, arrowhead=1, yshift=10,
                                                        font=dict(size=9))

                    fig_mean_std.update_layout(title="Mean + StdDev FTIR Spectra with Band Annotations",
                                               xaxis_title="Wavenumber (cm⁻¹)",
                                               yaxis_title="Mean + StdDev Absorbance",
                                               legend_title="Group",
                                               height=400,
                                               margin=dict(l=40, r=40, t=40, b=40))

                    st.plotly_chart(fig_mean_std, use_container_width=True)

                    st.markdown(
                        "This plot displays the upper envelope (Mean + StdDev) of absorbance values across each FTIR spectrum class, highlighting regions with high activity and variability.")

            #####################################################################################################################
            ##############    📉 Minimum of FTIR Spectra

            if show_minimum:

                st.subheader("📉 Minimum of FTIR Spectra")
                with st.expander(
                        "Minimum absorbance reveals the lowest signal response per wavenumber, indicating consistent absence or weak presence of functional groups across samples.",
                        expanded=True):
                    min_df = X.groupby(y).min()

                    fig_min = go.Figure()

                    for label in min_df.index:
                        fig_min.add_trace(go.Scatter(
                            x=selected_columns,
                            y=min_df.loc[label],
                            mode='lines',
                            name=f'{label} Minimum'
                        ))

                    for wavenumber, label_text in annotations.items():
                        if selected_range[0] <= wavenumber <= selected_range[1]:
                            fig_min.add_vline(x=wavenumber, line=dict(color="gray", width=1, dash="dash"))
                            max_y = min_df.max().max()
                            fig_min.add_annotation(x=wavenumber, y=max_y, text=label_text,
                                                   showarrow=True, arrowhead=1, yshift=10,
                                                   font=dict(size=9))

                    fig_min.update_layout(title="Minimum FTIR Spectra with Band Annotations",
                                          xaxis_title="Wavenumber (cm⁻¹)",
                                          yaxis_title="Minimum Absorbance",
                                          legend_title="Group",
                                          height=400,
                                          margin=dict(l=40, r=40, t=40, b=40))

                    st.plotly_chart(fig_min, use_container_width=True)

                    st.markdown(
                        "This plot shows the lowest absorbance values per wavenumber for each group, useful for identifying regions with consistent low or absent molecular features.")

            #####################################################################################################################
            ##############    📈 Maximum of FTIR Spectra

            if show_maximum:

                st.subheader("📈 Maximum of FTIR Spectra")
                with st.expander(
                        "Maximum absorbance highlights spectral regions with the highest molecular signal response, often corresponding to dominant functional groups or sample peaks.",
                        expanded=True):
                    max_df = X.groupby(y).max()

                    fig_max = go.Figure()

                    for label in max_df.index:
                        fig_max.add_trace(go.Scatter(
                            x=selected_columns,
                            y=max_df.loc[label],
                            mode='lines',
                            name=f'{label} Maximum'
                        ))

                    for wavenumber, label_text in annotations.items():
                        if selected_range[0] <= wavenumber <= selected_range[1]:
                            fig_max.add_vline(x=wavenumber, line=dict(color="gray", width=1, dash="dash"))
                            max_y = max_df.max().max()
                            fig_max.add_annotation(x=wavenumber, y=max_y, text=label_text,
                                                   showarrow=True, arrowhead=1, yshift=10,
                                                   font=dict(size=9))

                    fig_max.update_layout(title="Maximum FTIR Spectra with Band Annotations",
                                          xaxis_title="Wavenumber (cm⁻¹)",
                                          yaxis_title="Maximum Absorbance",
                                          legend_title="Group",
                                          height=400,
                                          margin=dict(l=40, r=40, t=40, b=40))

                    st.plotly_chart(fig_max, use_container_width=True)

                    st.markdown(
                        "This plot shows the maximum absorbance values across the FTIR spectrum for each group, identifying dominant peaks and highly active regions.")

            #####################################################################################################################

            # ================================================
            # 📉 Molecular Interpretation from Spectral Bands
            # ================================================

            ##############    🧪 Molecular Interpretation from Spectral Bands

            st.subheader("🧪 Molecular Interpretation from Spectral Bands")
            with st.expander(
                    "Molecular Interpretation from Spectral Bands links specific wavenumber regions to functional group vibrations, enabling identification of molecular structures and chemical compositions in the samples.",
                    expanded=True):
                vib_data = [
                    ["8500", "O–H Stretching and Bending", "Water", "Hydration marker in tissues"],
                    ["7000", "O–H Stretching", "Water", "Water content level (broad absorption)"],
                    ["6688", "N–H Stretching", "Protein / Collagen", "Backbone amide stretching (protein presence)"],
                    ["5800", "CH₂ Stretching", "Lipid", "Lipid content (fatty acid chains)"],
                    ["5200", "O–H Stretching and Bending", "Water", "Hydration-sensitive feature"],
                    ["4890", "N–H Bending", "Protein / Collagen", "Protein conformation changes"],
                    ["4610", "C–H Stretching & Deformation", "Protein / Collagen", "Side chain deformations"],
                    ["4310", "Sugar Ring Vibrations", "Proteoglycan", "ECM components like GAGs"],

                    ["3600–3200", "O–H Stretching", "Water, Hydroxyl", "Broad peak, indicates hydration or hydroxylation"],
                    ["3500–3300", "N–H Stretching", "Proteins", "Backbone amide stretching in proteins"],
                    ["3000–2800", "C–H Stretching", "Lipids, CH₂", "Fatty acids in lipids (symmetric and asymmetric)"],
                    ["1750–1650", "C=O Stretching", "Proteins, lipids", "Amide I band; carbonyls in esters/lipids"],
                    ["1550", "N–H Bending / C–N Stretch", "Amide II (Proteins)", "Collagen structure marker"],
                    ["1338", "CH₂ Side Chain Bending", "Collagen", "Indicator of collagen integrity"],
                    ["1100–900", "PO₄³⁻ Stretching", "Bone Mineral", "Hydroxyapatite phosphate groups"],
                    ["890–850", "CO₃²⁻ Bending", "Carbonate", "Carbonate substitution in apatite"],

                    ["1740", "C=O Stretching (Ester)", "Lipids, fatty acid esters",
                     "Indicates lipid presence in bone tissue"],
                    ["1650", "Amide I (C=O Stretch)", "Proteins (peptide bonds)",
                     "Backbone stretch in proteins (collagen-rich regions)"],
                    ["1630", "Water O–H Bending", "Water content", "Tissue hydration; can affect FTIR baselines"],
                    ["1550", "Amide II (C–N stretch + N–H bend)", "Proteins (peptide bonds)",
                     "Protein secondary structure (collagen fibrils)"],
                    ["1338", "CH₂ Side Chain Vibration", "Collagen", "Structural collagen-related bending mode"],
                    ["1200–1000", "C–O Stretching", "Alcohols, esters, ethers",
                     "Carbohydrates and organic matrix components"],
                    ["1115", "HPO₄²⁻ Stretching", "Bone mineral", "Immature mineral phase (hydrogen phosphate)"],
                    ["1060", "Sugar Ring C–O Stretch", "Carbohydrates", "From glycoproteins/proteoglycans"],
                    ["1030", "PO₄³⁻ Stretching", "Bone mineral (hydroxyapatite)", "Mature mineral content in bone"],
                    ["875", "CO₃²⁻ Bending", "Carbonates", "Substituted carbonate in apatite; affects stiffness"],
                    ["856", "C–S Bending", "Sulfated proteoglycans",
                     "ECM components important in growth plate / cartilage"],
                    ["800–900", "Aromatic C–H Bending (out-of-plane)", "Aromatic rings (fingerprint region)",
                     "Often present in structural proteins and metabolites"]
                ]
                vib_df = pd.DataFrame(vib_data,
                                      columns=["Wavenumber (cm⁻¹)", "Assignment / Mode", "Functional Group / Bond",
                                               "Tissue/Biochemical Relevance"])
                st.dataframe(vib_df.style.set_properties(**{
                    'background-color': '#f8f9fa',
                    'color': 'black',
                    'border-color': 'lightgray'
                }), use_container_width=True)

        ###################################### [FTIR spectral data Information ] ######################################

            st.markdown("""
            The FTIR spectral data from bone typically contains vibrational features associated with both organic and inorganic components:
        
            #### 🔬 Organic Matrix:
            - **1650 cm⁻¹ (Amide I)**: C=O stretching of proteins (mainly collagen).
            - **1550 cm⁻¹ (Amide II)**: N–H bending and C–N stretching.
            - **1338 cm⁻¹**: CH₂ side chain vibrations in collagen.
            - **1200–1000 cm⁻¹**: C–O stretching and other sugar ring vibrations.
        
            #### 🧱 Mineral Content:
            - **1115 cm⁻¹**: HPO₄²⁻ bending (mineral phase).
            - **1030 cm⁻¹**: PO₄³⁻ symmetric stretching – primary phosphate marker.
            - **875 cm⁻¹**: CO₃²⁻ bending – carbonate substitution in apatite.
            - **856 cm⁻¹**: C–S or proteoglycan components.
        
            #### 💧 Water and Other Features:
            - **1630 cm⁻¹**: Water O–H bending.
            - **7000–3600 cm⁻¹**: O–H stretching and bending of bound/free water.
            - **5800 cm⁻¹**: CH₂ stretching (lipids or side chains).
            - **5200–4300 cm⁻¹**: Combination bands and sugar ring-related vibrations.
        
            These bands help identify changes in bone tissue due to conditions like osteoporosis, collagen mutation (OIM), or hydration level.
            """)

        #####################################################################################################################

############################################# [ footer ] #############################################################################

st.markdown(
    """
    <div style='
        position: fixed; left: 0; bottom: 0; width: 100%;
        text-align: center; font-size: 0.9em; color: gray;
        background: #ffffffdd; padding: 8px 0;'>
        • Copyright © 2025 Komal Wavhal - Stevens Institute of Technology 🧬 AI-Powered Vibrational Spectroscopy for Bone Health.
    </div>
    """,
    unsafe_allow_html=True
)

##########################################################################################################################


from datetime import datetime

# 🧪 NEW SECTION — Professor's Sample-ID Grouped Leave-One-Group-Out Validation
# ════════════════════════════════════════════════════════════════════════════
# Everything above this line is your ORIGINAL app_k.py, completely unchanged.
# This section was appended at the end of the file on top of it, per the
# professor's reviewer-driven request (see email): re-run SVM / KNN /
# Logistic Regression / Random Forest / XGBoost using an Sample-ID-grouped
# validation strategy so no single animal's spectra appear in both the
# training and test sets, and so every Sample ID group contributes an equal amount
# of data (250 spectra per Sample ID, fixed random seed, Leave-One-Group-Out CV
# with Sample ID as the grouping variable).
#
# Model hyperparameters are intentionally IDENTICAL to the rest of this app
# (SVC(), KNeighborsClassifier(n_neighbors=3), LogisticRegression(max_iter=1000),
# RandomForestClassifier(n_estimators=100), XGBClassifier(eval_metric="logloss"))
# — only the train/test splitting strategy changes here, so the new accuracy
# numbers are a fair, apples-to-apples comparison to your existing results.
#
# This section is opt-in via a sidebar checkbox so it never changes what you
# see by default; toggle it on whenever you want to run this validation.
# ════════════════════════════════════════════════════════════════════════════

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.base import clone

st.sidebar.markdown("---")
show_professor_panel = st.sidebar.checkbox(
    "🧪 Professor's Sample-ID Grouped Validation (NEW)",
    value=False,
    help="Reviewer-requested validation: 250 spectra per Sample ID + Leave-One-Group-Out CV grouped by Sample ID.",
)

if show_professor_panel:
    st.markdown("---")
    st.header("🧪 Sample-ID Grouped Leave-One-Group-Out Validation")
    st.caption("Testing on professor's request")

    with st.expander("Why this section exists / methodology", expanded=False):
        st.markdown(
            """
- **Problem:** a plain random train/test split can put spectra from the same Sample ID group in both
  the training and test sets, and Sample ID groups can contribute very different amounts of data.
- **Fix:** sample a fixed number of spectra per Sample ID (default **250**, fixed random seed),
  then use **Leave-One-Group-Out cross-validation** with Sample ID as the group — each fold
  trains on all remaining Sample ID groups and tests on the 1 held-out group.
- **Models/hyperparameters are unchanged** from the rest of this app — only the splitting
  strategy changes here.
- **Scaling:** when a model requires scaling, the scaler is fitted only on the training Sample ID groups
  inside each LOGO fold and then applied to the held-out Sample ID group.
- **No Sample IDs are hard-coded.** The application reads all Sample IDs and WT/OIM labels directly from the uploaded Excel file.
            """
        )

    # No hard-coded Sample-ID lists are used by the modeling or report logic.
    # Group membership is read from the uploaded Excel file (Sample ID + Type columns).
    _MODEL_ORDER = ["SVM", "KNN", "Logistic Regression", "Random Forest", "XGBoost"]
    _REGIONS = {"Fingerprint": (800, 1800), "Collagen": (1300, 1700), "Mineral": (800, 1200)}
    _REGION_LABELS = {
        "Fingerprint": "Fingerprint region (800-1800 cm-1)",
        "Collagen": "Collagen/Amide region (1300-1700 cm-1)",
        "Mineral": "Mineral region (800-1200 cm-1)",
    }
    _SHEET_KIND = {"Raw": "spectral", "2nd derivative": "spectral", "Quant peak ratios": "ratio"}
    _TEMPLATE_COLUMNS = {
        ("Raw", "Fingerprint"): "B", ("Raw", "Collagen"): "C", ("Raw", "Mineral"): "D",
        ("2nd derivative", "Fingerprint"): "E", ("2nd derivative", "Collagen"): "F", ("2nd derivative", "Mineral"): "G",
        ("Quant peak ratios", "ratio"): "H",
    }
    _MODEL_ROW = {"SVM": 6, "KNN": 7, "Logistic Regression": 8, "Random Forest": 9, "XGBoost": 10}

    def _pv_find_group_label_columns(df):
        norm = {str(c).strip().lower(): c for c in df.columns}
        group_col = None
        for cand in ["sample id", "animal id", "mouse id", "id"]:
            if cand in norm:
                group_col = norm[cand]
                break
        label_col = None
        for cand in ["type", "class", "label", "group", "genotype"]:
            if cand in norm and norm[cand] != group_col:
                label_col = norm[cand]
                break
        if group_col is None or label_col is None:
            raise ValueError(
                "Could not find both a 'Sample ID' (animal) column and a 'Type' (WT/OIM) "
                f"column in this sheet. Columns found: {list(df.columns)[:12]}"
            )
        return group_col, label_col

    def _pv_wavenumber(col):
        s = str(col).strip()
        if "/" in s:
            return None
        try:
            return float(s)
        except ValueError:
            return None

    def _pv_region_columns(df, group_col, label_col, lo, hi):
        out = []
        for c in df.columns:
            if c in (group_col, label_col):
                continue
            wn = _pv_wavenumber(c)
            if wn is not None and lo <= wn <= hi:
                out.append(c)
        return out

    def _pv_ratio_columns(df, group_col, label_col):
        return [c for c in df.columns if c not in (group_col, label_col) and "/" in str(c)]

    def _pv_balance_per_animal(df, group_col, n_per_animal, seed):
        pieces, shortfalls = [], []
        for animal_id, g in df.groupby(group_col, sort=True):
            take = min(n_per_animal, len(g))
            if take < n_per_animal:
                shortfalls.append((animal_id, len(g)))
            pieces.append(g.sample(n=take, random_state=seed))
        balanced = pd.concat(pieces, ignore_index=True)
        return balanced, shortfalls

    def _pv_build_models(seed):
        # NOTE ON THE SPEED SETTINGS BELOW: n_jobs and cache_size are performance-only knobs.
        # They change how the work is spread across CPU cores and how much kernel cache SVM keeps
        # in RAM - they do NOT change any prediction, so every accuracy/precision/recall/F1 value
        # is bit-for-bit identical to a single-core run with the same seed. Hyper-parameters that
        # WOULD change results (n_estimators, max_depth, n_neighbors, C, gamma) are untouched.
        models = {
            # cache_size=1000 MB instead of the 200 MB default: SVM re-computes kernel values
            # constantly on 2,750 x 501 training folds, and a bigger cache avoids most of that.
            "SVM": SVC(cache_size=1000),
            # n_jobs=-1: distance computations for all 250 held-out spectra run across every core.
            "KNN": KNeighborsClassifier(n_neighbors=3, n_jobs=-1),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=seed),
            # n_jobs=-1: the 100 trees are grown in parallel. With random_state fixed the forest
            # is identical to the single-core version - only the wall-clock time changes.
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
        }
        if HAS_XGB:
            # n_jobs=-1 parallelises boosting across cores. tree_method is deliberately left at its
            # default, because changing it WOULD alter the split-finding and therefore the results.
            models["XGBoost"] = XGBClassifier(eval_metric="logloss", random_state=seed, n_jobs=-1)
        return models

    def _pv_positive_index(classes):
        # Index of the OIM (disease) class, used as the positive class for precision/recall/F1.
        try:
            return list(classes).index("OIM")  # LabelEncoder sorts alphabetically -> OIM is 0
        except Exception:
            return 1  # Fallback for any other labelling scheme

    def _pv_run_logo_cv(X, y, groups, model, classes=None):
        logo = LeaveOneGroupOut()
        fold_records = []
        all_true, all_pred, all_score = [], [], []
        # Resolve the positive class UP FRONT. LabelEncoder sorts alphabetically, so with labels
        # {OIM, WT} the disease class OIM is index 0 - NOT index 1. Reading predict_proba column 1
        # would therefore score the wrong class and invert the ROC-AUC (0.28 instead of 0.72).
        pos = _pv_positive_index(classes) if classes is not None else 1
        # Every model is standardised per fold, with the scaler fitted on the 11 TRAINING animals
        # only and applied to the held-out Sample ID group. This is required for Logistic Regression (its
        # solver will not converge on hundreds of unscaled wavenumber columns) and it materially
        # helps the distance/kernel models SVM (RBF) and KNN, which are penalised by unscaled
        # features. Fitting the scaler inside the fold is what keeps it leakage-free: no
        # information from the held-out Sample ID group ever reaches the training statistics.
        for tr, te in logo.split(X, y, groups):
            m = clone(model)
            X_tr, X_te = X.iloc[tr], X.iloc[te]
            _pv_scaler = StandardScaler()
            X_tr = pd.DataFrame(_pv_scaler.fit_transform(X_tr), index=X_tr.index, columns=X_tr.columns)
            X_te = pd.DataFrame(_pv_scaler.transform(X_te), index=X_te.index, columns=X_te.columns)
            m.fit(X_tr, y[tr])
            pred = m.predict(X_te)
            # Continuous score for the positive class, needed for the ROC-AUC column.
            try:
                if hasattr(m, "predict_proba"):
                    # Column `pos` is the probability of the positive (OIM) class.
                    all_score.extend(m.predict_proba(X_te)[:, pos].tolist())
                elif hasattr(m, "decision_function"):
                    # decision_function is signed towards class index 1, so flip it when the
                    # positive class is index 0.
                    _d = np.asarray(m.decision_function(X_te)).ravel()
                    all_score.extend((_d if pos == 1 else -_d).tolist())
                else:
                    all_score.extend([np.nan] * len(te))
            except Exception:
                all_score.extend([np.nan] * len(te))
            held_out = groups.iloc[te].iloc[0]
            fold_records.append({
                "Held-out Sample ID": held_out,
                "Test spectra": len(te),
                "Fold accuracy": accuracy_score(y[te], pred),
            })
            all_true.extend(y[te].tolist())
            all_pred.extend(pred.tolist())

        # Supplementary metrics the professor asked for on 9 Jan 2026: precision, recall and F1
        # for every model, computed on the pooled out-of-fold predictions (positive class = OIM).
        _prf = dict(pos_label=pos, average="binary", zero_division=0)
        try:
            _auc = roc_auc_score(np.asarray(all_true) == pos, np.asarray(all_score, dtype=float))
        except Exception:
            _auc = float("nan")  # Model produced no usable score, or only one class was present
        return {
            "overall_accuracy": accuracy_score(all_true, all_pred),
            "precision": precision_score(all_true, all_pred, **_prf),
            "recall": recall_score(all_true, all_pred, **_prf),
            "f1": f1_score(all_true, all_pred, **_prf),
            "roc_auc": _auc,
            "positive_class": (list(classes)[pos] if classes is not None else str(pos)),
            "fold_table": pd.DataFrame(fold_records),
            "confusion_matrix": confusion_matrix(all_true, all_pred),
            "n_features": X.shape[1],
        }

    def _pv_evaluate_sheet(df, sheet_name, models_to_run, n_per_animal, seed, progress_cb=None):
        group_col, label_col = _pv_find_group_label_columns(df)
        for c in df.columns:
            if c not in (group_col, label_col):
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=[c for c in df.columns if c not in (group_col, label_col)], how="all")

        balanced, shortfalls = _pv_balance_per_animal(df, group_col, n_per_animal, seed)

        le_pv = LabelEncoder()
        y_full = le_pv.fit_transform(balanced[label_col].astype(str))
        groups_full = balanced[group_col].astype(str)

        kind = _SHEET_KIND[sheet_name]
        region_keys = list(_REGIONS.keys()) if kind == "spectral" else ["ratio"]

        results = {}
        n_steps = max(1, len(region_keys) * len(models_to_run))
        step = 0
        models = _pv_build_models(seed)
        for region_key in region_keys:
            if kind == "spectral":
                lo, hi = _REGIONS[region_key]
                feat_cols = _pv_region_columns(balanced, group_col, label_col, lo, hi)
            else:
                feat_cols = _pv_ratio_columns(balanced, group_col, label_col)

            X = balanced[feat_cols].astype(float).fillna(0.0)

            results[region_key] = {}
            for model_name in models_to_run:
                res = _pv_run_logo_cv(X, y_full, groups_full, models[model_name],
                                      classes=list(le_pv.classes_))  # positive class = OIM
                results[region_key][model_name] = res
                step += 1
                if progress_cb:
                    progress_cb(step / n_steps, f"{sheet_name} — {region_key} — {model_name}")

        counts = balanced.groupby(group_col).size().rename("Spectra used").to_frame().reset_index()
        return results, counts, shortfalls, list(le_pv.classes_)

    def _pv_build_accuracy_table_excel(all_results, run_timestamp=None):
        """Build the professor's table. Sheet1 = accuracy; then one supplementary sheet per metric.

        The 9 Jan 2026 request was for a main accuracy table PLUS "supplementary tables: precision,
        recall, and F1-score for each model". Putting those metrics in the SAME grid layout as the
        accuracy table means they can be dropped straight into the manuscript as supplementary
        tables, instead of being read out of a long-format list.
        """
        import openpyxl
        from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

        wb_pv = openpyxl.Workbook()

        def _fill_metric_sheet(ws, metric_key, metric_label, title):
            """Write one metric into a worksheet using the professor's exact template layout."""
            ws["A1"] = title                                    # Row 1: what this table contains
            ws["A2"] = ("250 spectra per Sample ID (fixed seed), Leave-One-Group-Out CV with Sample ID "
                        "as the grouping variable")             # Row 2: the validation design
            if run_timestamp:                                   # Row 3: provenance for reproducibility
                ws["A3"] = f"Run completed: {run_timestamp}"

            ws["B4"] = "Raw spectra"                            # Three top-level column groups,
            ws["E4"] = "Second derivative spectra"              # exactly as in the blank template
            ws["H4"] = "Peak ratios"
            ws.merge_cells("B4:D4")
            ws.merge_cells("E4:G4")
            ws.merge_cells("H4:H5")

            ws["B5"] = _REGION_LABELS["Fingerprint"]            # Region sub-headers under each group
            ws["C5"] = _REGION_LABELS["Collagen"]
            ws["D5"] = _REGION_LABELS["Mineral"]
            ws["E5"] = _REGION_LABELS["Fingerprint"]
            ws["F5"] = _REGION_LABELS["Collagen"]
            ws["G5"] = _REGION_LABELS["Mineral"]

            for model_name, row in _MODEL_ROW.items():          # One row per model, fixed order
                ws.cell(row=row, column=1, value=model_name)

            for (sheet_name, region_key), col_letter in _TEMPLATE_COLUMNS.items():
                per_model = all_results.get((sheet_name, region_key), {})
                for model_name, row in _MODEL_ROW.items():
                    res = per_model.get(model_name)             # May be absent if not run
                    if res is not None and res.get(metric_key) is not None:
                        val = res.get(metric_key)               # The metric for this cell
                        try:
                            ws[f"{col_letter}{row}"] = round(float(val), 2)
                        except (TypeError, ValueError):
                            pass                                # Leave blank rather than write NaN

            thin = Side(style="thin", color="999999")           # Light grid on every populated cell
            for row in ws.iter_rows(min_row=1, max_row=10, min_col=1, max_col=8):
                for cell in row:
                    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                    cell.border = Border(left=thin, right=thin, top=thin, bottom=thin)

            for coord in ["B4", "E4", "H4"]:                    # Dark headers for the three groups
                ws[coord].font = Font(bold=True, color="FFFFFF")
                ws[coord].fill = PatternFill("solid", fgColor="1F3864")
            for coord in ["B5", "C5", "D5", "E5", "F5", "G5"]:  # Lighter headers for the regions
                ws[coord].font = Font(bold=True, color="FFFFFF")
                ws[coord].fill = PatternFill("solid", fgColor="2E75B6")

            for model_name, row in _MODEL_ROW.items():          # Model names bold and left-aligned
                c = ws.cell(row=row, column=1)
                c.font = Font(bold=True)
                c.alignment = Alignment(horizontal="left", vertical="center")

            for row in ws.iter_rows(min_row=6, max_row=10, min_col=2, max_col=8):
                for cell in row:                                # Shade by value, so patterns pop out
                    if isinstance(cell.value, (int, float)):
                        v = cell.value
                        cell.number_format = "0.00"
                        shade = (
                            "DCE6F1" if v < 0.7 else
                            "9DC3E6" if v < 0.8 else
                            "2E75B6" if v < 0.9 else
                            "1F4E79"
                        )
                        cell.fill = PatternFill("solid", fgColor=shade)
                        cell.font = Font(bold=True, color="000000" if v < 0.8 else "FFFFFF")

            ws.column_dimensions["A"].width = 22                # Column widths matching the template
            for col in "BCDEFGH":
                ws.column_dimensions[col].width = 26
            ws.row_dimensions[1].height = 30
            ws.row_dimensions[2].height = 30
            ws.row_dimensions[4].height = 20
            ws.row_dimensions[5].height = 40

        # Sheet1 stays exactly as before: the main accuracy table the professor asked to fill.
        _fill_metric_sheet(
            wb_pv.active, "overall_accuracy", "Accuracy",
            "Filled with accuracy values — Sample-ID-grouped Leave-One-Group-Out validation")
        wb_pv.active.title = "Sheet1"

        # Supplementary tables, same layout, one metric each (9 Jan 2026 request).
        for _key, _sheet_name, _title in [
            ("precision", "Precision (Suppl.)",
             "SUPPLEMENTARY — Precision (positive class = OIM), Sample-ID-grouped LOGO validation"),
            ("recall", "Recall (Suppl.)",
             "SUPPLEMENTARY — Recall / sensitivity (positive class = OIM), Sample-ID-grouped LOGO validation"),
            ("f1", "F1-score (Suppl.)",
             "SUPPLEMENTARY — F1-score (positive class = OIM), Sample-ID-grouped LOGO validation"),
            ("roc_auc", "ROC-AUC (Suppl.)",
             "SUPPLEMENTARY — ROC-AUC (positive class = OIM), Sample-ID-grouped LOGO validation"),
        ]:
            _fill_metric_sheet(wb_pv.create_sheet(_sheet_name), _key, _sheet_name, _title)

        out = io.BytesIO()
        wb_pv.save(out)
        return out.getvalue()

    def _pv_build_accuracy_table_excel_OLD(all_results, run_timestamp=None):
        import openpyxl
        from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

        wb_pv = openpyxl.Workbook()
        ws = wb_pv.active
        ws.title = "Sheet1"

        ws["A1"] = "Filled with accuracy values — Sample-ID-grouped Leave-One-Group-Out validation"
        ws["A2"] = "250 spectra per Sample ID (fixed seed), Leave-One-Group-Out CV with Sample ID as the grouping variable"
        if run_timestamp:
            ws["A3"] = f"Run completed: {run_timestamp}"

        ws["B4"] = "Raw spectra"
        ws["E4"] = "Second derivative spectra"
        ws["H4"] = "Peak ratios"
        ws.merge_cells("B4:D4")
        ws.merge_cells("E4:G4")
        ws.merge_cells("H4:H5")

        ws["B5"] = _REGION_LABELS["Fingerprint"]
        ws["C5"] = _REGION_LABELS["Collagen"]
        ws["D5"] = _REGION_LABELS["Mineral"]
        ws["E5"] = _REGION_LABELS["Fingerprint"]
        ws["F5"] = _REGION_LABELS["Collagen"]
        ws["G5"] = _REGION_LABELS["Mineral"]

        for model_name, row in _MODEL_ROW.items():
            ws.cell(row=row, column=1, value=model_name)

        for (sheet_name, region_key), col_letter in _TEMPLATE_COLUMNS.items():
            per_model = all_results.get((sheet_name, region_key), {})
            for model_name, row in _MODEL_ROW.items():
                res = per_model.get(model_name)
                if res is not None:
                    ws[f"{col_letter}{row}"] = round(res["overall_accuracy"], 2)

        thin = Side(style="thin", color="999999")
        for row in ws.iter_rows(min_row=1, max_row=10, min_col=1, max_col=8):
            for cell in row:
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                cell.border = Border(left=thin, right=thin, top=thin, bottom=thin)

        for coord in ["B4", "E4", "H4"]:
            ws[coord].font = Font(bold=True, color="FFFFFF")
            ws[coord].fill = PatternFill("solid", fgColor="1F3864")
        for coord in ["B5", "C5", "D5", "E5", "F5", "G5"]:
            ws[coord].font = Font(bold=True, color="FFFFFF")
            ws[coord].fill = PatternFill("solid", fgColor="2E75B6")

        for model_name, row in _MODEL_ROW.items():
            c = ws.cell(row=row, column=1)
            c.font = Font(bold=True)
            c.alignment = Alignment(horizontal="left", vertical="center")

        for row in ws.iter_rows(min_row=6, max_row=10, min_col=2, max_col=8):
            for cell in row:
                if isinstance(cell.value, (int, float)):
                    v = cell.value
                    cell.number_format = "0.00"
                    shade = (
                        "DCE6F1" if v < 0.7 else
                        "9DC3E6" if v < 0.8 else
                        "2E75B6" if v < 0.9 else
                        "1F4E79"
                    )
                    cell.fill = PatternFill("solid", fgColor=shade)
                    cell.font = Font(bold=True, color="000000" if v < 0.8 else "FFFFFF")

        ws.column_dimensions["A"].width = 22
        for col in "BCDEFGH":
            ws.column_dimensions[col].width = 26
        ws.row_dimensions[1].height = 30
        ws.row_dimensions[2].height = 30
        ws.row_dimensions[4].height = 20
        ws.row_dimensions[5].height = 40

        out = io.BytesIO()
        wb_pv.save(out)
        return out.getvalue()

    def _pv_build_detailed_workbook(all_results, counts_by_sheet, shortfalls_by_sheet, classes_by_sheet, n_per_animal, seed, run_timestamp=None):
        summary_rows, fold_rows, cm_rows, count_rows = [], [], [], []
        for (sheet_name, region_key), per_model in all_results.items():
            classes = classes_by_sheet.get(sheet_name, ["0", "1"])
            for model_name, res in per_model.items():
                ft = res["fold_table"]
                summary_rows.append({
                    "Run timestamp": run_timestamp or "unknown",
                    "Sheet": sheet_name, "Feature set": region_key, "Model": model_name,
                    "Number of features": res["n_features"], "Sample-ID groups (folds)": len(ft),
                    "Spectra per Sample ID": n_per_animal, "Random seed": seed,
                    "Overall accuracy (pooled across all 12 held-out Sample ID groups)": res["overall_accuracy"],
                    # Supplementary metrics requested on 9 Jan 2026 ("precision, recall and
                    # F1-score for each model"), written straight into the Excel workbook so they
                    # arrive in the editable format the professor asked for.
                    "Precision": res.get("precision", float("nan")),
                    "Recall": res.get("recall", float("nan")),
                    "F1-score": res.get("f1", float("nan")),
                    "ROC-AUC": res.get("roc_auc", float("nan")),
                    "Positive class": res.get("positive_class", "OIM"),
                    "Mean per-group accuracy": ft["Fold accuracy"].mean(),
                    "Std per-group accuracy": ft["Fold accuracy"].std(),
                    "Min per-group accuracy": ft["Fold accuracy"].min(),
                    "Max per-group accuracy": ft["Fold accuracy"].max(),
                })
                ft2 = ft.copy()
                ft2.insert(0, "Model", model_name)
                ft2.insert(0, "Feature set", region_key)
                ft2.insert(0, "Sheet", sheet_name)
                fold_rows.append(ft2)

                cm = res["confusion_matrix"]
                row = {"Sheet": sheet_name, "Feature set": region_key, "Model": model_name}
                for i, ci in enumerate(classes):
                    for j, cj in enumerate(classes):
                        if i < cm.shape[0] and j < cm.shape[1]:
                            row[f"True {ci} -> Pred {cj}"] = int(cm[i, j])
                # ROW PERCENTAGES as well as counts. On the 12 Aug call the professor said the
                # confusion-matrix presentation was "perfect ... because you have the percent, but
                # you also have the number of cells", so the editable table needs both. Percentages
                # are per TRUE class (row-normalised), which is what makes them readable as
                # sensitivity/specificity rather than as a share of the whole matrix.
                for i, ci in enumerate(classes):
                    _rt = float(cm[i, :].sum()) if i < cm.shape[0] else 0.0   # Row total
                    for j, cj in enumerate(classes):
                        if i < cm.shape[0] and j < cm.shape[1]:
                            row[f"True {ci} -> Pred {cj} (%)"] = (
                                round(100.0 * cm[i, j] / _rt, 2) if _rt > 0 else 0.0)
                if cm.shape[0] >= 2 and cm.shape[1] >= 2:          # Convenience summary columns
                    _tot = float(cm.sum())
                    row["Correct (n)"] = int(np.trace(cm))          # Diagonal = correct
                    row["Overall accuracy (%)"] = round(100.0 * np.trace(cm) / _tot, 2) if _tot else 0.0
                cm_rows.append(row)

        for sheet_name, counts in counts_by_sheet.items():
            c = counts.copy()
            c.insert(0, "Sheet", sheet_name)
            shortfall = shortfalls_by_sheet.get(sheet_name, [])
            id_col = c.columns[1]
            short_ids = {a for a, _ in shortfall}
            c["Had fewer than requested?"] = c[id_col].apply(lambda a: "YES" if a in short_ids else "no")
            count_rows.append(c)

        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)
            pd.concat(fold_rows, ignore_index=True).to_excel(writer, sheet_name="Per-Group Accuracy", index=False)
            pd.DataFrame(cm_rows).to_excel(writer, sheet_name="Confusion Matrices", index=False)
            if count_rows:
                pd.concat(count_rows, ignore_index=True).to_excel(writer, sheet_name="Spectra Counts Used", index=False)
        return out.getvalue()

    def _pv_build_comparison_workbook(all_results, model_choices, n_per_animal, seed, run_timestamp=None):
        # Builds the "for-professor" workbook every run: New (this LOGO-CV run) accuracy
        # only, one tab per dataset (Raw spectra / 2nd derivative / Quant peak ratios),
        # plus a Summary & Notes tab and a Per-Group Detail tab built from the live
        # fold_table data in all_results.
        # (The old prior-methodology reference numbers and the old-only "Quant Peak
        # Intensity" column were removed at the user's request -- they were only ever
        # there for sanity-checking the LOGO-CV fix against the old random-split
        # results and aren't needed in the deliverable shared with the professor.)
        import openpyxl
        from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
        from openpyxl.utils import get_column_letter
        from openpyxl.worksheet.table import Table, TableStyleInfo

        FONT_NAME = "Arial"
        TITLE_FILL = PatternFill("solid", fgColor="1F3864")
        HEADER_FILL = PatternFill("solid", fgColor="2E75B6")
        SUBHEADER_FILL = PatternFill("solid", fgColor="D9E2F3")
        MODEL_FILL = PatternFill("solid", fgColor="F2F2F2")
        WT_FILL = PatternFill("solid", fgColor="DDEBF7")
        OIM_FILL = PatternFill("solid", fgColor="FCE4D6")
        WHITE_BOLD = Font(name=FONT_NAME, bold=True, color="FFFFFF")
        BLACK_BOLD = Font(name=FONT_NAME, bold=True, color="000000")
        NORMAL = Font(name=FONT_NAME, color="000000")
        THIN = Side(style="thin", color="999999")
        BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
        CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
        LEFT = Alignment(horizontal="left", vertical="center", wrap_text=True)

        def _style(cell, font=None, fill=None, align=None, fmt=None, border=True):
            if font:
                cell.font = font
            if fill:
                cell.fill = fill
            cell.alignment = align or CENTER
            if fmt:
                cell.number_format = fmt
            if border:
                cell.border = BORDER

        models_run = [m for m in _MODEL_ORDER if m in model_choices]

        wb_cmp = openpyxl.Workbook()
        ws_sum = wb_cmp.active
        ws_sum.title = "Summary & Notes"
        ws_sum.sheet_view.showGridLines = False

        lr_zero_sheets = []
        if "Logistic Regression" in models_run:
            for (sheet_name, region_key), per_model in all_results.items():
                res = per_model.get("Logistic Regression")
                if res and region_key != "ratio" and round(res["overall_accuracy"], 4) == 0.0:
                    lr_zero_sheets.append(f"{sheet_name} / {region_key}")

        ws_sum.merge_cells("A1:F1")
        ws_sum["A1"] = "Sample-ID Grouped Leave-One-Group-Out Validation — Summary for Professor"
        _style(ws_sum["A1"], font=Font(name=FONT_NAME, bold=True, size=14, color="FFFFFF"), fill=TITLE_FILL, align=LEFT, border=False)
        ws_sum.row_dimensions[1].height = 24

        lines = [
            ("Why this was run", True),
            ("A reviewer flagged that spectra from the same Sample ID group could appear in both the training and test sets of a "
             "random split, and that Sample ID groups contribute unequal amounts of data. This run uses Leave-One-Group-Out (LOGO) "
             "cross-validation with Sample ID as the grouping variable, so no Sample ID group's spectra ever appear in both train "
             "and test, plus a fixed number of spectra per Sample ID to remove the imbalance concern.", False),
            ("", False),
            ("Run settings (this run)", True),
            (f"Run completed: {run_timestamp or 'unknown'}.", False),
            (f"Spectra per Sample ID: {n_per_animal}.  Random seed: {seed}.  Models run: {', '.join(models_run)}.", False),
            ("", False),
            ("Logistic Regression — one bug fix applied", True),
            ("Logistic Regression alone gets per-fold feature scaling (StandardScaler, fit on the 11 training "
             "Sample ID groups only, applied to the 1 held-out Sample ID group — no leakage) before fitting. Earlier runs without "
             "this fix reported exactly 0.00 accuracy on every held-out Sample ID group for the spectral sheets (Raw, "
             "2nd derivative) because the solver doesn't converge on hundreds of unscaled, correlated wavenumber "
             "features. This is a methodology bug fix, not a change to the validation strategy, and does not "
             "affect SVM, KNN, Random Forest, or XGBoost.", False),
            ("", False),
            ("How to read the other tabs", True),
            ("Each tab (Raw spectra, 2nd derivative, Quant peak ratios) shows this run's LOGO-CV accuracy per "
             "model/region. The Per-Group Detail tab lists the exact held-out accuracy for every Sample ID group, model, "
             "and feature set from this run.", False),
            ("", False),
        ]
        if lr_zero_sheets:
            lines.append(("Logistic Regression — still 0.00 even with scaling applied", True))
            lines.append((
                "Logistic Regression still returned exactly 0.00 accuracy (zero variance across every held-out "
                "Sample ID group) on: " + "; ".join(lr_zero_sheets) + " — even though per-fold feature scaling is now applied "
                "(see the note above). That rules out the simple unscaled-solver explanation, so this would need "
                "closer investigation (e.g. checking class balance per fold, solver choice, or regularization) "
                "before these specific cells go in the manuscript.", False))
            lines.append(("", False))

        r = 3
        for text, is_header in lines:
            ws_sum.merge_cells(start_row=r, start_column=1, end_row=r, end_column=6)
            cell = ws_sum.cell(row=r, column=1, value=text)
            if is_header:
                _style(cell, font=Font(name=FONT_NAME, bold=True, size=11, color="1F3864"), align=LEFT, border=False)
            else:
                _style(cell, font=Font(name=FONT_NAME, size=10), align=Alignment(horizontal="left", vertical="top", wrap_text=True), border=False)
                ws_sum.row_dimensions[r].height = 30 if len(text) > 80 else 16
            r += 1
        ws_sum.column_dimensions["A"].width = 18
        for c in "BCDEF":
            ws_sum.column_dimensions[c].width = 18

        for tab_title, sheet_key, title_text in [
            ("Raw spectra", "Raw", "Raw spectra — accuracy results"),
            ("2nd derivative", "2nd derivative", "2nd derivative spectra — accuracy results"),
        ]:
            ws = wb_cmp.create_sheet(tab_title)
            ws.sheet_view.showGridLines = False
            ws.merge_cells("A1:D1")
            ws["A1"] = title_text
            _style(ws["A1"], font=Font(name=FONT_NAME, bold=True, size=13, color="FFFFFF"), fill=TITLE_FILL, align=LEFT, border=False)
            ws.merge_cells("A2:D2")
            ws["A2"] = f"Leave-One-Group-Out CV, {n_per_animal} spectra/Sample ID, seed {seed}."
            _style(ws["A2"], font=Font(name=FONT_NAME, italic=True, size=9), align=LEFT, border=False)

            row0 = 4
            ws.cell(row=row0, column=1, value="Model")
            _style(ws.cell(row=row0, column=1), font=WHITE_BOLD, fill=HEADER_FILL)

            col = 2
            for region_key in ["Fingerprint", "Collagen", "Mineral"]:
                c = ws.cell(row=row0, column=col, value=_REGION_LABELS[region_key])
                _style(c, font=WHITE_BOLD, fill=HEADER_FILL)
                col += 1

            r2 = row0 + 1
            for model_name in models_run:
                ws.cell(row=r2, column=1, value=model_name)
                _style(ws.cell(row=r2, column=1), font=BLACK_BOLD, fill=MODEL_FILL, align=LEFT)
                col = 2
                for region_key in ["Fingerprint", "Collagen", "Mineral"]:
                    res = all_results.get((sheet_key, region_key), {}).get(model_name)
                    new_val = round(res["overall_accuracy"], 4) if res else None
                    ws.cell(row=r2, column=col, value=new_val)
                    _style(ws.cell(row=r2, column=col), font=NORMAL, fmt="0.00")
                    col += 1
                r2 += 1

            ws.column_dimensions["A"].width = 20
            for c in range(2, col):
                ws.column_dimensions[get_column_letter(c)].width = 14
            ws.row_dimensions[1].height = 22
            ws.row_dimensions[row0].height = 24
            ws.freeze_panes = "B5"

        ws = wb_cmp.create_sheet("Quant peak ratios")
        ws.sheet_view.showGridLines = False
        ws.merge_cells("A1:B1")
        ws["A1"] = "Quant peak ratios — accuracy results"
        _style(ws["A1"], font=Font(name=FONT_NAME, bold=True, size=13, color="FFFFFF"), fill=TITLE_FILL, align=LEFT, border=False)
        ws.merge_cells("A2:B2")
        ws["A2"] = f"Leave-One-Group-Out CV, {n_per_animal} spectra/Sample ID, seed {seed}."
        _style(ws["A2"], font=Font(name=FONT_NAME, italic=True, size=9), align=LEFT, border=False)

        headers = ["Model", "Accuracy"]
        for j, h in enumerate(headers, start=1):
            c = ws.cell(row=4, column=j, value=h)
            _style(c, font=WHITE_BOLD, fill=HEADER_FILL)

        r2 = 5
        for model_name in models_run:
            res = all_results.get(("Quant peak ratios", "ratio"), {}).get(model_name)
            new_val = round(res["overall_accuracy"], 4) if res else None
            ws.cell(row=r2, column=1, value=model_name)
            _style(ws.cell(row=r2, column=1), font=BLACK_BOLD, fill=MODEL_FILL, align=LEFT)
            ws.cell(row=r2, column=2, value=new_val)
            _style(ws.cell(row=r2, column=2), font=NORMAL, fmt="0.00")
            r2 += 1
        ws.column_dimensions["A"].width = 20
        ws.column_dimensions["B"].width = 12
        ws.row_dimensions[1].height = 22
        ws.row_dimensions[4].height = 20

        ws = wb_cmp.create_sheet("Per-Group Detail")
        ws.sheet_view.showGridLines = False
        ws.merge_cells("A1:G1")
        ws["A1"] = "Per-Group Detail — exact held-out accuracy for every Sample ID group, model, and feature set"
        _style(ws["A1"], font=Font(name=FONT_NAME, bold=True, size=13, color="FFFFFF"), fill=TITLE_FILL, align=LEFT, border=False)
        ws.merge_cells("A2:G2")
        ws["A2"] = "Each row is one LOGO-CV fold: the named Sample ID group held out as the test set for that model/feature set."
        _style(ws["A2"], font=Font(name=FONT_NAME, italic=True, size=9), align=LEFT, border=False)

        # No hard-coded WT/OIM Sample-ID list is written here. The report is driven entirely by
        # the uploaded file and the actual LOGO folds.
        start_row = 4
        headers2 = ["Sheet", "Feature set", "Model", "Held-out Sample ID", "Test spectra", "Fold accuracy"]
        for j, h in enumerate(headers2, start=1):
            c = ws.cell(row=start_row, column=j, value=h)
            _style(c, font=WHITE_BOLD, fill=HEADER_FILL)

        r3 = start_row + 1
        for (sheet_name, region_key), per_model in all_results.items():
            for model_name, res in per_model.items():
                if model_name not in models_run:
                    continue
                for _, fold in res["fold_table"].iterrows():
                    animal = fold["Held-out Sample ID"]
                    ws.cell(row=r3, column=1, value=sheet_name)
                    ws.cell(row=r3, column=2, value=region_key)
                    ws.cell(row=r3, column=3, value=model_name)
                    ws.cell(row=r3, column=4, value=animal)
                    ws.cell(row=r3, column=5, value=int(fold["Test spectra"]))
                    acc_cell = ws.cell(row=r3, column=6, value=float(fold["Fold accuracy"]))
                    acc_cell.number_format = "0.000"
                    for col_i in range(1, 7):
                        ws.cell(row=r3, column=col_i).font = NORMAL
                        ws.cell(row=r3, column=col_i).border = BORDER
                        ws.cell(row=r3, column=col_i).alignment = CENTER
                    r3 += 1
        last_row = r3 - 1
        if last_row >= start_row + 1:
            tab = Table(displayName="PerAnimalDetailLive", ref=f"A{start_row}:G{last_row}")
            tab.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True)
            ws.add_table(tab)
        for colc, w in {"A": 16, "B": 18, "C": 22, "D": 16, "E": 9, "F": 12, "G": 13}.items():
            ws.column_dimensions[colc].width = w
        ws.column_dimensions["I"].width = 10
        ws.column_dimensions["J"].width = 8
        ws.freeze_panes = f"A{start_row + 1}"

        out = io.BytesIO()
        wb_cmp.save(out)
        return out.getvalue()


    # ──────────────────────────────────────────────────────────────────────────
    # Publication figure helpers for the SAME balanced 250-spectra/animal data.
    # These helpers are isolated inside Professor mode and do not alter the
    # original application analyses above.
    # ──────────────────────────────────────────────────────────────────────────
    def _pv_prepare_figure_data(xls_obj, sheet_name, region_key, n_per_animal, seed):
        df_fig = xls_obj.parse(sheet_name)
        group_col, label_col = _pv_find_group_label_columns(df_fig)

        for c in df_fig.columns:
            if c not in (group_col, label_col):
                df_fig[c] = pd.to_numeric(df_fig[c], errors="coerce")

        balanced_fig, shortfalls_fig = _pv_balance_per_animal(df_fig, group_col, n_per_animal, seed)
        if shortfalls_fig:
            raise ValueError(
                f"Some Sample ID groups have fewer than {n_per_animal} spectra: {shortfalls_fig}. "
                "Figures must use the same balanced dataset as the validation."
            )

        if _SHEET_KIND[sheet_name] == "spectral":
            lo, hi = _REGIONS[region_key]
            feature_cols = _pv_region_columns(balanced_fig, group_col, label_col, lo, hi)
        else:
            feature_cols = _pv_ratio_columns(balanced_fig, group_col, label_col)

        if not feature_cols:
            raise ValueError(f"No usable features found for {sheet_name} — {region_key}.")

        clean = balanced_fig[[group_col, label_col] + feature_cols].copy()
        clean[feature_cols] = clean[feature_cols].astype(float).fillna(0.0)
        clean[group_col] = clean[group_col].astype(str)
        clean[label_col] = clean[label_col].astype(str)

        le_fig = LabelEncoder()
        y_codes = le_fig.fit_transform(clean[label_col])
        classes = list(le_fig.classes_)
        if set(classes) != {"WT", "OIM"}:
            raise ValueError(f"Expected WT/OIM labels, but found {classes}.")

        return {
            "df": clean,
            "X": clean[feature_cols],
            "y_raw": clean[label_col],
            "y": y_codes,
            "groups": clean[group_col],
            "feature_cols": feature_cols,
            "group_col": group_col,
            "label_col": label_col,
            "classes": classes,
        }

    def _pv_fig_png(fig, dpi=300, autocrop=True):
        """Render a figure to PNG bytes, trimming any blank border that survives bbox_inches.

        WHY THE EXTRA CROP: shap's summary_plot and waterfall size their own figure and place
        their axes by hand. bbox_inches="tight" only crops to the AXES bounding box, so when the
        drawn content occupies just part of those axes - which is what happens when shap's own
        sizing does not match the canvas - a blank band survives underneath the plot. Cropping
        the rendered pixels to their non-white content removes it whatever the cause, and is a
        no-op for figures that were already tight.
        """
        buf = io.BytesIO()                                   # In-memory destination
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
        raw = buf.getvalue()                                 # Bytes as matplotlib produced them
        if not autocrop:                                     # Caller can opt out
            return raw
        try:                                                 # Pillow ships with matplotlib
            from PIL import Image                            # Imported lazily, only when cropping
            im = Image.open(io.BytesIO(raw)).convert("RGB")  # Load the rendered image
            arr = np.asarray(im)                             # Height x Width x 3 pixel array
            nonwhite = (arr < 247).any(axis=2)               # True where anything was drawn
            rows = np.where(nonwhite.any(axis=1))[0]         # Image rows containing content
            cols = np.where(nonwhite.any(axis=0))[0]         # Image columns containing content
            if rows.size == 0 or cols.size == 0:             # Completely blank figure - leave it
                return raw
            pad = max(4, int(0.01 * max(im.size)))           # Small uniform margin, ~1% of size
            box = (max(int(cols[0]) - pad, 0), max(int(rows[0]) - pad, 0),
                   min(int(cols[-1]) + pad + 1, im.width),
                   min(int(rows[-1]) + pad + 1, im.height))  # Crop rectangle around the content
            out = io.BytesIO()                               # Buffer for the cropped image
            im.crop(box).save(out, format="PNG")             # Write the trimmed PNG
            return out.getvalue()                            # Cropped bytes
        except Exception:                                    # Any problem - fall back safely
            return raw                                       # Never let cropping break an export

    def _pv_safe_feature_number(col):
        try:
            return float(str(col).split("/")[0].split("_")[0])
        except Exception:
            return None

    def _pv_confidence_ellipse(ax, x, y, n_std=2.0, **kwargs):
        from matplotlib.patches import Ellipse
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
            return
        cov = np.cov(x, y)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * n_std * np.sqrt(np.maximum(vals, 0))
        ellipse = Ellipse((np.mean(x), np.mean(y)), width, height, angle=angle, fill=False, **kwargs)
        ax.add_patch(ellipse)

    def _pv_make_pca_figures(fig_data):
        X = fig_data["X"]
        y_raw = fig_data["y_raw"]
        groups = fig_data["groups"]
        feature_cols = fig_data["feature_cols"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        n_comp = min(3, X_scaled.shape[1], X_scaled.shape[0] - 1)
        pca = PCA(n_components=n_comp, random_state=42)
        scores = pca.fit_transform(X_scaled)

        # PCA score plot
        fig_scores, ax = plt.subplots(figsize=(9, 7))
        markers = {"WT": "o", "OIM": "^"}
        colors = {"WT": "#2E86AB", "OIM": "#D7263D"}   # One fixed colour per phenotype
        for cls in ["WT", "OIM"]:
            mask = (y_raw.values == cls)
            ax.scatter(
                scores[mask, 0], scores[mask, 1], s=20, alpha=0.45,
                marker=markers.get(cls, "o"), label=cls, c=colors[cls], edgecolors="none"
            )
            # Colour the 95% ellipse to match its class - drawing both in black makes the
            # figure unreadable, since there is no way to tell which ellipse belongs to which group.
            _pv_confidence_ellipse(ax, scores[mask, 0], scores[mask, 1], n_std=2.0, linewidth=2,
                                   edgecolor=colors[cls])
        ax.axhline(0, linewidth=0.7, alpha=0.4)
        ax.axvline(0, linewidth=0.7, alpha=0.4)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
        # PCA itself is unsupervised, so it has no intrinsic classification accuracy.
        # To satisfy the manuscript request for an accuracy shown beside the PCA score plot,
        # we report a SEPARATE animal-held-out classifier operating on PC1+PC2 only:
        # StandardScaler -> PCA(2) -> LogisticRegression, all fitted inside each LOGO fold.
        _pca_oof_pred = np.full(len(y_raw), -1, dtype=int)
        _pca_y_codes = LabelEncoder().fit_transform(y_raw.astype(str))
        _pca_logo = LeaveOneGroupOut()
        for _tr, _te in _pca_logo.split(X, _pca_y_codes, groups):
            _sc_fold = StandardScaler().fit(X.iloc[_tr])
            _Xtr_z = _sc_fold.transform(X.iloc[_tr])
            _Xte_z = _sc_fold.transform(X.iloc[_te])
            _n_pc_fold = min(2, _Xtr_z.shape[1], max(1, _Xtr_z.shape[0] - 1))
            _pca_fold = PCA(n_components=_n_pc_fold, random_state=42).fit(_Xtr_z)
            _Ztr = _pca_fold.transform(_Xtr_z)
            _Zte = _pca_fold.transform(_Xte_z)
            _lr_fold = LogisticRegression(max_iter=3000, random_state=42)
            _lr_fold.fit(_Ztr, _pca_y_codes[_tr])
            _pca_oof_pred[_te] = _lr_fold.predict(_Zte)
        _pca_logo_accuracy = float(accuracy_score(_pca_y_codes, _pca_oof_pred))

        ax.set_title("PCA scores — balanced dataset (250 spectra per Sample ID)")
        ax.text(
            0.985, 0.965,
            f"PCA+LR LOGO accuracy = {_pca_logo_accuracy * 100:.2f}%",
            transform=ax.transAxes,
            ha="right", va="top", fontsize=10.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.35", alpha=0.92),
        )
        ax.legend(title="Phenotype")
        fig_scores.tight_layout()
        fig_scores._pv_pca_logo_accuracy = _pca_logo_accuracy

        # PCA + Logistic Regression animal-held-out confusion matrix.
        _pca_classes = list(LabelEncoder().fit(y_raw.astype(str)).classes_)
        _pca_cm = confusion_matrix(_pca_y_codes, _pca_oof_pred, labels=list(range(len(_pca_classes))))
        _pca_cm_pct = _pca_cm / np.maximum(_pca_cm.sum(axis=1, keepdims=True), 1) * 100
        fig_pca_cm, ax_pcm = plt.subplots(figsize=(6.8, 5.8))
        im_pcm = ax_pcm.imshow(_pca_cm_pct, vmin=0, vmax=100, cmap="Blues")
        ax_pcm.set_xticks(range(len(_pca_classes)), _pca_classes)
        ax_pcm.set_yticks(range(len(_pca_classes)), _pca_classes)
        ax_pcm.set_xlabel("Predicted phenotype")
        ax_pcm.set_ylabel("True phenotype")
        ax_pcm.set_title("PCA (PC1+PC2) + Logistic Regression — pooled LOGO confusion matrix (%)")
        for _i in range(_pca_cm.shape[0]):
            for _j in range(_pca_cm.shape[1]):
                ax_pcm.text(_j, _i, f"{_pca_cm_pct[_i, _j]:.1f}%\n(n={_pca_cm[_i, _j]})", ha="center", va="center")
        fig_pca_cm.colorbar(im_pcm, ax=ax_pcm, label="% of true class")
        fig_pca_cm.tight_layout()

        # PCA explained variance
        fig_var, ax2 = plt.subplots(figsize=(7, 5))
        comp_labels = [f"PC{i+1}" for i in range(n_comp)]
        ax2.bar(comp_labels, pca.explained_variance_ratio_ * 100)
        ax2.set_ylabel("Explained variance (%)")
        ax2.set_title("PCA explained variance")
        for i, v in enumerate(pca.explained_variance_ratio_ * 100):
            ax2.text(i, v, f"{v:.1f}%", ha="center", va="bottom")
        fig_var.tight_layout()

        # PCA loadings, line for spectral sheets / ranked bars for ratios
        feature_numbers = [_pv_safe_feature_number(c) for c in feature_cols]
        if all(v is not None for v in feature_numbers) and len(feature_cols) > 10:
            order = np.argsort(feature_numbers)
            xvals = np.asarray(feature_numbers)[order]
            fig_load, ax3 = plt.subplots(figsize=(10, 5.5))
            ax3.plot(xvals, pca.components_[0][order], label="PC1 loading")
            ax3.plot(xvals, pca.components_[1][order], label="PC2 loading")
            ax3.axhline(0, linewidth=0.7)
            ax3.set_xlabel("Wavenumber (cm⁻¹)")
            ax3.set_ylabel("Loading")
            ax3.set_title("PCA loadings")
            ax3.invert_xaxis()
            ax3.legend()
            fig_load.tight_layout()
        else:
            importance = np.sqrt(pca.components_[0] ** 2 + pca.components_[1] ** 2)
            top_idx = np.argsort(importance)[-20:][::-1]
            fig_load, ax3 = plt.subplots(figsize=(9, 6))
            ax3.barh([str(feature_cols[i]) for i in top_idx][::-1], importance[top_idx][::-1])
            ax3.set_xlabel("Combined |PC1/PC2 loading|")
            ax3.set_title("Top PCA loading features")
            fig_load.tight_layout()

        return {
            "PCA_scores": fig_scores,
            "PCA_explained_variance": fig_var,
            "PCA_loadings": fig_load,
            "PCA_LOGO_confusion_matrix": fig_pca_cm,
        }

    def _pv_vip_scores(pls_model, X_scaled, y_vector):
        # Standard VIP formula for PLS with one response variable.
        t = pls_model.x_scores_
        w = pls_model.x_weights_
        q = pls_model.y_loadings_
        p, h = w.shape
        s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
        total_s = np.sum(s)
        if total_s == 0:
            return np.zeros(p)
        return np.sqrt(p * (w ** 2 @ s).ravel() / total_s)

    def _pv_make_plsda_figures(fig_data):
        X = fig_data["X"]
        y = fig_data["y"]
        y_raw = fig_data["y_raw"]
        feature_cols = fig_data["feature_cols"]

        groups = fig_data["groups"]
        n_comp = min(3, X.shape[1], X.shape[0] - 1)

        # ANIMAL-BOUND PLS-DA SCORES (this is the important correction).
        # PLS-DA is a SUPERVISED method: it uses the WT/OIM labels to build its latent axes.
        # Fitting it on the full dataset and then plotting those same spectra shows an
        # optimistically separated cloud - exactly the concern the Dev et al. reviewer raised,
        # one level up. Instead, for each of the 12 folds we fit PLS-DA on 11 animals and
        # project ONLY the held-out Sample ID group into that model's latent space. Every point below is
        # therefore positioned by a model that never saw it.
        scores = np.full((len(y), max(2, n_comp)), np.nan)
        # Store the held-out PLS-DA prediction for every spectrum as well.  This is used
        # ONLY for the accuracy printed on the PLS-DA score figure.  The prediction for
        # an animal is always produced by the fold model trained on the remaining Sample ID groups.
        _plsda_oof_continuous = np.full(len(y), np.nan, dtype=float)
        _logo_pls = LeaveOneGroupOut()
        for _tr, _te in _logo_pls.split(X, y, groups):
            _sc = StandardScaler().fit(X.iloc[_tr])              # scaler fitted on TRAIN animals only
            _p = PLSRegression(n_components=n_comp, scale=False)  # fresh model for this fold
            _p.fit(_sc.transform(X.iloc[_tr]), y[_tr].astype(float).reshape(-1, 1))
            _X_te_scaled = _sc.transform(X.iloc[_te])
            _t = _p.transform(_X_te_scaled)                       # project the unseen animal
            scores[_te, :_t.shape[1]] = _t                        # store its out-of-fold coordinates
            _plsda_oof_continuous[_te] = np.asarray(_p.predict(_X_te_scaled)).reshape(-1)

        # Binary PLS-DA classification: labels were encoded as 0/1, therefore 0.5 is the
        # natural classification threshold.  This is a true pooled LOGO accuracy because
        # every prediction comes from a model that never saw that animal.
        _plsda_oof_pred = (_plsda_oof_continuous >= 0.5).astype(int)
        _plsda_logo_accuracy = accuracy_score(y, _plsda_oof_pred)

        # A separate full-data fit is kept ONLY to compute VIP, which is an interpretation
        # statistic describing the whole dataset rather than a per-sample prediction.
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pls = PLSRegression(n_components=n_comp, scale=False)
        pls.fit(X_scaled, y.astype(float).reshape(-1, 1))

        fig_scores, ax = plt.subplots(figsize=(9, 7))
        markers = {"WT": "o", "OIM": "^"}
        colors = {"WT": "#2E86AB", "OIM": "#D7263D"}   # Same colours as the PCA panel, for consistency
        for cls in ["WT", "OIM"]:
            mask = (y_raw.values == cls)
            ax.scatter(scores[mask, 0], scores[mask, 1], s=20, alpha=0.45,
                       marker=markers.get(cls, "o"), label=cls, c=colors[cls], edgecolors="none")
            # Ellipse colour matches the class, so the two 95% ellipses are distinguishable.
            _pv_confidence_ellipse(ax, scores[mask, 0], scores[mask, 1], n_std=2.0, linewidth=2,
                                   edgecolor=colors[cls])
        ax.axhline(0, linewidth=0.7, alpha=0.4)
        ax.axvline(0, linewidth=0.7, alpha=0.4)
        ax.set_xlabel("PLS-DA latent variable 1 (out-of-fold)")
        ax.set_ylabel("PLS-DA latent variable 2 (out-of-fold)")
        ax.set_title("PLS-DA scores — Sample-ID-grouped leave-one-group-out projection\n"
                     "(250 spectra per Sample ID; each Sample ID group projected by a model that never saw it)")
        # Professor requested the classification accuracy directly on the PLS-DA figure.
        # This is the pooled held-out LOGO accuracy, not training accuracy.
        ax.text(
            0.985, 0.965,
            f"LOGO accuracy = {_plsda_logo_accuracy * 100:.2f}%",
            transform=ax.transAxes,
            ha="right", va="top", fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.35", alpha=0.92),
        )
        ax.legend(title="Phenotype")
        fig_scores.tight_layout()
        # Attach the value to the figure object so the UI/export layer can reuse the exact
        # same accuracy without recomputing it differently.
        fig_scores._pv_plsda_logo_accuracy = float(_plsda_logo_accuracy)

        # PLS-DA pooled animal-held-out confusion matrix from the SAME OOF predictions
        # that produce the accuracy printed on the score plot.
        _plsda_classes = list(fig_data["classes"])
        _plsda_cm = confusion_matrix(y, _plsda_oof_pred, labels=list(range(len(_plsda_classes))))
        _plsda_cm_pct = _plsda_cm / np.maximum(_plsda_cm.sum(axis=1, keepdims=True), 1) * 100
        fig_plsda_cm, ax_pcm = plt.subplots(figsize=(6.8, 5.8))
        im_pcm = ax_pcm.imshow(_plsda_cm_pct, vmin=0, vmax=100, cmap="Blues")
        ax_pcm.set_xticks(range(len(_plsda_classes)), _plsda_classes)
        ax_pcm.set_yticks(range(len(_plsda_classes)), _plsda_classes)
        ax_pcm.set_xlabel("Predicted phenotype")
        ax_pcm.set_ylabel("True phenotype")
        ax_pcm.set_title("PLS-DA — pooled LOGO confusion matrix (%)")
        for _i in range(_plsda_cm.shape[0]):
            for _j in range(_plsda_cm.shape[1]):
                ax_pcm.text(_j, _i, f"{_plsda_cm_pct[_i, _j]:.1f}%\n(n={_plsda_cm[_i, _j]})", ha="center", va="center")
        fig_plsda_cm.colorbar(im_pcm, ax=ax_pcm, label="% of true class")
        fig_plsda_cm.tight_layout()

        vip = _pv_vip_scores(pls, X_scaled, y)
        top_idx = np.argsort(vip)[-25:][::-1]
        fig_vip, ax2 = plt.subplots(figsize=(9, 7))
        ax2.barh([str(feature_cols[i]) for i in top_idx][::-1], vip[top_idx][::-1])
        ax2.axvline(1.0, linestyle="--", linewidth=1.2, label="VIP = 1")
        ax2.set_xlabel("VIP score")
        ax2.set_title("PLS-DA variable importance (top 25)")
        ax2.legend()
        fig_vip.tight_layout()

        return {
            "PLSDA_scores": fig_scores,
            "PLSDA_VIP": fig_vip,
            "PLSDA_LOGO_confusion_matrix": fig_plsda_cm,
        }


    def _pv_quantitative_editable_score_workbooks(fig_data, seed):
        """Create Excel-editable PCA and PLS-DA score plots for quantitative parameters.

        The PCA workbook contains the balanced-data PC1/PC2 coordinates and an editable
        Excel scatter chart.  The PLS-DA workbook contains the animal-held-out LOGO LV1/LV2
        coordinates, held-out predictions, and the pooled LOGO accuracy used on the figure.
        """
        X = fig_data["X"]
        y = np.asarray(fig_data["y"])
        y_raw = fig_data["y_raw"].astype(str).reset_index(drop=True)
        groups = fig_data["groups"].astype(str).reset_index(drop=True)
        classes = list(fig_data["classes"])
        n_comp = min(3, X.shape[1], X.shape[0] - 1)

        # ---------------- PCA: same computation as the on-screen PCA score plot ----------------
        pca_scaler = StandardScaler()
        X_pca_scaled = pca_scaler.fit_transform(X)
        pca_model = PCA(n_components=n_comp, random_state=42)
        pca_scores = pca_model.fit_transform(X_pca_scaled)
        pca_df = pd.DataFrame({
            "Sample ID": groups,
            "Phenotype": y_raw,
            "PC1": pca_scores[:, 0],
            "PC2": pca_scores[:, 1],
        })
        if pca_scores.shape[1] >= 3:
            pca_df["PC3"] = pca_scores[:, 2]

        pca_summary = pd.DataFrame({
            "Parameter": [
                "Analysis", "Dataset", "Spectra per Sample ID", "Random seed",
                "PC1 explained variance (%)", "PC2 explained variance (%)"
            ],
            "Value": [
                "PCA (unsupervised, balanced quantitative parameters)",
                "Quantitative spectral parameters / peak ratios",
                250, seed,
                float(pca_model.explained_variance_ratio_[0] * 100),
                float(pca_model.explained_variance_ratio_[1] * 100),
            ]
        })

        pca_buffer = io.BytesIO()
        with pd.ExcelWriter(pca_buffer, engine="xlsxwriter") as writer:
            pca_df.to_excel(writer, sheet_name="PCA Scores", index=False)
            pca_summary.to_excel(writer, sheet_name="Summary", index=False)
            wb = writer.book
            ws = writer.sheets["PCA Scores"]
            ws.freeze_panes(1, 0)
            for col_i, width in enumerate([18, 13, 14, 14, 14]):
                ws.set_column(col_i, col_i, width)

            chart = wb.add_chart({"type": "scatter"})
            helper_col = max(len(pca_df.columns) + 2, 7)
            helper_row = 1
            phenotype_styles = {
                "WT": {"type": "circle", "size": 5, "border": {"color": "#2E86AB"}, "fill": {"color": "#2E86AB"}},
                "OIM": {"type": "triangle", "size": 6, "border": {"color": "#D7263D"}, "fill": {"color": "#D7263D"}},
            }
            for phenotype in [c for c in ["WT", "OIM"] if c in set(pca_df["Phenotype"])]:
                sub = pca_df[pca_df["Phenotype"] == phenotype][["PC1", "PC2"]].reset_index(drop=True)
                ws.write(helper_row - 1, helper_col, f"{phenotype} PC1")
                ws.write(helper_row - 1, helper_col + 1, f"{phenotype} PC2")
                ws.write_column(helper_row, helper_col, sub["PC1"].tolist())
                ws.write_column(helper_row, helper_col + 1, sub["PC2"].tolist())
                start = helper_row
                end = helper_row + len(sub) - 1
                chart.add_series({
                    "name": phenotype,
                    "categories": ["PCA Scores", start, helper_col, end, helper_col],
                    "values": ["PCA Scores", start, helper_col + 1, end, helper_col + 1],
                    "marker": phenotype_styles.get(phenotype, {"type": "circle", "size": 5}),
                    "line": {"none": True},
                })
                helper_row = end + 3

            chart.set_title({"name": "PCA scores — quantitative parameters (250 spectra per Sample ID)"})
            chart.set_x_axis({"name": f"PC1 ({pca_model.explained_variance_ratio_[0] * 100:.1f}% variance)"})
            chart.set_y_axis({"name": f"PC2 ({pca_model.explained_variance_ratio_[1] * 100:.1f}% variance)"})
            chart.set_legend({"position": "bottom"})
            chart.set_size({"width": 780, "height": 560})
            ws.insert_chart(1, len(pca_df.columns) + 8, chart)
        pca_bytes = pca_buffer.getvalue()

        # ---------------- PLS-DA: same Sample-ID-grouped LOGO logic as the figure ----------------
        pls_scores = np.full((len(y), max(2, n_comp)), np.nan)
        oof_cont = np.full(len(y), np.nan, dtype=float)
        oof_pred = np.full(len(y), -1, dtype=int)
        logo = LeaveOneGroupOut()
        for tr, te in logo.split(X, y, groups):
            scaler = StandardScaler().fit(X.iloc[tr])
            pls = PLSRegression(n_components=n_comp, scale=False)
            pls.fit(scaler.transform(X.iloc[tr]), y[tr].astype(float).reshape(-1, 1))
            X_te_scaled = scaler.transform(X.iloc[te])
            t = pls.transform(X_te_scaled)
            pls_scores[te, :t.shape[1]] = t
            pred_cont = np.asarray(pls.predict(X_te_scaled)).reshape(-1)
            oof_cont[te] = pred_cont
            oof_pred[te] = (pred_cont >= 0.5).astype(int)

        logo_accuracy = float(accuracy_score(y, oof_pred))
        pred_labels = np.asarray([classes[int(i)] for i in oof_pred], dtype=object)
        pls_df = pd.DataFrame({
            "Sample ID": groups,
            "Phenotype": y_raw,
            "PLSDA1 (OOF)": pls_scores[:, 0],
            "PLSDA2 (OOF)": pls_scores[:, 1],
            "PLS-DA continuous prediction (OOF)": oof_cont,
            "Predicted phenotype (OOF)": pred_labels,
            "Correct": pred_labels == y_raw.to_numpy(),
        })
        if pls_scores.shape[1] >= 3:
            pls_df["PLSDA3 (OOF)"] = pls_scores[:, 2]

        pls_summary = pd.DataFrame({
            "Parameter": [
                "Analysis", "Dataset", "Spectra per Sample ID", "Random seed",
                "Validation", "Grouping variable", "Classification target", "Pooled LOGO accuracy", "Pooled LOGO accuracy (%)"
            ],
            "Value": [
                "PLS-DA (Sample-ID-grouped out-of-fold projection)",
                "Quantitative spectral parameters / peak ratios",
                250, seed,
                "Leave-One-Group-Out (one Sample ID group held out per fold)",
                "Sample ID", "WT vs OIM",
                logo_accuracy, logo_accuracy * 100,
            ]
        })

        pls_buffer = io.BytesIO()
        with pd.ExcelWriter(pls_buffer, engine="xlsxwriter") as writer:
            pls_df.to_excel(writer, sheet_name="PLSDA OOF Scores", index=False)
            pls_summary.to_excel(writer, sheet_name="Summary", index=False)
            wb = writer.book
            ws = writer.sheets["PLSDA OOF Scores"]
            ws.freeze_panes(1, 0)
            ws.set_column(0, 0, 18)
            ws.set_column(1, 1, 13)
            ws.set_column(2, 3, 16)
            ws.set_column(4, 7, 28)

            chart = wb.add_chart({"type": "scatter"})
            helper_col = max(len(pls_df.columns) + 2, 10)
            helper_row = 1
            phenotype_styles = {
                "WT": {"type": "circle", "size": 5, "border": {"color": "#2E86AB"}, "fill": {"color": "#2E86AB"}},
                "OIM": {"type": "triangle", "size": 6, "border": {"color": "#D7263D"}, "fill": {"color": "#D7263D"}},
            }
            for phenotype in [c for c in ["WT", "OIM"] if c in set(pls_df["Phenotype"])]:
                sub = pls_df[pls_df["Phenotype"] == phenotype][["PLSDA1 (OOF)", "PLSDA2 (OOF)"]].reset_index(drop=True)
                ws.write(helper_row - 1, helper_col, f"{phenotype} LV1")
                ws.write(helper_row - 1, helper_col + 1, f"{phenotype} LV2")
                ws.write_column(helper_row, helper_col, sub["PLSDA1 (OOF)"].tolist())
                ws.write_column(helper_row, helper_col + 1, sub["PLSDA2 (OOF)"].tolist())
                start = helper_row
                end = helper_row + len(sub) - 1
                chart.add_series({
                    "name": phenotype,
                    "categories": ["PLSDA OOF Scores", start, helper_col, end, helper_col],
                    "values": ["PLSDA OOF Scores", start, helper_col + 1, end, helper_col + 1],
                    "marker": phenotype_styles.get(phenotype, {"type": "circle", "size": 5}),
                    "line": {"none": True},
                })
                helper_row = end + 3

            chart.set_title({"name": f"PLS-DA OOF scores — LOGO accuracy = {logo_accuracy * 100:.2f}%"})
            chart.set_x_axis({"name": "PLS-DA latent variable 1 (out-of-fold)"})
            chart.set_y_axis({"name": "PLS-DA latent variable 2 (out-of-fold)"})
            chart.set_legend({"position": "bottom"})
            chart.set_size({"width": 780, "height": 560})
            ws.insert_chart(1, len(pls_df.columns) + 8, chart)
        pls_bytes = pls_buffer.getvalue()

        return {
            "pca_xlsx": pca_bytes,
            "plsda_xlsx": pls_bytes,
            "plsda_logo_accuracy": logo_accuracy,
            "pca_scores_df": pca_df,
            "plsda_scores_df": pls_df,
        }


    def _pv_bone_band_annotation(wn):
        """Bone FTIR sub-region annotation used in the main app_k.py-style interpretation tables."""
        try:
            w = float(wn)
        except Exception:
            return ""
        # More specific named peaks first.
        specific = [
            (1650, 8, "Amide I (Protein/Collagen)"),
            (1550, 8, "Amide II (Protein/Collagen)"),
            (1338, 8, "CH (Collagen)"),
            (1115, 8, "HPO4 (Bone Mineral)"),
            (1030, 12, "PO4 Stretching (Bone Mineral)"),
            (875, 8, "CO3 Bending (Carbonate)"),
        ]
        for center, tol, label in specific:
            if abs(w-center) <= tol:
                return label
        ranges = [
            (1600, 1700, "Amide I (1700–1600 cm⁻¹)"),
            (1500, 1600, "Amide II (1600–1500 cm⁻¹)"),
            (1430, 1490, "CH (1490–1430 cm⁻¹)"),
            (1400, 1430, "CH/CO3 (1430–1400 cm⁻¹)"),
            (1330, 1350, "CH (1350–1330 cm⁻¹)"),
            (1200, 1300, "Amide III (1300–1200 cm⁻¹)"),
            (1100, 1200, "HPO4 (1200–1100 cm⁻¹)"),
            (930, 1100, "PO4 (1100–930 cm⁻¹)"),
            (860, 890, "CO3 (890–860 cm⁻¹)"),
        ]
        for lo, hi, label in ranges:
            if lo <= w <= hi:
                return label
        return "Other fingerprint feature"

    def _pv_feature_number_or_nan(v):
        try:
            return float(str(v).split('/')[0].split('_')[0].strip())
        except Exception:
            return np.nan

    def _pv_class_side_from_scores(scores_1d, y_raw):
        tmp = pd.DataFrame({"Phenotype": pd.Series(y_raw).astype(str).values, "Score": np.asarray(scores_1d, dtype=float)})
        means = tmp.groupby("Phenotype", observed=True)["Score"].mean().sort_values(ascending=False)
        if len(means) < 2:
            return None, None
        return str(means.index[0]), str(means.index[-1])

    def _pv_favors_from_sign(v, positive_class, negative_class):
        if positive_class is None or negative_class is None or pd.isna(v):
            return ""
        return f"favors {positive_class}" if float(v) >= 0 else f"favors {negative_class}"

    def _pv_row_pct_df(cm_df):
        arr = cm_df.to_numpy(dtype=float)
        den = np.maximum(arr.sum(axis=1, keepdims=True), 1)
        pct = arr / den * 100.0
        return pd.DataFrame(pct, index=cm_df.index, columns=cm_df.columns)

    def _pv_make_interpretation_figures(export_data, sheet_name, region_key):
        """app_k.py-style annotated loading figures + score comparison + CM figures for one region."""
        figs = {}
        # PCA annotated loading line (PC1)
        pldf = export_data["pca_loadings_all"].copy()
        num = pd.to_numeric(pldf["Numeric Wavenumber"], errors="coerce")
        ok = num.notna()
        if ok.sum() > 3:
            d = pldf.loc[ok].sort_values("Numeric Wavenumber")
            fig, ax = plt.subplots(figsize=(11, 5.2))
            ax.plot(d["Numeric Wavenumber"], d["PC1 loading"], lw=1.8)
            ax.axhline(0, color="black", ls="--", lw=0.8)
            ax.set_xlabel("Wavenumber (cm⁻¹)")
            ax.set_ylabel("Loading")
            ax.set_title(f"{sheet_name} — {region_key} — PCA PC1 loadings")
            # annotate a small set of biologically meaningful points that fall inside the region
            targets=[(1650,"Amide I"),(1550,"Amide II"),(1338,"CH"),(1115,"HPO4"),(1030,"PO4"),(875,"CO3")]
            for target,label in targets:
                ix=(d["Numeric Wavenumber"]-target).abs().idxmin()
                if abs(float(d.loc[ix,"Numeric Wavenumber"])-target) <= 15:
                    x=float(d.loc[ix,"Numeric Wavenumber"]); y=float(d.loc[ix,"PC1 loading"])
                    ax.axvline(x, color="0.45", ls="--", lw=0.8)
                    ax.annotate(label, xy=(x,y), xytext=(0,18), textcoords="offset points", ha="center", fontsize=8,
                                arrowprops=dict(arrowstyle="-|>", lw=0.7, color="0.35"))
            fig.tight_layout(); figs["PCA_PC1_loadings_annotated"] = fig

        # PLS-DA annotated loading line (LV1) from the full balanced interpretation fit.
        ldf = export_data["plsda_loadings_all"].copy()
        num2 = pd.to_numeric(ldf["Numeric Wavenumber"], errors="coerce")
        ok2 = num2.notna()
        if ok2.sum() > 3:
            d = ldf.loc[ok2].sort_values("Numeric Wavenumber")
            fig, ax = plt.subplots(figsize=(11, 5.2))
            ax.plot(d["Numeric Wavenumber"], d["PLSDA1 loading"], lw=1.8)
            ax.axhline(0, color="black", ls="--", lw=0.8)
            ax.set_xlabel("Wavenumber (cm⁻¹)")
            ax.set_ylabel("Loading")
            ax.set_title(f"{sheet_name} — {region_key} — PLS-DA LV1 loadings (interpretation fit)")
            targets=[(1650,"Amide I"),(1550,"Amide II"),(1338,"CH"),(1115,"HPO4"),(1030,"PO4"),(875,"CO3")]
            for target,label in targets:
                ix=(d["Numeric Wavenumber"]-target).abs().idxmin()
                if abs(float(d.loc[ix,"Numeric Wavenumber"])-target) <= 15:
                    x=float(d.loc[ix,"Numeric Wavenumber"]); y=float(d.loc[ix,"PLSDA1 loading"])
                    ax.axvline(x, color="0.45", ls="--", lw=0.8)
                    ax.annotate(label, xy=(x,y), xytext=(0,18), textcoords="offset points", ha="center", fontsize=8,
                                arrowprops=dict(arrowstyle="-|>", lw=0.7, color="0.35"))
            fig.tight_layout(); figs["PLSDA_LV1_loadings_annotated"] = fig

        # PCA + PLS-DA score comparison, matching the legacy app_k visual structure.
        ps=export_data["pca_scores"]; ds=export_data["plsda_scores"]
        fig, axes=plt.subplots(1,2,figsize=(14,5.5))
        for ax,df,x,y,title,acc in [
            (axes[0],ps,"PC1","PC2","PCA",export_data["pca_accuracy"]),
            (axes[1],ds,"PLSDA1 (OOF)","PLSDA2 (OOF)","PLS-DA",export_data["plsda_accuracy"]),
        ]:
            for cls, marker in [("WT","o"),("OIM","^")]:
                sub=df[df["Phenotype"].astype(str)==cls]
                ax.scatter(sub[x],sub[y],s=14,alpha=.5,label=cls,marker=marker)
                _pv_confidence_ellipse(ax, sub[x], sub[y], n_std=2.0, linewidth=1.6)
            ax.axhline(0,lw=.6,alpha=.35); ax.axvline(0,lw=.6,alpha=.35)
            ax.set_xlabel(x.replace(" (OOF)","")); ax.set_ylabel(y.replace(" (OOF)","")); ax.set_title(title)
            ax.text(.98,.97,f"Accuracy = {acc*100:.2f}%",transform=ax.transAxes,ha="right",va="top",fontsize=9,
                    bbox=dict(boxstyle="round,pad=.25",fc="white",ec="0.5",alpha=.9))
        axes[0].legend(title="Class",frameon=False)
        fig.suptitle(f"{sheet_name} — {region_key} — PCA vs PLS-DA")
        fig.tight_layout(); figs["PCA_vs_PLSDA_scores"] = fig

        # Confusion matrices side-by-side.
        fig, axes = plt.subplots(1,2,figsize=(10.5,4.6))
        for ax,cm,title in [(axes[0],export_data["pca_confusion"],"PCA+LR LOGO"),(axes[1],export_data["plsda_confusion"],"PLS-DA LOGO")]:
            pct=_pv_row_pct_df(cm); im=ax.imshow(pct.to_numpy(),vmin=0,vmax=100,cmap="Blues")
            labels=[str(i).replace("True ","") for i in cm.index]
            ax.set_xticks(range(len(labels)),labels); ax.set_yticks(range(len(labels)),labels)
            ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    ax.text(j,i,f"{pct.iloc[i,j]:.1f}%\n(n={int(cm.iloc[i,j])})",ha="center",va="center",fontsize=8)
        fig.suptitle(f"{sheet_name} — {region_key} — pooled Sample-ID-held-out confusion matrices")
        fig.tight_layout(); figs["PCA_PLSDA_confusion_matrices"] = fig
        return figs

    def _pv_region_analysis_export_data(fig_data, seed):
        """Complete region result package using the 250-per-Sample-ID + LOGO design.

        Descriptive quantities: full-balanced PCA scores/loadings and full-balanced PLS-DA loadings/VIP.
        Validation quantities: PCA+LR and PLS-DA predictions are pooled out-of-fold Sample-ID-held-out results.
        """
        X = fig_data["X"].reset_index(drop=True)
        y = np.asarray(fig_data["y"])
        y_raw = fig_data["y_raw"].astype(str).reset_index(drop=True)
        groups = fig_data["groups"].astype(str).reset_index(drop=True)
        feature_cols = list(fig_data["feature_cols"])
        classes = list(fig_data["classes"])

        # PCA descriptive fit on the balanced dataset.
        pca_scaler = StandardScaler().fit(X)
        Xz = pca_scaler.transform(X)
        n_full = min(3, Xz.shape[1], Xz.shape[0] - 1)
        pca_full = PCA(n_components=n_full, random_state=42).fit(Xz)
        pca_scores = pca_full.transform(Xz)
        pca_scores_df = pd.DataFrame({"Sample ID": groups, "Phenotype": y_raw, "PC1": pca_scores[:,0], "PC2": pca_scores[:,1]})
        if pca_scores.shape[1] >= 3: pca_scores_df["PC3"] = pca_scores[:,2]
        pca_pos, pca_neg = _pv_class_side_from_scores(pca_scores[:,0], y_raw)
        pca_loadings_df = pd.DataFrame({
            "Wavenumber (cm⁻¹)": [str(c) for c in feature_cols],
            "Numeric Wavenumber": [_pv_feature_number_or_nan(c) for c in feature_cols],
            "PC1 loading": pca_full.components_[0],
            "PC2 loading": pca_full.components_[1],
        })
        if pca_full.components_.shape[0] >= 3: pca_loadings_df["PC3 loading"] = pca_full.components_[2]
        pca_loadings_df["|PC1 loading|"] = pca_loadings_df["PC1 loading"].abs()
        pca_loadings_df["Combined |PC1/PC2|"] = np.sqrt(pca_loadings_df["PC1 loading"]**2 + pca_loadings_df["PC2 loading"]**2)
        pca_loadings_df["Band Annotation"] = pca_loadings_df["Numeric Wavenumber"].map(_pv_bone_band_annotation)
        pca_loadings_df["Interpretation"] = pca_loadings_df["PC1 loading"].map(lambda v:_pv_favors_from_sign(v,pca_pos,pca_neg))
        pca_influential = pca_loadings_df.sort_values("|PC1 loading|",ascending=False).reset_index(drop=True)
        pca_importance = pca_loadings_df.sort_values("Combined |PC1/PC2|",ascending=False).reset_index(drop=True)

        # PCA+LR Sample-ID-held-out validation using PC1+PC2 fitted inside each fold.
        pca_oof_pred=np.full(len(y),-1,dtype=int); logo=LeaveOneGroupOut()
        for tr,te in logo.split(X,y,groups):
            sc=StandardScaler().fit(X.iloc[tr]); Xtrz=sc.transform(X.iloc[tr]); Xtez=sc.transform(X.iloc[te])
            n_pc=min(2,Xtrz.shape[1],max(1,Xtrz.shape[0]-1)); pf=PCA(n_components=n_pc,random_state=42).fit(Xtrz)
            lr=LogisticRegression(max_iter=3000,random_state=seed).fit(pf.transform(Xtrz),y[tr])
            pca_oof_pred[te]=lr.predict(pf.transform(Xtez))
        pca_acc=float(accuracy_score(y,pca_oof_pred)); pca_cm=confusion_matrix(y,pca_oof_pred,labels=list(range(len(classes))))
        pca_cm_df=pd.DataFrame(pca_cm,index=[f"True {c}" for c in classes],columns=[f"Pred {c}" for c in classes])
        pca_cm_pct_df=_pv_row_pct_df(pca_cm_df)

        # PLS-DA out-of-fold scores + predictions. Every Sample ID is positioned/predicted by a fold that excluded it.
        n_pls=min(3,X.shape[1],X.shape[0]-1); pls_scores=np.full((len(y),max(2,n_pls)),np.nan); pls_cont=np.full(len(y),np.nan); pls_pred=np.full(len(y),-1,dtype=int)
        for tr,te in logo.split(X,y,groups):
            sc=StandardScaler().fit(X.iloc[tr]); Xtr=sc.transform(X.iloc[tr]); Xte=sc.transform(X.iloc[te])
            pls=PLSRegression(n_components=n_pls,scale=False).fit(Xtr,y[tr].astype(float).reshape(-1,1))
            t=pls.transform(Xte); pls_scores[te,:t.shape[1]]=t
            cont=np.asarray(pls.predict(Xte)).reshape(-1); pls_cont[te]=cont; pls_pred[te]=(cont>=0.5).astype(int)
        pls_acc=float(accuracy_score(y,pls_pred)); pred_labels=np.asarray([classes[int(v)] for v in pls_pred],dtype=object)
        pls_scores_df=pd.DataFrame({"Sample ID":groups,"Phenotype":y_raw,"PLSDA1 (OOF)":pls_scores[:,0],"PLSDA2 (OOF)":pls_scores[:,1],
                                    "Continuous prediction (OOF)":pls_cont,"Predicted phenotype (OOF)":pred_labels,"Correct":pred_labels==y_raw.to_numpy()})
        if pls_scores.shape[1]>=3: pls_scores_df["PLSDA3 (OOF)"]=pls_scores[:,2]
        pls_cm=confusion_matrix(y,pls_pred,labels=list(range(len(classes)))); pls_cm_df=pd.DataFrame(pls_cm,index=[f"True {c}" for c in classes],columns=[f"Pred {c}" for c in classes]); pls_cm_pct_df=_pv_row_pct_df(pls_cm_df)

        # Full balanced PLS-DA interpretation fit only for loadings and VIP (not used for reported validation accuracy).
        sc_full=StandardScaler().fit(X); Xfull=sc_full.transform(X)
        pls_full=PLSRegression(n_components=n_pls,scale=False).fit(Xfull,y.astype(float).reshape(-1,1))
        full_scores=pls_full.transform(Xfull); pls_pos,pls_neg=_pv_class_side_from_scores(full_scores[:,0],y_raw)
        vip=_pv_vip_scores(pls_full,Xfull,y)
        pls_loadings_df=pd.DataFrame({
            "Wavenumber (cm⁻¹)": [str(c) for c in feature_cols],
            "Numeric Wavenumber": [_pv_feature_number_or_nan(c) for c in feature_cols],
            "PLSDA1 loading": pls_full.x_loadings_[:,0],
        })
        if pls_full.x_loadings_.shape[1]>=2: pls_loadings_df["PLSDA2 loading"] = pls_full.x_loadings_[:,1]
        if pls_full.x_loadings_.shape[1]>=3: pls_loadings_df["PLSDA3 loading"] = pls_full.x_loadings_[:,2]
        pls_loadings_df["|PLSDA1 loading|"] = pls_loadings_df["PLSDA1 loading"].abs()
        pls_loadings_df["Band Annotation"] = pls_loadings_df["Numeric Wavenumber"].map(_pv_bone_band_annotation)
        pls_loadings_df["Interpretation"] = pls_loadings_df["PLSDA1 loading"].map(lambda v:_pv_favors_from_sign(v,pls_pos,pls_neg))
        pls_influential=pls_loadings_df.sort_values("|PLSDA1 loading|",ascending=False).reset_index(drop=True)
        vip_df=pd.DataFrame({
            "Wavenumber (cm⁻¹)": [str(c) for c in feature_cols],
            "VIP": vip,
            "Abs Loading (selected)": np.abs(pls_full.x_loadings_[:,0]),
            "Numeric Wavenumber": [_pv_feature_number_or_nan(c) for c in feature_cols],
        })
        vip_df["Band Annotation"] = vip_df["Numeric Wavenumber"].map(_pv_bone_band_annotation)
        vip_df=vip_df.sort_values("VIP",ascending=False).reset_index(drop=True)

        return {
            "pca_scores":pca_scores_df,"pca_loadings":pca_loadings_df,"pca_loadings_all":pca_loadings_df,
            "pca_influential":pca_influential,"pca_importance":pca_importance,"pca_accuracy":pca_acc,
            "pca_confusion":pca_cm_df,"pca_confusion_pct":pca_cm_pct_df,"pca_explained_variance":pca_full.explained_variance_ratio_,
            "plsda_scores":pls_scores_df,"plsda_accuracy":pls_acc,"plsda_confusion":pls_cm_df,"plsda_confusion_pct":pls_cm_pct_df,
            "plsda_loadings":pls_loadings_df,"plsda_loadings_all":pls_loadings_df,"plsda_influential":pls_influential,"plsda_vip":vip_df,
            # VIP ordered by WAVENUMBER (1800 -> 800), not by magnitude. The ranked table above is
            # useful for reading off the strongest variables, but it cannot be plotted against the
            # spectrum: its x-order is 1600, 1598, 1602, 1596 ... Professor Querido asked for VIP
            # across the whole range so it can be overlaid on the mean spectra, which needs this.
            "plsda_vip_by_wn":vip_df.sort_values("Numeric Wavenumber",ascending=False).reset_index(drop=True),
            "classes":classes,
        }

    def _pv_add_editable_scatter_series(wb, ws, chart, df, xcol, ycol, phen_col="Phenotype", helper_col=10):
        styles = {
            "WT": {"type": "circle", "size": 5, "border": {"color": "#2E86AB"}, "fill": {"color": "#2E86AB"}},
            "OIM": {"type": "triangle", "size": 6, "border": {"color": "#D7263D"}, "fill": {"color": "#D7263D"}},
        }
        helper_row = 1
        for phenotype in [c for c in ["WT", "OIM"] if c in set(df[phen_col].astype(str))]:
            sub = df[df[phen_col].astype(str) == phenotype][[xcol, ycol]].reset_index(drop=True)
            ws.write(helper_row - 1, helper_col, f"{phenotype} X")
            ws.write(helper_row - 1, helper_col + 1, f"{phenotype} Y")
            ws.write_column(helper_row, helper_col, sub[xcol].tolist())
            ws.write_column(helper_row, helper_col + 1, sub[ycol].tolist())
            start, end = helper_row, helper_row + len(sub) - 1
            chart.add_series({
                "name": phenotype,
                "categories": [ws.get_name(), start, helper_col, end, helper_col],
                "values": [ws.get_name(), start, helper_col + 1, end, helper_col + 1],
                "marker": styles[phenotype],
                "line": {"none": True},
            })
            helper_row = end + 3

    def _pv_editable_region_workbook(fig_data, sheet_name, region_key, seed):
        """Complete app_k.py-style editable workbook for one Professor-validation region."""
        d=_pv_region_analysis_export_data(fig_data,seed); out=io.BytesIO()
        with pd.ExcelWriter(out,engine="xlsxwriter") as writer:
            summary=pd.DataFrame({"Parameter":["Sheet","Feature region","Spectra per Sample ID","Fixed random seed","Validation","Classification target","PCA displayed accuracy","PLS-DA LOGO accuracy","PCA accuracy definition","PLS-DA loadings/VIP definition"],
                                  "Value":[sheet_name,region_key,250,seed,"Leave-One-Group-Out by Sample ID","WT vs OIM",d["pca_accuracy"],d["plsda_accuracy"],
                                           "LOGO Logistic Regression using PC1+PC2; PCA score plot itself is unsupervised",
                                           "Full balanced interpretation fit only; validation accuracy remains out-of-fold LOGO"]})
            summary.to_excel(writer,sheet_name="Summary",index=False)
            d["pca_scores"].to_excel(writer,sheet_name="PCA Scores",index=False)
            d["pca_loadings_all"].to_excel(writer,sheet_name="PCA Loadings",index=False)
            d["pca_influential"].to_excel(writer,sheet_name="PCA Influential",index=False)
            d["pca_importance"].to_excel(writer,sheet_name="PCA Importance",index=False)
            d["pca_confusion"].to_excel(writer,sheet_name="PCA CM Counts")
            d["pca_confusion_pct"].to_excel(writer,sheet_name="PCA CM Percent")
            d["plsda_scores"].to_excel(writer,sheet_name="PLSDA Scores",index=False)
            d["plsda_loadings_all"].to_excel(writer,sheet_name="PLSDA Loadings",index=False)
            d["plsda_influential"].to_excel(writer,sheet_name="PLSDA Influential",index=False)
            d["plsda_vip"].to_excel(writer,sheet_name="PLSDA VIP",index=False)
            d["plsda_vip_by_wn"].to_excel(writer,sheet_name="PLSDA VIP full range",index=False)
            d["plsda_confusion"].to_excel(writer,sheet_name="PLSDA CM Counts")
            d["plsda_confusion_pct"].to_excel(writer,sheet_name="PLSDA CM Percent")
            wb=writer.book
            # Editable score charts
            ws_plot=wb.add_worksheet("Editable Score Plots"); writer.sheets["Editable Score Plots"]=ws_plot
            ws_pca=writer.sheets["PCA Scores"]; pc=wb.add_chart({"type":"scatter"}); _pv_add_editable_scatter_series(wb,ws_pca,pc,d["pca_scores"],"PC1","PC2",helper_col=max(8,len(d["pca_scores"].columns)+2))
            ev=d["pca_explained_variance"]; pc.set_title({"name":f"{sheet_name} — {region_key} — PCA | PCA+LR LOGO accuracy = {d['pca_accuracy']*100:.2f}%"}); pc.set_x_axis({"name":f"PC1 ({ev[0]*100:.1f}% variance)"}); pc.set_y_axis({"name":f"PC2 ({ev[1]*100:.1f}% variance)"}); pc.set_legend({"position":"bottom"}); pc.set_size({"width":720,"height":520}); ws_plot.insert_chart("B2",pc)
            ws_pls=writer.sheets["PLSDA Scores"]; dc=wb.add_chart({"type":"scatter"}); _pv_add_editable_scatter_series(wb,ws_pls,dc,d["plsda_scores"],"PLSDA1 (OOF)","PLSDA2 (OOF)",helper_col=max(10,len(d["plsda_scores"].columns)+2))
            dc.set_title({"name":f"{sheet_name} — {region_key} — PLS-DA OOF | LOGO accuracy = {d['plsda_accuracy']*100:.2f}%"}); dc.set_x_axis({"name":"PLS-DA latent variable 1 (out-of-fold)"}); dc.set_y_axis({"name":"PLS-DA latent variable 2 (out-of-fold)"}); dc.set_legend({"position":"bottom"}); dc.set_size({"width":720,"height":520}); ws_plot.insert_chart("L2",dc)
            # Editable loading/VIP charts matching app_k.py-style exports
            ws_lp=wb.add_worksheet("Editable Loadings VIP"); writer.sheets["Editable Loadings VIP"]=ws_lp
            pws=writer.sheets["PCA Loadings"]; plc=wb.add_chart({"type":"scatter","subtype":"straight"})
            nr=len(d["pca_loadings_all"]); plc.add_series({"name":"PC1 loading","categories":["PCA Loadings",1,1,nr,1],"values":["PCA Loadings",1,2,nr,2],"line":{"width":1.5}}); plc.set_title({"name":"PCA PC1 loadings"}); plc.set_x_axis({"name":"Wavenumber (cm⁻¹)"}); plc.set_y_axis({"name":"Loading"}); plc.set_size({"width":720,"height":450}); ws_lp.insert_chart("B2",plc)
            dws=writer.sheets["PLSDA Loadings"]; dlc=wb.add_chart({"type":"scatter","subtype":"straight"}); nr2=len(d["plsda_loadings_all"]); dlc.add_series({"name":"PLSDA1 loading","categories":["PLSDA Loadings",1,1,nr2,1],"values":["PLSDA Loadings",1,2,nr2,2],"line":{"width":1.5}}); dlc.set_title({"name":"PLS-DA LV1 loadings"}); dlc.set_x_axis({"name":"Wavenumber (cm⁻¹)"}); dlc.set_y_axis({"name":"Loading"}); dlc.set_size({"width":720,"height":450}); ws_lp.insert_chart("L2",dlc)
            # FULL-RANGE VIP CHART: every wavenumber from 1800 to 800, plotted in spectral order
            # so it can be compared directly against the spectrum. A top-20 column chart was the
            # thing the professor specifically objected to, because in spectroscopy the top 20
            # entries are mostly the same band repeated.
            _nvip=len(d["plsda_vip_by_wn"])
            vc=wb.add_chart({"type":"scatter","subtype":"straight"})
            vc.add_series({"name":"VIP","categories":["PLSDA VIP full range",1,3,_nvip,3],
                           "values":["PLSDA VIP full range",1,1,_nvip,1],
                           "line":{"width":1.25,"color":"#222222"}})
            vc.set_title({"name":f"PLS-DA VIP across the full range ({_nvip} wavenumbers)"})
            vc.set_x_axis({"name":"Wavenumber (cm-1)","reverse":True})
            vc.set_y_axis({"name":"VIP score"}); vc.set_legend({"none":True})
            vc.set_size({"width":900,"height":420}); ws_lp.insert_chart("B27",vc)
            # Keep the ranked top-20 as a small secondary chart - useful, but not the main view.
            vc2=wb.add_chart({"type":"column"}); _tn=min(20,len(d["plsda_vip"]))
            vc2.add_series({"name":"VIP","categories":["PLSDA VIP",1,0,_tn,0],"values":["PLSDA VIP",1,1,_tn,1]})
            vc2.set_title({"name":"Top 20 VIP features (ranked, secondary view)"})
            vc2.set_x_axis({"name":"Wavenumber"}); vc2.set_y_axis({"name":"VIP"})
            vc2.set_size({"width":720,"height":380}); ws_lp.insert_chart("B50",vc2)
            for ws in writer.sheets.values():
                try: ws.freeze_panes(1,0)
                except Exception: pass
        return out.getvalue(),d

    def _pv_generate_12_plot_package(xls_obj, n_per_animal, seed):
        """Six regions x (PCA + PLS-DA) = 12 score plots, plus full app_k-style result exports."""
        combos=[("Raw","Fingerprint"),("Raw","Collagen"),("Raw","Mineral"),("2nd derivative","Fingerprint"),("2nd derivative","Collagen"),("2nd derivative","Mineral")]
        website=[]; region_workbooks={}; export_data={}
        for sheet_name,region_key in combos:
            fd=_pv_prepare_figure_data(xls_obj,sheet_name,region_key,n_per_animal,seed)
            fp=_pv_make_pca_figures(fd); fl=_pv_make_plsda_figures(fd); pca_fig=fp["PCA_scores"]; pls_fig=fl["PLSDA_scores"]
            wb_bytes,d=_pv_editable_region_workbook(fd,sheet_name,region_key,seed); interp=_pv_make_interpretation_figures(d,sheet_name,region_key)
            pca_png=_pv_fig_png(pca_fig); pls_png=_pv_fig_png(pls_fig)
            interp_png={k:_pv_fig_png(v) for k,v in interp.items()}
            key=f"{sheet_name}_{region_key}".replace(" ","_"); region_workbooks[key]=wb_bytes; export_data[(sheet_name,region_key)]=d
            website.append({"sheet":sheet_name,"region":region_key,"pca_png":pca_png,"plsda_png":pls_png,"pca_accuracy":d["pca_accuracy"],"plsda_accuracy":d["plsda_accuracy"],
                            "pca_loadings_png":interp_png.get("PCA_PC1_loadings_annotated"),"plsda_loadings_png":interp_png.get("PLSDA_LV1_loadings_annotated"),
                            "comparison_png":interp_png.get("PCA_vs_PLSDA_scores"),"cm_png":interp_png.get("PCA_PLSDA_confusion_matrices"),
                            "pca_influential":d["pca_influential"].head(15),"pca_importance":d["pca_importance"].head(15),"plsda_influential":d["plsda_influential"].head(15),"plsda_vip":d["plsda_vip"].head(15)})
            for f in list(fp.values())+list(fl.values())+list(interp.values()):
                try: plt.close(f)
                except Exception: pass
        # Master workbook: source data + all editable score charts + loading/VIP/CM tables for every region.
        master=io.BytesIO()
        with pd.ExcelWriter(master,engine="xlsxwriter") as writer:
            wb=writer.book; dash=wb.add_worksheet("12 Plot Layout"); writer.sheets["12 Plot Layout"]=dash; summary_rows=[]
            for idx,((sheet_name,region_key),d) in enumerate(export_data.items()):
                safe=f"{sheet_name[:3]}_{region_key[:8]}".replace(" ","_"); pca_sheet=f"PCA_{safe}"[:31]; pls_sheet=f"PLSDA_{safe}"[:31]; load_sheet=f"Load_{safe}"[:31]; pload_sheet=f"PLoad_{safe}"[:31]; vip_sheet=f"VIP_{safe}"[:31]; cm_sheet=f"CM_{safe}"[:31]
                d["pca_scores"].to_excel(writer,sheet_name=pca_sheet,index=False); d["plsda_scores"].to_excel(writer,sheet_name=pls_sheet,index=False)
                d["pca_loadings_all"].to_excel(writer,sheet_name=load_sheet,index=False); d["plsda_loadings_all"].to_excel(writer,sheet_name=pload_sheet,index=False); d["plsda_vip"].to_excel(writer,sheet_name=vip_sheet,index=False)
                d["pca_confusion"].to_excel(writer,sheet_name=cm_sheet,startrow=1,startcol=0); d["pca_confusion_pct"].to_excel(writer,sheet_name=cm_sheet,startrow=7,startcol=0); d["plsda_confusion"].to_excel(writer,sheet_name=cm_sheet,startrow=1,startcol=6); d["plsda_confusion_pct"].to_excel(writer,sheet_name=cm_sheet,startrow=7,startcol=6)
                pws=writer.sheets[pca_sheet]; dws=writer.sheets[pls_sheet]
                pc=wb.add_chart({"type":"scatter"}); _pv_add_editable_scatter_series(wb,pws,pc,d["pca_scores"],"PC1","PC2",helper_col=max(8,len(d["pca_scores"].columns)+2)); ev=d["pca_explained_variance"]; pc.set_title({"name":f"{sheet_name} — {region_key} — PCA | PCA+LR LOGO acc = {d['pca_accuracy']*100:.2f}%"}); pc.set_x_axis({"name":f"PC1 ({ev[0]*100:.1f}% variance)"}); pc.set_y_axis({"name":f"PC2 ({ev[1]*100:.1f}% variance)"}); pc.set_legend({"position":"bottom"}); pc.set_size({"width":700,"height":500})
                dc=wb.add_chart({"type":"scatter"}); _pv_add_editable_scatter_series(wb,dws,dc,d["plsda_scores"],"PLSDA1 (OOF)","PLSDA2 (OOF)",helper_col=max(10,len(d["plsda_scores"].columns)+2)); dc.set_title({"name":f"{sheet_name} — {region_key} — PLS-DA | LOGO acc = {d['plsda_accuracy']*100:.2f}%"}); dc.set_x_axis({"name":"PLS-DA LV1 (OOF)"}); dc.set_y_axis({"name":"PLS-DA LV2 (OOF)"}); dc.set_legend({"position":"bottom"}); dc.set_size({"width":700,"height":500})
                row=1+idx*28; dash.write(row-1,1,f"{sheet_name} — {region_key}"); dash.insert_chart(row,1,pc); dash.insert_chart(row,10,dc)
                summary_rows.append({"Sheet":sheet_name,"Region":region_key,"PCA+LR LOGO accuracy":d["pca_accuracy"],"PLS-DA LOGO accuracy":d["plsda_accuracy"]})
            pd.DataFrame(summary_rows).to_excel(writer,sheet_name="Accuracy Summary",index=False); dash.set_column("A:A",2); dash.set_column("B:R",12)
        return {"website":website,"master_xlsx":master.getvalue(),"region_xlsx":region_workbooks}

    def _pv_run_logo_predictions(fig_data, model_name, seed):
        X = fig_data["X"]
        y = fig_data["y"]
        groups = fig_data["groups"]
        classes = fig_data["classes"]
        model = _pv_build_models(seed)[model_name]
        logo = LeaveOneGroupOut()
        all_true, all_pred = [], []
        animal_rows = []
        for tr, te in logo.split(X, y, groups):
            m = clone(model)
            X_tr, X_te = X.iloc[tr], X.iloc[te]
            # Scale every model per fold (fit on training Sample ID groups only) so the confusion matrix
            # and per-group accuracy match the numbers in the main table exactly.
            scaler = StandardScaler()
            X_tr = pd.DataFrame(scaler.fit_transform(X_tr), columns=X.columns, index=X_tr.index)
            X_te = pd.DataFrame(scaler.transform(X_te), columns=X.columns, index=X_te.index)
            m.fit(X_tr, y[tr])
            pred = m.predict(X_te)
            held = groups.iloc[te].iloc[0]
            animal_rows.append({"Animal": held, "Accuracy": accuracy_score(y[te], pred)})
            all_true.extend(y[te].tolist())
            all_pred.extend(pred.tolist())
        return np.asarray(all_true), np.asarray(all_pred), pd.DataFrame(animal_rows), classes

    def _pv_make_validation_figures(fig_data, model_names, seed):
        """Confusion matrix + per-group accuracy for EVERY model, as requested on 9 Jan 2026.

        Returns one combined grid figure covering all models (the manuscript panel), one combined
        per-group accuracy chart, and the individual per-model figures for the supplementary.
        """
        if isinstance(model_names, str):          # Accept a single name for backward compatibility
            model_names = [model_names]           # Wrap it so the loop below always works
        out = {}                                  # Collects every figure produced here
        cms, animals = {}, {}                     # Raw material for the two combined panels

        for _mn in model_names:                   # One leave-one-group-out pass per model
            _figs = _pv_make_validation_figures_one(fig_data, _mn, seed)   # Existing single-model code
            out[f"LOGO_confusion_matrix_{_mn.replace(' ', '_')}"] = _figs["LOGO_confusion_matrix"]
            out[f"Per_group_accuracy_{_mn.replace(' ', '_')}"] = _figs["Per_group_accuracy"]
            cms[_mn] = _figs["_cm_counts"]        # Raw 2x2 counts, reused by the combined grid
            animals[_mn] = _figs["_animal_df"]    # Per-group accuracies, reused by the combined chart
        classes = fig_data["classes"]             # Class order for the axis tick labels

        # ---- Combined grid: every model's confusion matrix in a single manuscript figure ----
        _n = len(cms)                                          # How many panels the grid needs
        _ncols = min(3, max(1, _n))                            # At most three panels per row
        _nrows = int(np.ceil(_n / _ncols))                     # Enough rows to hold every model
        fig_grid, axes = plt.subplots(_nrows, _ncols, figsize=(4.2 * _ncols, 3.9 * _nrows), squeeze=False)
        for _k, (_mn, _cm) in enumerate(cms.items()):          # Fill one panel per model
            _ax = axes[_k // _ncols][_k % _ncols]              # Locate this model's panel
            _pct = _cm / np.maximum(_cm.sum(axis=1, keepdims=True), 1) * 100   # Row-normalised
            _ax.imshow(_pct, vmin=0, vmax=100, cmap="Blues")   # Heat-map background
            _ax.set_xticks(range(len(classes)), classes)       # Predicted class on the x-axis
            _ax.set_yticks(range(len(classes)), classes)       # True class on the y-axis
            for _i in range(_cm.shape[0]):                     # Annotate every cell
                for _j in range(_cm.shape[1]):                 # with count and row percentage
                    _ax.text(_j, _i, f"{_pct[_i, _j]:.1f}%\n(n={_cm[_i, _j]})", ha="center",
                             va="center", fontsize=8,
                             color="white" if _pct[_i, _j] > 50 else "black")
            _ax.set_title(_mn, fontsize=10)                    # Model name above its panel
            _ax.set_xlabel("Predicted"); _ax.set_ylabel("True")   # Axis meaning
        for _k in range(_n, _nrows * _ncols):                  # Hide unused panels in the last row
            axes[_k // _ncols][_k % _ncols].axis("off")        # Blank rather than an empty box
        fig_grid.suptitle("Pooled out-of-fold confusion matrices — all models "
                          "(leave-one-group-out)", y=1.02)    # Figure title
        fig_grid.tight_layout()                                # Prevent label overlap
        out["LOGO_confusion_matrices_ALL_MODELS"] = fig_grid   # The panel that goes in the paper

        # ---- Combined per-group accuracy: all models side by side, one group per animal ----
        _tbl = pd.DataFrame({_mn: _df.set_index("Animal")["Accuracy"] * 100
                             for _mn, _df in animals.items()})  # Rows = animals, columns = models
        _order = sorted(_tbl.index.astype(str).tolist())
        _tbl = _tbl.reindex(_order)                            # Deterministic order from uploaded Sample IDs
        fig_pa, _axp = plt.subplots(figsize=(max(9, 1.1 * len(_tbl)), 5.5))   # Width scales with animals
        _tbl.plot(kind="bar", ax=_axp, width=0.82)             # Grouped bars, one colour per model
        _axp.set_ylim(0, 100)                                  # Percentage scale
        _axp.set_ylabel("Held-out Sample ID accuracy (%)")        # Y-axis meaning
        _axp.set_xlabel("Held-out Sample ID")   # X-axis meaning
        _axp.axhline(50, color="grey", ls="--", lw=1.0)        # Chance level for two balanced classes
        _axp.set_title("Accuracy for each held-out Sample ID group — all models")   # Panel title
        _axp.legend(fontsize=8, ncol=2, frameon=False)         # Compact legend
        fig_pa.tight_layout()                                  # Tidy spacing
        out["Per_group_accuracy_ALL_MODELS"] = fig_pa          # Supplementary panel
        return out                                             # Every figure this block produced

    def _pv_make_validation_figures_one(fig_data, model_name, seed):
        all_true, all_pred, animal_df, classes = _pv_run_logo_predictions(fig_data, model_name, seed)
        cm = confusion_matrix(all_true, all_pred, labels=list(range(len(classes))))
        cm_pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100

        fig_cm, ax = plt.subplots(figsize=(6.8, 5.8))
        im = ax.imshow(cm_pct, vmin=0, vmax=100, cmap="Blues")
        ax.set_xticks(range(len(classes)), classes)
        ax.set_yticks(range(len(classes)), classes)
        ax.set_xlabel("Predicted phenotype")
        ax.set_ylabel("True phenotype")
        ax.set_title(f"{model_name} — pooled LOGO confusion matrix (%)")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm_pct[i, j]:.1f}%\n(n={cm[i, j]})", ha="center", va="center")
        fig_cm.colorbar(im, ax=ax, label="% of true class")
        fig_cm.tight_layout()

        animal_df = animal_df.sort_values("Animal")
        fig_animal, ax2 = plt.subplots(figsize=(9, 5.5))
        bars = ax2.bar(animal_df["Animal"], animal_df["Accuracy"] * 100)
        ax2.set_ylim(0, 100)
        ax2.set_ylabel("Held-out Sample ID accuracy (%)")
        ax2.set_xlabel("Held-out Sample ID")
        ax2.set_title(f"{model_name} — accuracy for each unseen Sample ID group")
        ax2.tick_params(axis="x", rotation=45)
        for bar, val in zip(bars, animal_df["Accuracy"] * 100):
            ax2.text(bar.get_x() + bar.get_width()/2, val, f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
        fig_animal.tight_layout()

        # The raw counts and the per-group table are returned alongside the figures so the
        # combined all-model panels above can be built without re-running the whole CV.
        return {"LOGO_confusion_matrix": fig_cm, "Per_group_accuracy": fig_animal,
                "_cm_counts": cm, "_animal_df": animal_df}

    def _pv_normalize_shap_values(values):
        if isinstance(values, list):
            values = values[-1]
        arr = np.asarray(values)
        if arr.ndim == 3:
            # Common layouts: samples x features x classes, or classes x samples x features.
            if arr.shape[-1] == 2:
                arr = arr[:, :, 1]
            elif arr.shape[0] == 2:
                arr = arr[1]
            else:
                arr = np.mean(arr, axis=-1)
        return np.asarray(arr)

    def _pv_make_oof_shap_figures(fig_data, model_name, seed, samples_per_animal=25):
        if not HAS_SHAP:
            raise RuntimeError("The shap package is not installed in this environment.")
        if model_name not in ["Random Forest", "XGBoost", "Logistic Regression"]:
            raise ValueError("For rigorous and practical out-of-fold SHAP, choose Random Forest, XGBoost, or Logistic Regression.")

        X = fig_data["X"]
        y = fig_data["y"]
        groups = fig_data["groups"]
        feature_cols = fig_data["feature_cols"]
        base_model = _pv_build_models(seed)[model_name]
        logo = LeaveOneGroupOut()
        shap_parts = []
        data_parts = []
        base_parts = []   # Each fold's expected value = the baseline a waterfall plot starts from

        for fold_no, (tr, te) in enumerate(logo.split(X, y, groups)):
            m = clone(base_model)
            X_tr = X.iloc[tr]
            X_te_all = X.iloc[te]
            y_tr = y[tr]

            # Deterministically explain a subset of the held-out Sample ID group to control runtime.
            n_take = min(int(samples_per_animal), len(X_te_all))
            X_te = X_te_all.sample(n=n_take, random_state=seed + fold_no)

            # CONSISTENCY FIX: scale EVERY model here, exactly as _pv_run_logo_cv now does.
            # Previously only Logistic Regression was scaled, so the tree models being explained
            # were not the same models that produced the accuracy table - the SHAP figures would
            # have described a slightly different model from the one being reported.
            # Fitting the scaler on the 11 training Sample ID groups only keeps this leakage-free.
            scaler = StandardScaler()
            X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr), columns=X.columns, index=X_tr.index)
            X_te_s = pd.DataFrame(scaler.transform(X_te), columns=X.columns, index=X_te.index)
            m.fit(X_tr_s, y_tr)
            if isinstance(m, LogisticRegression):
                explainer = shap.LinearExplainer(m, X_tr_s)   # Exact and fast for linear models
            else:
                explainer = shap.TreeExplainer(m)             # Exact and fast for tree ensembles
            vals = explainer.shap_values(X_te_s)
            # Scaled values are also what gets plotted, which incidentally fixes the unreadable
            # "-0" axis labels on the waterfall: raw 2nd-derivative values are around 1e-6 and
            # round to -0, whereas z-scores print as readable numbers.
            X_te = X_te_s

            vals = _pv_normalize_shap_values(vals)
            if vals.shape[0] != len(X_te) and vals.shape[1] == len(X_te):
                vals = vals.T
            shap_parts.append(vals)
            data_parts.append(X_te)
            # Record this fold's baseline (mean model output) for the waterfall plot below.
            try:
                _ev = explainer.expected_value                       # Scalar, list or array
                if isinstance(_ev, (list, np.ndarray)) and np.ndim(_ev) > 0:
                    _ev = np.asarray(_ev).ravel()[-1]                # Positive-class baseline
                base_parts.append(float(_ev))                        # Store as a plain float
            except Exception:
                base_parts.append(0.0)                               # Safe fallback if unavailable

        shap_values = np.vstack(shap_parts)
        shap_data = pd.concat(data_parts, axis=0)

        # FIGURE SPACING NOTE: do NOT pre-create a figure here and do NOT call plt.tight_layout()
        # afterwards. shap.summary_plot sizes the figure itself (set_size_inches, derived from
        # max_display) and positions its axes and colour bar by hand. Pre-creating a canvas means
        # shap's sizing fights a canvas of the wrong shape, and tight_layout cannot reposition
        # hand-placed axes - the result is a large empty band under the plot that bbox_inches
        # cannot crop, because the empty region lies INSIDE the axes bounding box.
        _n_disp_bee = min(20, len(feature_cols))            # Rows shap will actually draw
        # A FRESH FIGURE IS MANDATORY HERE. shap.summary_plot and shap.plots.waterfall have no
        # "figure" argument - they draw into whatever matplotlib currently considers the active
        # figure. Without an explicit new figure, the second shap call lands ON TOP of the first,
        # producing a waterfall overlaid on a beeswarm with two conflicting x-axes.
        plt.figure()                                         # Fresh, empty canvas for the beeswarm
        shap.summary_plot(shap_values, shap_data, feature_names=feature_cols,
                          show=False, max_display=_n_disp_bee)   # shap draws into that canvas
        fig_bee = plt.gcf()                                  # Take whichever figure shap just used
        fig_bee.set_size_inches(9.0, max(3.2, 0.32 * _n_disp_bee + 1.4))   # Height tracks row count
        fig_bee.axes[0].set_title(f"Out-of-fold SHAP beeswarm — {model_name}", fontsize=11)
        fig_bee.subplots_adjust(left=0.28, right=0.94, top=0.94, bottom=0.10)   # Explicit margins

        mean_abs = np.mean(np.abs(shap_values), axis=0)
        top_idx = np.argsort(mean_abs)[-20:][::-1]
        fig_imp, ax = plt.subplots(figsize=(9, 6.5))
        ax.barh([str(feature_cols[i]) for i in top_idx][::-1], mean_abs[top_idx][::-1])
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"Out-of-fold SHAP feature importance — {model_name}")
        fig_imp.tight_layout()

        # ---- Representative waterfall plot (explicitly requested on 9 Jan 2026) ----
        # The median-impact spectrum is chosen rather than the most extreme one, so the panel
        # shows a typical explanation rather than an outlier.
        _impact = np.abs(shap_values).sum(axis=1)                     # Total impact per spectrum
        _rep = int(np.argsort(_impact)[len(_impact) // 2])            # Median-impact spectrum
        _base = float(np.mean(base_parts)) if base_parts else 0.0     # Average baseline across folds
        _expl = shap.Explanation(values=shap_values[_rep],            # SHAP values for that spectrum
                                 data=np.asarray(shap_data.iloc[_rep].values, dtype=float),
                                 feature_names=[str(c) for c in feature_cols],
                                 base_values=_base)                   # Correct, non-zero baseline
        # Same spacing rule as the beeswarm above: let shap size its own canvas, then set explicit
        # margins instead of tight_layout, which cannot handle shap's hand-placed axes.
        _n_disp_wf = min(15, len(feature_cols))                       # Rows shap will draw
        plt.figure()                                                  # Fresh canvas - see above
        shap.plots.waterfall(_expl, max_display=_n_disp_wf, show=False)   # Draw into it
        fig_wf = plt.gcf()                                            # Take shap's figure
        fig_wf.set_size_inches(9.0, max(3.4, 0.38 * _n_disp_wf + 1.6))    # Height tracks row count
        fig_wf.axes[0].set_title(f"Representative out-of-fold SHAP waterfall — {model_name}",
                                 fontsize=11)                         # Title on shap's own axes
        fig_wf.subplots_adjust(left=0.30, right=0.94, top=0.92, bottom=0.12)   # Explicit margins

        _tag = model_name.replace(' ', '_')                           # Filename-safe model name
        return {f"SHAP_beeswarm_OOF_{_tag}": fig_bee,
                f"SHAP_feature_importance_OOF_{_tag}": fig_imp,
                f"SHAP_waterfall_OOF_{_tag}": fig_wf}

    def _pv_make_all_shap_figures(fig_data, model_names, seed, samples_per_animal=25):
        """Run the out-of-fold SHAP block for every requested model and merge the results."""
        if isinstance(model_names, str):        # Accept a single name for backward compatibility
            model_names = [model_names]         # Wrap it so the loop always works
        merged = {}                             # Collects the figures from every model
        for _mn in model_names:                 # One full out-of-fold SHAP pass per model
            merged.update(_pv_make_oof_shap_figures(fig_data, _mn, seed, samples_per_animal))
        return merged                           # Every SHAP figure, keyed by model name

    def _pv_build_figure_zip(figures, metadata_text):
        import zipfile
        out = io.BytesIO()
        with zipfile.ZipFile(out, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("README_methodology.txt", metadata_text)
            for name, fig in figures.items():
                zf.writestr(f"{name}.png", _pv_fig_png(fig))     # 300 dpi raster for drafts
                # Vector copy: fully editable in Illustrator/Inkscape, which is what the
                # professor meant by "editable files when possible" for the figures.
                _svg = io.BytesIO()                              # In-memory vector buffer
                fig.savefig(_svg, format="svg", bbox_inches="tight", facecolor="white")
                zf.writestr(f"{name}.svg", _svg.getvalue())       # Add the SVG next to the PNG
        return out.getvalue()

    _pv_uploaded = st.file_uploader(
        "Upload the Excel file with Sample IDs (Sample ID column)", type=["xlsx"], key="pv_uploader"
    )

    if _pv_uploaded is None:
        st.info("Upload the revised WT/OIM Excel file containing the Sample ID column to begin.")
    else:
        _pv_xls = pd.ExcelFile(_pv_uploaded)
        _pv_sheet_names = [s for s in _pv_xls.sheet_names if s in _SHEET_KIND]
        if not _pv_sheet_names:
            st.error(
                "None of the expected sheets (Raw, 2nd derivative, Quant peak ratios) were found. "
                f"Sheets found: {_pv_xls.sheet_names}"
            )
        else:
            _pv_preview_sheet = st.selectbox("Preview a sheet", _pv_sheet_names, index=0, key="pv_preview_sheet")
            _pv_preview_df = _pv_xls.parse(_pv_preview_sheet, nrows=5)
            st.dataframe(_pv_preview_df, use_container_width=True)

            _pv_ready = False
            try:
                _pv_g_col, _pv_l_col = _pv_find_group_label_columns(_pv_preview_df)
                st.success(f"Detected Sample-ID column **{_pv_g_col}** and label column **{_pv_l_col}** — ready for LOGO-CV.")
                _pv_ready = True
            except ValueError as e:
                st.error(str(e))

            if _pv_ready:
                st.sidebar.markdown("**Professor validation settings**")
                _pv_n_per_animal = int(st.sidebar.number_input("Spectra per Sample ID", 50, 2000, 250, step=50, key="pv_n"))
                _pv_seed = int(st.sidebar.number_input("Random seed", 0, 10_000, 42, step=1, key="pv_seed"))
                _pv_available_models = [m for m in _MODEL_ORDER if m != "XGBoost" or HAS_XGB]
                _pv_model_choices = st.sidebar.multiselect(
                    "Models", _pv_available_models, default=_pv_available_models, key="pv_models"
                )

                _pv_mode = st.radio(
                    "What do you want to run?",
                    ["Just the previewed sheet", "Full table (all sheets — reproduces the whole attached table)"],
                    horizontal=False, key="pv_mode"
                )
                st.caption(
                    "The full table runs Raw and 2nd derivative across all 3 regions plus the Quant peak "
                    "ratios — up to ~600 individual model fits. This can take several minutes."
                )

                if st.button("\U0001F680 Run validation", type="primary", key="pv_run"):
                    if not _pv_model_choices:
                        st.warning("Select at least one model.")
                    else:
                        _pv_sheets_to_run = [_pv_preview_sheet] if _pv_mode.startswith("Just") else _pv_sheet_names

                        _pv_all_results, _pv_counts_by_sheet = {}, {}
                        _pv_shortfalls_by_sheet, _pv_classes_by_sheet = {}, {}
                        _pv_progress = st.progress(0.0)
                        _pv_status = st.empty()
                        _pv_total_sheets = len(_pv_sheets_to_run)
                        _pv_failed = False

                        for _pv_s_idx, _pv_sheet_name in enumerate(_pv_sheets_to_run):
                            _pv_df = _pv_xls.parse(_pv_sheet_name)

                            def _pv_cb(frac, msg, _s=_pv_s_idx, _t=_pv_total_sheets):
                                _pv_progress.progress(min((_s + frac) / _t, 1.0))
                                _pv_status.write(f"Running: {msg}")

                            try:
                                _pv_results, _pv_counts, _pv_shortfalls, _pv_classes = _pv_evaluate_sheet(
                                    _pv_df, _pv_sheet_name, _pv_model_choices, _pv_n_per_animal, _pv_seed,
                                    progress_cb=_pv_cb
                                )
                            except ValueError as e:
                                st.error(f"{_pv_sheet_name}: {e}")
                                _pv_failed = True
                                break

                            for _region_key, _per_model in _pv_results.items():
                                _pv_all_results[(_pv_sheet_name, _region_key)] = _per_model
                            _pv_counts_by_sheet[_pv_sheet_name] = _pv_counts
                            _pv_shortfalls_by_sheet[_pv_sheet_name] = _pv_shortfalls
                            _pv_classes_by_sheet[_pv_sheet_name] = _pv_classes

                            if _pv_shortfalls:
                                st.warning(
                                    f"{_pv_sheet_name}: Sample ID groups with fewer than {_pv_n_per_animal} spectra "
                                    f"(used all available): {_pv_shortfalls}"
                                )

                        if not _pv_failed:
                            _pv_status.success("Done.")
                            _pv_progress.progress(1.0)

                            # Store everything needed to render results + downloads in session_state.
                            # Streamlit reruns the whole script on every widget click, including the
                            # download buttons below — without this, clicking one download button would
                            # make the "if st.button(Run validation)" block evaluate False again on the
                            # rerun, wiping the table and the other download button. Stashing the computed
                            # results here means the rendering section further below (which reads from
                            # session_state, not from this button block) survives that rerun.
                            # Stamp this exact run with the current date/time *before* building the
                            # export files, so the timestamp can be embedded both in the file names
                            # and inside the workbook contents themselves (belt-and-suspenders: even
                            # if a file gets renamed later, the in-sheet stamp still proves when it
                            # was generated). This is what lets a freshly-run file be told apart from
                            # an older one with the same base name already sitting in Downloads.
                            _pv_run_dt = datetime.now()
                            _pv_run_ts = _pv_run_dt.strftime("%Y%m%d_%H%M%S")
                            _pv_run_ts_display = _pv_run_dt.strftime("%Y-%m-%d %H:%M:%S")

                            st.session_state["pv_all_results"] = _pv_all_results
                            st.session_state["pv_model_choices"] = _pv_model_choices
                            st.session_state["pv_n_per_animal_used"] = _pv_n_per_animal
                            st.session_state["pv_seed_used"] = _pv_seed
                            st.session_state["pv_run_timestamp"] = _pv_run_ts
                            st.session_state["pv_run_timestamp_display"] = _pv_run_ts_display
                            st.session_state["pv_accuracy_xlsx"] = _pv_build_accuracy_table_excel(
                                _pv_all_results, run_timestamp=_pv_run_ts_display
                            )
                            st.session_state["pv_detailed_xlsx"] = _pv_build_detailed_workbook(
                                _pv_all_results, _pv_counts_by_sheet, _pv_shortfalls_by_sheet,
                                _pv_classes_by_sheet, _pv_n_per_animal, _pv_seed, run_timestamp=_pv_run_ts_display
                            )
                            st.session_state["pv_comparison_xlsx"] = _pv_build_comparison_workbook(
                                _pv_all_results, _pv_model_choices, _pv_n_per_animal, _pv_seed,
                                run_timestamp=_pv_run_ts_display
                            )
                            st.session_state["pv_has_results"] = True

            # ----------------------------------------------------------------
            # Render results + download buttons from session_state. This block
            # lives OUTSIDE "if st.button(Run validation)" on purpose, so that
            # clicking either download button (which triggers a Streamlit
            # rerun) just re-renders the same stored results instead of making
            # them disappear. The table, expander, and both download buttons
            # all stay on screen across as many clicks as you like.
            # ----------------------------------------------------------------
            if st.session_state.get("pv_has_results"):
                _pv_all_results = st.session_state["pv_all_results"]
                _pv_model_choices = st.session_state["pv_model_choices"]
                _pv_n_per_animal = st.session_state["pv_n_per_animal_used"]
                _pv_seed = st.session_state["pv_seed_used"]

                st.subheader("✅ Result table")
                st.caption(
                    f"Values are overall accuracy (pooled across all held-out Sample ID groups), using "
                    f"{_pv_n_per_animal} spectra/Sample ID and Leave-One-Group-Out CV, seed={_pv_seed}."
                )
                _pv_run_ts_display = st.session_state.get("pv_run_timestamp_display", "unknown")
                st.info(f"🕒 This run completed at **{_pv_run_ts_display}**. The 3 files below are stamped with this same "
                        f"time in their file names — if a file you're looking at in Downloads doesn't have this "
                        f"timestamp, it's an old one, not from this run.")

                _pv_display_rows = []
                for _model_name in _pv_model_choices:
                    _row = {"Model": _model_name}
                    for (_sheet_name, _region_key), _col_letter in _TEMPLATE_COLUMNS.items():
                        _res = _pv_all_results.get((_sheet_name, _region_key), {}).get(_model_name)
                        _label = f"{_sheet_name} | {_region_key}"
                        # Use np.nan (not None) for "not run in this pass" -- keeps the column a clean
                        # float64 dtype so na_rep="—" below actually applies. Python None mixed with
                        # floats leaves the column as dtype=object, and Styler then prints the literal
                        # string "None" instead of substituting na_rep.
                        _row[_label] = round(_res["overall_accuracy"], 4) if _res else np.nan
                    _pv_display_rows.append(_row)
                _pv_display_df = pd.DataFrame(_pv_display_rows)
                st.dataframe(_pv_display_df.style.format(precision=4, na_rep="—"), use_container_width=True)

                with st.expander("📌 Per-group (per-fold) accuracy", expanded=False):
                    for (_sheet_name, _region_key), _per_model in _pv_all_results.items():
                        for _model_name, _res in _per_model.items():
                            st.markdown(f"**{_sheet_name} — {_region_key} — {_model_name}**")
                            st.dataframe(
                                _res["fold_table"].style.format({"Fold accuracy": "{:.4f}"}),
                                use_container_width=True
                            )

                # File names carry the run timestamp (set above, e.g. 20260628_184409) so a
                # freshly-downloaded file is never visually indistinguishable from an older one
                # with the same base name already sitting in Downloads.
                _pv_run_ts = st.session_state.get("pv_run_timestamp", "unknown")

                _pv_col1, _pv_col2, _pv_col3 = st.columns(3)
                with _pv_col1:
                    st.download_button(
                        "📥 Download Accuracy_Table_Filled.xlsx", data=st.session_state["pv_accuracy_xlsx"],
                        file_name=f"Accuracy_Table_Filled_{_pv_run_ts}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="pv_dl1"
                    )
                with _pv_col2:
                    st.download_button(
                        "📥 Download Accuracy_Table_Detailed_Results.xlsx", data=st.session_state["pv_detailed_xlsx"],
                        file_name=f"Accuracy_Table_Detailed_Results_{_pv_run_ts}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="pv_dl2"
                    )
                with _pv_col3:
                    st.download_button(
                        "📥 Download Professor_Comparison_Tables.xlsx", data=st.session_state["pv_comparison_xlsx"],
                        file_name=f"Professor_Comparison_Tables_{_pv_run_ts}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="pv_dl3"
                    )

                st.success("Use the first file's numbers to update the manuscript table. The third file is the full Old-vs-New, per-group breakdown ready to share with your professor.")

                # ──────────────────────────────────────────────────────────────
                # Publication-ready figures using the exact same balanced data.
                # ──────────────────────────────────────────────────────────────
                st.markdown("---")
                # ---- Supplementary metrics table (professor's 9 Jan 2026 request) ----
                st.subheader("📋 Supplementary metrics — precision, recall and F1 for every model")
                st.caption(
                    "Pooled out-of-fold results from the leave-one-group-out folds, positive class = OIM. "
                    "This is the supplementary table requested alongside the main accuracy table."
                )
                _pv_prf_rows = []                                  # One row per (sheet, region, model)
                for (_s_name, _r_key), _models_res in _pv_all_results.items():
                    for _m_name, _m_res in _models_res.items():    # Every model evaluated on that combo
                        _pv_prf_rows.append({
                            "Sheet": _s_name,                      # Which spectral representation
                            "Feature region": "Peak ratios" if _r_key == "ratio" else _r_key,
                            "Model": _m_name,                      # Which classifier
                            "Accuracy": _m_res.get("overall_accuracy", float("nan")),
                            "Precision": _m_res.get("precision", float("nan")),
                            "Recall": _m_res.get("recall", float("nan")),
                            "F1-score": _m_res.get("f1", float("nan")),
                            "ROC-AUC": _m_res.get("roc_auc", float("nan")),
                            "Positive class": _m_res.get("positive_class", "OIM"),
                        })
                _pv_prf_df = pd.DataFrame(_pv_prf_rows)            # Assemble the supplementary table
                st.dataframe(_pv_prf_df.round(4), width="stretch") # Show it full width
                st.download_button(                                # Editable CSV copy for the paper
                    "📥 Download supplementary metrics (CSV)",
                    data=_pv_prf_df.to_csv(index=False).encode("utf-8"),
                    file_name="Supplementary_Precision_Recall_F1.csv",
                    mime="text/csv",
                    key="pv_download_prf_csv",
                )

                st.subheader("📊 Publication-ready diagrams from the balanced Sample-ID-grouped analysis")
                st.caption(
                    "These figures use the same 250 spectra per Sample ID and fixed seed as the accuracy table. "
                    "PCA and PLS-DA are descriptive projections of the balanced dataset; confusion matrix, "
                    "per-group accuracy, and SHAP are generated from Sample-ID-held-out LOGO evaluation."
                )


                st.markdown("### 🧩 12-plot PCA + PLS-DA regional comparison")
                st.caption(
                    "Generates Raw and 2nd-derivative Fingerprint, Collagen-rich and Mineral-rich regions: "
                    "6 PCA score plots + 6 PLS-DA score plots. Each row shows PCA and PLS-DA side by side. "
                    "All plots use the same 250 spectra per Sample ID and fixed seed. PCA accuracy is explicitly "
                    "PCA(PC1+PC2)+Logistic Regression under LOGO; PCA itself remains unsupervised."
                )
                if st.button("🧩 Generate 12-plot regional PCA + PLS-DA layout + editable Excel", key="pv_generate_12plot_layout"):
                    try:
                        with st.spinner("Generating 12 regional score plots and editable Excel charts..."):
                            _pv_12pkg = _pv_generate_12_plot_package(_pv_xls, _pv_n_per_animal, _pv_seed)
                            st.session_state["pv_12plot_website"] = _pv_12pkg["website"]
                            st.session_state["pv_12plot_master_xlsx"] = _pv_12pkg["master_xlsx"]
                            st.session_state["pv_12plot_region_xlsx"] = _pv_12pkg["region_xlsx"]
                            st.session_state["pv_12plot_timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
                        st.success("12-plot regional PCA + PLS-DA package generated.")
                    except Exception as _pv_12err:
                        st.error("Could not generate the 12-plot regional package.")
                        st.exception(_pv_12err)

                if st.session_state.get("pv_12plot_website"):
                    st.markdown("#### 12-plot layout on website")
                    for _item in st.session_state["pv_12plot_website"]:
                        st.markdown(f"##### {_item['sheet']} — {_item['region']}")
                        _cpa, _cpl = st.columns(2)
                        with _cpa:
                            st.image(_item["pca_png"], caption=f"PCA scores | PCA+LR LOGO accuracy = {_item['pca_accuracy']*100:.2f}%", width="stretch")
                        with _cpl:
                            st.image(_item["plsda_png"], caption=f"PLS-DA OOF scores | LOGO accuracy = {_item['plsda_accuracy']*100:.2f}%", width="stretch")
                        with st.expander(f"Full results — {_item['sheet']} — {_item['region']}", expanded=False):
                            if _item.get("comparison_png") is not None:
                                st.image(_item["comparison_png"], caption="PCA + PLS-DA scores side by side with accuracy", width="stretch")
                            _lc1, _lc2 = st.columns(2)
                            with _lc1:
                                if _item.get("pca_loadings_png") is not None:
                                    st.image(_item["pca_loadings_png"], caption="PCA PC1 loading plot with bone-band annotations", width="stretch")
                                st.markdown("**Most influential PCA wavenumbers (by |PC1 loading|)**")
                                st.dataframe(_item["pca_influential"], width="stretch")
                                st.markdown("**PCA variable importance (combined |PC1/PC2|)**")
                                st.dataframe(_item["pca_importance"], width="stretch")
                            with _lc2:
                                if _item.get("plsda_loadings_png") is not None:
                                    st.image(_item["plsda_loadings_png"], caption="PLS-DA LV1 loading plot with bone-band annotations", width="stretch")
                                st.markdown("**Most influential PLS-DA wavenumbers (by |loading|)**")
                                st.dataframe(_item["plsda_influential"], width="stretch")
                                st.markdown("**PLS-DA VIP with bond annotations**")
                                st.dataframe(_item["plsda_vip"], width="stretch")
                            if _item.get("cm_png") is not None:
                                st.image(_item["cm_png"], caption="PCA+LR and PLS-DA Sample-ID-held-out confusion matrices", width="stretch")

                    st.download_button(
                        "📥 Download master 12-plot editable Excel workbook",
                        data=st.session_state["pv_12plot_master_xlsx"],
                        file_name=f"Professor_12_Plot_PCA_PLSDA_Editable_{st.session_state.get('pv_12plot_timestamp','run')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="pv_download_12plot_master",
                    )
                    st.markdown("##### Individual editable Excel files by sheet/region")
                    _rcols = st.columns(3)
                    for _ri, (_rkey, _rbytes) in enumerate(st.session_state.get("pv_12plot_region_xlsx", {}).items()):
                        with _rcols[_ri % 3]:
                            st.download_button(
                                f"📥 {_rkey.replace('_',' ')}",
                                data=_rbytes,
                                file_name=f"{_rkey}_PCA_PLSDA_Editable_{st.session_state.get('pv_12plot_timestamp','run')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key=f"pv_download_region_{_rkey}",
                            )

                _pv_figure_combos = sorted(list(_pv_all_results.keys()), key=lambda x: (x[0], x[1]))
                _pv_combo_labels = {
                    combo: f"{combo[0]} — {combo[1] if combo[1] != 'ratio' else 'Peak ratios'}"
                    for combo in _pv_figure_combos
                }
                _pv_selected_combo = st.selectbox(
                    "Choose the sheet and feature region for figures",
                    _pv_figure_combos,
                    format_func=lambda x: _pv_combo_labels[x],
                    key="pv_figure_combo",
                )
                _pv_fig_sheet, _pv_fig_region = _pv_selected_combo
                _pv_models_for_combo = list(_pv_all_results[_pv_selected_combo].keys())
                _pv_fig_models = st.multiselect(
                    "Models for LOGO confusion matrices and per-group accuracy "
                    "(all selected by default, as the professor asked for all models)",
                    _pv_models_for_combo,
                    default=_pv_models_for_combo,      # Every model, matching the 9 Jan 2026 request
                    key="pv_figure_models",
                )
                if not _pv_fig_models:                 # Guard against an empty selection
                    _pv_fig_models = list(_pv_models_for_combo)   # Fall back to all models

                _pv_include_shap = st.checkbox(
                    "Generate out-of-fold SHAP diagrams (slower)",
                    value=False,
                    key="pv_include_shap_figures",
                    help="Each held-out Sample ID group is explained only by a model trained on the remaining Sample ID groups.",
                )
                _pv_shap_supported = [m for m in _pv_models_for_combo if m in ["Random Forest", "XGBoost", "Logistic Regression"]]
                _pv_shap_models = []       # Models to explain; filled in from the multiselect below
                _pv_shap_samples = 25      # Held-out spectra explained per Sample ID group, per model
                if _pv_include_shap:
                    if not _pv_shap_supported:
                        st.warning("Run Random Forest, XGBoost, or Logistic Regression to create out-of-fold SHAP figures.")
                    else:
                        _pv_shap_models = st.multiselect(
                            "Models for out-of-fold SHAP (beeswarm + representative waterfall)",
                            _pv_shap_supported,
                            default=_pv_shap_supported,   # Every model SHAP can practically explain
                            key="pv_shap_figure_models",
                            help="SVM (RBF) and KNN are excluded: they need KernelExplainer, which "
                                 "is not tractable on 501 spectral features. Worth stating in the "
                                 "figure caption rather than leaving unexplained.",
                        )
                        if not _pv_shap_models:            # Guard against an empty selection
                            _pv_shap_models = list(_pv_shap_supported)   # Fall back to all supported
                        _pv_shap_samples = int(st.number_input(
                            "Held-out spectra explained per Sample ID group",
                            min_value=5,
                            max_value=100,
                            value=25,
                            step=5,
                            key="pv_shap_samples_per_animal",
                        ))

                if st.button("🎨 Generate correct PCA, PLS-DA, validation and SHAP diagrams", type="primary", key="pv_generate_figures"):
                    try:
                        with st.spinner("Generating figures from the balanced 250-spectra-per-group dataset..."):
                            _pv_fig_data = _pv_prepare_figure_data(
                                _pv_xls,
                                _pv_fig_sheet,
                                _pv_fig_region,
                                _pv_n_per_animal,
                                _pv_seed,
                            )
                            _pv_figures = {}
                            _pv_figures.update(_pv_make_pca_figures(_pv_fig_data))
                            _pv_figures.update(_pv_make_plsda_figures(_pv_fig_data))

                            # Every selected Raw / 2nd derivative / Quant run now also gets an
                            # editable Excel workbook with PCA scores, PCA loadings, PCA+LR LOGO
                            # confusion matrix, PLS-DA OOF scores, PLS-DA VIP and PLS-DA confusion matrix.
                            _pv_region_editable_xlsx, _pv_region_export_data = _pv_editable_region_workbook(
                                _pv_fig_data, _pv_fig_sheet, _pv_fig_region, _pv_seed
                            )
                            st.session_state["pv_selected_region_editable_xlsx"] = _pv_region_editable_xlsx
                            st.session_state["pv_selected_region_pca_accuracy"] = _pv_region_export_data["pca_accuracy"]
                            st.session_state["pv_selected_region_plsda_accuracy"] = _pv_region_export_data["plsda_accuracy"]
                            st.session_state["pv_selected_region_export_data"] = _pv_region_export_data
                            # Add app_k.py-style interpretation/summary figures to the website and publication ZIP.
                            _pv_figures.update(
                                _pv_make_interpretation_figures(
                                    _pv_region_export_data, _pv_fig_sheet, _pv_fig_region
                                )
                            )

                            # STEP 1 requested by Professor: when the selected dataset is the
                            # quantitative-parameter / peak-ratio sheet, also create true Excel-editable
                            # PCA and PLS-DA score plots.  The PLS-DA workbook uses the same pooled
                            # Sample-ID-held-out LOGO accuracy that is printed directly on the PNG/SVG plot.
                            if _pv_fig_region == "ratio" or "quant" in str(_pv_fig_sheet).lower():
                                _pv_quant_editable = _pv_quantitative_editable_score_workbooks(_pv_fig_data, _pv_seed)
                                st.session_state["pv_quant_pca_editable_xlsx"] = _pv_quant_editable["pca_xlsx"]
                                st.session_state["pv_quant_plsda_editable_xlsx"] = _pv_quant_editable["plsda_xlsx"]
                                st.session_state["pv_quant_plsda_logo_accuracy"] = _pv_quant_editable["plsda_logo_accuracy"]
                            else:
                                st.session_state.pop("pv_quant_pca_editable_xlsx", None)
                                st.session_state.pop("pv_quant_plsda_editable_xlsx", None)
                                st.session_state.pop("pv_quant_plsda_logo_accuracy", None)

                            # Confusion matrices + per-group accuracy for EVERY selected model
                            _pv_figures.update(_pv_make_validation_figures(_pv_fig_data, _pv_fig_models, _pv_seed))

                            if _pv_include_shap and _pv_shap_models:
                                # Beeswarm + feature importance + representative waterfall, per model
                                _pv_figures.update(
                                    _pv_make_all_shap_figures(
                                        _pv_fig_data,
                                        _pv_shap_models,
                                        _pv_seed,
                                        samples_per_animal=_pv_shap_samples,
                                    )
                                )

                            _pv_pngs = {name: _pv_fig_png(fig) for name, fig in _pv_figures.items()}
                            for _fig in _pv_figures.values():
                                plt.close(_fig)

                            _pv_methodology = (
                                f"Sheet: {_pv_fig_sheet}\n"
                                f"Feature region: {_pv_fig_region}\n"
                                f"Spectra per Sample ID: {_pv_n_per_animal}\n"
                                f"Fixed random seed: {_pv_seed}\n"
                                f"Classification target: WT vs OIM\n"
                                f"Grouping variable: Sample ID\n"
                                f"Validation: Leave-One-Group-Out CV\n"
                                f"Confusion-matrix models: {', '.join(_pv_fig_models)}\n"
                                f"SHAP models: {', '.join(_pv_shap_models) if _pv_shap_models else 'not generated'}\n\n"
                                "PCA is an UNSUPERVISED projection fitted to the complete balanced dataset. It never uses the\n"
                                "WT/OIM labels, so there is no train/test leakage to guard against; this should be stated in\n"
                                "the figure caption.\n"
                                "PLS-DA is SUPERVISED, so its score plot is built from OUT-OF-FOLD projections: for each of the\n"
                                "each fold the model is fitted on all remaining Sample ID groups and only the held-out Sample ID group is projected into that\n"
                                "model's latent space. No Sample ID group contributes to the model that positions its own points.\n"
                                "VIP scores come from a separate full-data fit and are an interpretation statistic only.\n"
                                "Confusion matrices and per-group accuracy use pooled held-out predictions from all LOGO folds,\n"
                                "for every selected model.\n"
                                "SHAP is out-of-fold: every explained spectrum belongs to the held-out Sample ID group and its\n"
                                "explainer/model was trained only on the other Sample ID groups. SVM (RBF) and KNN are excluded\n"
                                "because KernelExplainer is not tractable on 501 spectral features.\n"
                                "Feature scaling is fitted INSIDE each fold on the training Sample ID groups only, for every model.\n"
                                "Each figure is exported as PNG (300 dpi) and SVG (vector, editable).\n"
                            )

                            st.session_state["pv_figure_pngs"] = _pv_pngs
                            st.session_state["pv_figure_zip"] = _pv_build_figure_zip(_pv_figures, _pv_methodology)
                            st.session_state["pv_figure_metadata"] = _pv_methodology
                            st.session_state["pv_figure_title"] = _pv_combo_labels[_pv_selected_combo]
                            st.session_state["pv_figure_timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
                        st.success("Publication diagrams generated successfully.")
                    except Exception as _pv_fig_error:
                        st.error("Could not generate the diagrams.")
                        st.exception(_pv_fig_error)

                if st.session_state.get("pv_figure_pngs"):
                    st.markdown(f"### Generated figures — {st.session_state.get('pv_figure_title', '')}")
                    _pv_caption_map = {
                        "PCA_scores": "PCA scores: WT/OIM distribution using the balanced dataset.",
                        "PCA_explained_variance": "Variance explained by the PCA components.",
                        "PCA_loadings": "PCA loadings identifying the spectral variables driving PC1/PC2.",
                        "PCA_LOGO_confusion_matrix": "PCA is unsupervised; this confusion matrix is from a separate animal-held-out Logistic Regression classifier using PC1+PC2 only.",
                        "PLSDA_scores": "PLS-DA out-of-fold score projection: each Sample ID group is positioned by a "
                                        "model fitted on the remaining Sample ID groups (Sample-ID-grouped). The accuracy printed "
                                        "on the plot is the pooled held-out LOGO classification accuracy.",
                        "LOGO_confusion_matrices_ALL_MODELS": "Pooled out-of-fold confusion matrices for every model.",
                        "Per_group_accuracy_ALL_MODELS": "Held-out accuracy per Sample ID group, all models side by side.",
                        "PLSDA_VIP": "PLS-DA VIP scores; VIP > 1 is commonly treated as influential.",
                        "PLSDA_LOGO_confusion_matrix": "PLS-DA pooled held-out WT/OIM predictions from all 12 Sample-ID-grouped LOGO folds.",
                        "PCA_PC1_loadings_annotated": "PCA PC1 loadings with bone FTIR band annotations; descriptive full-balanced fit.",
                        "PLSDA_LV1_loadings_annotated": "PLS-DA LV1 loadings with bone FTIR band annotations; full-balanced interpretation fit only.",
                        "PCA_vs_PLSDA_scores": "PCA and PLS-DA score plots side by side with their explicitly defined Sample-ID-held-out accuracies.",
                        "PCA_PLSDA_confusion_matrices": "PCA+LR and PLS-DA pooled Sample-ID-held-out confusion matrices shown side by side.",
                        "LOGO_confusion_matrix": "Pooled WT/OIM predictions from all 12 held-out-Sample-ID folds.",
                        "Per_group_accuracy": "Accuracy for each Sample ID group when that entire group was unseen during training.",
                        "SHAP_beeswarm_OOF": "Out-of-fold SHAP beeswarm: explanations for unseen-Sample-ID-group spectra only.",
                        "SHAP_feature_importance_OOF": "Mean absolute out-of-fold SHAP importance.",
                    }
                    for _name, _png in st.session_state["pv_figure_pngs"].items():
                        st.markdown(f"#### {_name.replace('_', ' ')}")
                        st.image(_png, caption=_pv_caption_map.get(_name, ""), width="stretch")
                        st.download_button(
                            f"📥 Download {_name}.png",
                            data=_png,
                            file_name=f"{_name}_{st.session_state.get('pv_figure_timestamp', 'run')}.png",
                            mime="image/png",
                            key=f"pv_download_{_name}",
                        )



                    # Editable Excel for the currently selected sheet/region (Raw, 2nd derivative, or Quant).
                    if st.session_state.get("pv_selected_region_editable_xlsx"):
                        st.markdown("### ✏️ Editable PCA + PLS-DA analysis workbook — current selection")
                        _pca_acc_sel = st.session_state.get("pv_selected_region_pca_accuracy")
                        _pls_acc_sel = st.session_state.get("pv_selected_region_plsda_accuracy")
                        if _pca_acc_sel is not None and _pls_acc_sel is not None:
                            st.info(
                                f"PCA+LR LOGO accuracy: **{_pca_acc_sel*100:.2f}%** | "
                                f"PLS-DA LOGO accuracy: **{_pls_acc_sel*100:.2f}%**"
                            )
                        st.download_button(
                            "📥 Download editable PCA + PLS-DA workbook (scores, loadings, VIP, confusion matrices)",
                            data=st.session_state["pv_selected_region_editable_xlsx"],
                            file_name=f"PCA_PLSDA_{st.session_state.get('pv_figure_title','selection').replace(' — ','_').replace(' ','_')}_{st.session_state.get('pv_figure_timestamp','run')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="pv_download_selected_region_editable",
                        )
                        _sel_d = st.session_state.get("pv_selected_region_export_data")
                        if _sel_d:
                            st.markdown("#### Extractable result tables — current selection")
                            _t1,_t2 = st.columns(2)
                            with _t1:
                                st.markdown("**PCA — most influential wavenumbers (by |PC1 loading|)**")
                                st.dataframe(_sel_d["pca_influential"].head(20), width="stretch")
                                st.markdown("**PCA — variable importance with bond annotations**")
                                st.dataframe(_sel_d["pca_importance"].head(20), width="stretch")
                                st.markdown("**PCA+LR LOGO confusion matrix — counts**")
                                st.dataframe(_sel_d["pca_confusion"], width="stretch")
                                st.markdown("**PCA+LR LOGO confusion matrix — %**")
                                st.dataframe(_sel_d["pca_confusion_pct"].round(2), width="stretch")
                            with _t2:
                                st.markdown("**PLS-DA — most influential wavenumbers (by |LV1 loading|)**")
                                st.dataframe(_sel_d["plsda_influential"].head(20), width="stretch")
                                st.markdown("**PLS-DA — VIP with bond annotations**")
                                st.dataframe(_sel_d["plsda_vip"].head(20), width="stretch")
                                st.markdown("**PLS-DA LOGO confusion matrix — counts**")
                                st.dataframe(_sel_d["plsda_confusion"], width="stretch")
                                st.markdown("**PLS-DA LOGO confusion matrix — %**")
                                st.dataframe(_sel_d["plsda_confusion_pct"].round(2), width="stretch")

                    # Excel-editable quantitative-parameter score plots requested by Professor.
                    # These buttons intentionally appear only for the Quant peak-ratio/quantitative run.
                    if st.session_state.get("pv_quant_pca_editable_xlsx") and st.session_state.get("pv_quant_plsda_editable_xlsx"):
                        st.markdown("### ✏️ Editable quantitative-parameter score plots (Excel)")
                        _pv_q_acc = st.session_state.get("pv_quant_plsda_logo_accuracy")
                        if _pv_q_acc is not None:
                            st.success(f"PLS-DA pooled Sample-ID-held-out LOGO accuracy shown on the figure: **{_pv_q_acc * 100:.2f}%**")
                        _qcol1, _qcol2 = st.columns(2)
                        with _qcol1:
                            st.download_button(
                                "📥 PCA scores — quantitative parameters (editable Excel)",
                                data=st.session_state["pv_quant_pca_editable_xlsx"],
                                file_name=f"PCA_Quantitative_Parameters_Editable_{st.session_state.get('pv_figure_timestamp', 'run')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="pv_download_quant_pca_editable",
                            )
                        with _qcol2:
                            st.download_button(
                                "📥 PLS-DA scores + LOGO accuracy — quantitative parameters (editable Excel)",
                                data=st.session_state["pv_quant_plsda_editable_xlsx"],
                                file_name=f"PLSDA_Quantitative_Parameters_Editable_{st.session_state.get('pv_figure_timestamp', 'run')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="pv_download_quant_plsda_editable",
                            )

                    st.download_button(
                        "📦 Download all publication diagrams (ZIP)",
                        data=st.session_state["pv_figure_zip"],
                        file_name=f"Professor_Publication_Figures_{st.session_state.get('pv_figure_timestamp', 'run')}.zip",
                        mime="application/zip",
                        key="pv_download_all_figures",
                    )
                    with st.expander("Figure methodology note", expanded=False):
                        st.code(st.session_state.get("pv_figure_metadata", ""))


# =============================================================================================
# ####################  SECTION: SPECTRAL ZONE-BASED SHAP  (NEW - APPENDED)  ###################
# =============================================================================================
# Self-contained section added at the end of the file. It defines its own helpers with a "_zs_"
# prefix so nothing above is touched, shadowed or altered. Every existing result - including the
# current per-wavenumber SHAP figures - behaves exactly as before.
#
# WHAT THIS ADDS
#   1. ZONE-BASED SHAP following
#        Contreras, Winterfeld, Popp & Bocklitz, "Spectral Zones-Based SHAP/LIME: Enhancing
#        Interpretability in Spectral Deep Learning Models Through Grouped Feature Analysis",
#        Analytical Chemistry 2024, 96(39), 15588-15597. DOI 10.1021/acs.analchem.4c02329
#      Two methods are offered:
#        Method A - aggregate ordinary per-wavenumber SHAP inside each zone (exact, reconciles
#                   perfectly with the SHAP figures already in the manuscript).
#        Method B - the paper's own approach: treat each zone as ONE feature and perturb the
#                   whole band at once. With ~10 zones all 2^10 coalitions are enumerated, so the
#                   Shapley values are EXACT rather than sampled.
#
#   2. FULLY EDITABLE SHAP EXPORTS, which is what Prof. Querido asked for on the call: the SHAP
#      values for EVERY wavenumber from 1800 to 800 as a spreadsheet, not a top-20 list. In
#      spectroscopy neighbouring channels are correlated, so a top-20 list is mostly the same band
#      repeated; the full-range table is what allows the values to be plotted against the spectrum.
#
# THE NINE ZONES - exactly as specified by Prof. Querido
#   Amide I 1700-1600 | Amide II 1600-1500 | CH 1490-1430 | CH/CO3 1430-1400 | CH 1350-1330
#   Amide III 1300-1200 | HPO4 1200-1100 | PO4 1100-930 | CO3 890-860
# =============================================================================================

import itertools as _zs_itertools          # Enumerates the zone coalitions for exact Shapley
import hashlib as _zs_hashlib              # Stable, process-independent hashing for the seeding
import math as _zs_math                    # Factorials for the Shapley weighting term
# openpyxl is imported under its own alias here. The code above this section only ever imports
# openpyxl's SUBMODULES (openpyxl.styles, openpyxl.utils, ...) from inside functions, so the bare
# name `openpyxl` is not guaranteed to exist at module level. Aliasing it keeps this section
# self-contained and immune to how the rest of the file happens to import things.
import openpyxl as _zs_openpyxl            # Streaming reader for the large source workbook
import zipfile as _zs_zipfile              # Packs the zone figures into a downloadable archive
# NOTE: openpyxl, zipfile and shap are all imported ELSEWHERE in this file only from inside
# conditional blocks, so their bare names are not guaranteed to exist when this section runs.
# Importing them here under _zs_ aliases makes this section independent of that.
# shap is deliberately NOT imported at module level: if the package is missing, a top-level
# import would abort the whole application and take every existing section down with it.
# It is imported lazily instead, so a missing shap only disables the features that need it.
def _zs_get_shap():
    """Import shap on first use and return it, with a clear message if it is unavailable."""
    try:
        import shap as _m                  # Only reached when a SHAP-based feature is requested
        return _m                          # Hand the module back to the caller
    except Exception as _e:                # Package missing or broken in this environment
        raise RuntimeError(
            "The 'shap' package is required for Method A and for the editable full-spectrum "
            "SHAP tables. Install it with `pip install shap`, or use Method B, which is pure "
            "numpy and needs no SHAP package."
        ) from _e

# ------------------------- START: Zone definitions and constants -----------------------------
# HIS listed order matters: 1600, 1430, 1200 and 1100 each appear in two zones, so a wavenumber
# is assigned to the FIRST zone in this list that contains it. Deterministic and documented.
_ZS_ZONES = [
    ("Amide I",        1600, 1700),   # Protein C=O stretch - the dominant collagen band
    ("Amide II",       1500, 1600),   # N-H bend / C-N stretch, also collagen
    ("CH (1490-1430)", 1430, 1490),   # CH2/CH3 deformation, organic matrix
    ("CH/CO3",         1400, 1430),   # Overlapping CH deformation and carbonate
    ("CH (1350-1330)", 1330, 1350),   # CH wag - a second, narrower CH window
    ("Amide III",      1200, 1300),   # C-N stretch / N-H bend, collagen secondary structure
    ("HPO4",           1100, 1200),   # Acid phosphate, immature-mineral marker
    ("PO4",             930, 1100),   # Phosphate v1/v3 - the main bone mineral band
    ("CO3",             860,  890),   # Carbonate v2, mineral substitution
]
# Channels in none of the nine zones (1800-1700, 1500-1490, 1400-1350, 1330-1300, 930-890,
# 860-800). This bucket MUST exist, otherwise the zone contributions would not sum back to the
# model output and the exactness check below could not be performed.
_ZS_UNASSIGNED = "Unassigned (gaps)"

_ZS_COLORS = {  # One colour per zone so mineral and organic bands are visually separable
    "Amide I": "#B4436C", "Amide II": "#D7263D", "CH (1490-1430)": "#F46036",
    "CH/CO3": "#E5A823", "CH (1350-1330)": "#8FB339", "Amide III": "#2E86AB",
    "HPO4": "#1B998B", "PO4": "#2660A4", "CO3": "#5B5F97", _ZS_UNASSIGNED: "#B0B0B0",
}
_ZS_WT = ["A103", "A97", "A235", "A247", "A249", "P4"]          # The six wild-type animals
_ZS_OIM = ["A226", "A227", "A228", "A230", "A250", "A260"]      # The six OIM animals
_ZS_ALL_IDS = _ZS_WT + _ZS_OIM                                   # Whitelist of valid animals
_ZS_CLASS_COLORS = {"WT": "#2E86AB", "OIM": "#D7263D"}           # Same palette as elsewhere
# -------------------------- END: Zone definitions and constants ------------------------------


# ---------------------------- START: Zone assignment helpers ---------------------------------
def _zs_wavenumber(col):
    """Return the numeric wavenumber for a column header, or None when it is not one."""
    try:                                    # Headers are normally plain numbers like 1800
        return float(str(col).strip())      # Successful conversion means a spectral channel
    except ValueError:                      # Ratio columns such as "1030/1670" land here
        return None                         # Signals "not a wavenumber"


def _zs_assign(feature_cols):
    """Map every feature column into exactly one zone. No column is dropped or double-counted."""
    members = {n: [] for n, _, _ in _ZS_ZONES}       # zone name -> list of column positions
    members[_ZS_UNASSIGNED] = []                     # Catch-all bucket for the gaps
    audit = []                                       # One row per column, for the workbook
    for i, col in enumerate(feature_cols):           # Walk the feature columns in order
        wn = _zs_wavenumber(col)                     # Numeric wavenumber, or None
        placed = _ZS_UNASSIGNED                      # Default until a zone claims it
        if wn is not None:                           # Only spectral channels can be zoned
            for name, lo, hi in _ZS_ZONES:           # FIRST match in his order wins
                if lo <= wn <= hi:                   # Inclusive both ends, as he wrote them
                    placed = name                    # Claim it
                    break                            # Resolves the shared endpoints
        members[placed].append(i)                    # Record the position
        audit.append({"Feature": str(col), "Wavenumber": wn, "Zone": placed})  # Audit row
    order = [n for n, _, _ in _ZS_ZONES] + [_ZS_UNASSIGNED]      # Stable display order
    names = [n for n in order if members[n]]                     # Drop empty zones
    idx = [np.asarray(members[n], dtype=int) for n in names]     # As numpy index arrays
    return names, idx, pd.DataFrame(audit)                       # Everything the caller needs


def _zs_zone_table(names, idx):
    """Small descriptive table of the partition, written into the workbook for documentation."""
    rng = {n: (lo, hi) for n, lo, hi in _ZS_ZONES}   # Look-up of each zone's printed range
    rows = []                                        # One row per zone
    for n, ii in zip(names, idx):                    # Walk in display order
        lo, hi = rng.get(n, (np.nan, np.nan))        # Unassigned has no single range
        rows.append({"Zone": n,
                     "Range (cm-1)": f"{int(lo)}-{int(hi)}" if not np.isnan(lo) else "gaps",
                     "Channels": len(ii)})           # Width of the zone in channels
    df = pd.DataFrame(rows)                          # Assemble
    df["% of spectrum"] = (df["Channels"] / df["Channels"].sum() * 100).round(1)  # Relative width
    return df                                        # Ready for Excel
# ----------------------------- END: Zone assignment helpers ----------------------------------


# ------------------------- START: Data loading, balancing, models ----------------------------
def _zs_read_sheet(path, sheet_name):
    """Stream one worksheet with openpyxl - the workbook is ~80 MB and pandas is memory-hungry."""
    wb = _zs_openpyxl.load_workbook(path, read_only=True, data_only=True)  # Streaming, values only
    try:                                              # try/finally releases the handle safely
        ws = wb[sheet_name]                           # Select the sheet
        rows = ws.iter_rows(values_only=True)         # Lazy iterator, constant memory per row
        header = [str(h).strip() for h in next(rows)] # First row holds the headers
        meta, feats = [], []                          # Text metadata and numeric block
        for r in rows:                                # One row at a time
            if r[0] is None:                          # Blank Sample ID = trailing empty row
                continue                              # Skip it
            meta.append(r[:2])                        # 'Sample ID' and 'Type'
            feats.append([np.nan if v is None else v for v in r[2:]])   # None -> NaN
        arr = np.asarray(feats, dtype=float)          # One contiguous float64 matrix
    finally:                                          # Always runs
        wb.close()                                    # Release the parser buffers
    df = pd.DataFrame(arr, columns=header[2:])        # Feature block with real headers
    df.insert(0, header[1], [m[1] for m in meta])     # Re-attach the class label
    df.insert(0, header[0], [m[0] for m in meta])     # Re-attach the animal ID
    return df                                         # Same shape pandas would give


def _zs_load_balanced(path, sheet, n_per_animal, seed):
    """Read a sheet, keep the 12 known animals, and sample exactly n_per_animal from each."""
    df = _zs_read_sheet(path, sheet)                                  # Low-memory read
    df.columns = [str(c).strip() for c in df.columns]                 # Tidy the headers
    idc, lbc = df.columns[0], df.columns[1]                           # ID and label columns
    df[idc] = df[idc].astype(str).str.strip()                         # Clean animal IDs
    df[lbc] = df[lbc].astype(str).str.strip().str.upper()             # Normalise WT / OIM
    df = df[df[idc].isin(_ZS_ALL_IDS)].copy()                         # Keep his 12 animals
    fc = [c for c in df.columns if c not in (idc, lbc)]               # Feature columns
    df[fc] = df[fc].apply(pd.to_numeric, errors="coerce")             # Text -> NaN
    df = df.dropna(subset=fc).reset_index(drop=True)                  # Drop incomplete spectra
    parts = []                                                        # Per-animal subsamples
    for animal, g in df.groupby(idc, sort=True):                      # Fixed iteration order
        if len(g) < n_per_animal:                                     # Guard the design
            raise ValueError(f"Animal {animal} has only {len(g)} spectra (< {n_per_animal}).")
        # REPRODUCIBILITY: Python's built-in hash() is SALTED PER PROCESS for strings, so
        # abs(hash(animal)) returns a different value on every launch of the app. Using it as a
        # random_state silently selected a different 250 spectra each run, which broke the fixed
        # -seed reproducibility the professor asked for on 1 June 2026. blake2b is a stable
        # digest: identical input always gives identical output, in every process, on every
        # machine, forever.
        rs = (seed + int.from_bytes(
            _zs_hashlib.blake2b(str(animal).encode("utf-8"), digest_size=4).digest(),
            "big")) % (2**32 - 1)                                     # Deterministic per animal
        parts.append(g.sample(n=n_per_animal, random_state=rs))       # Fixed-size subsample
    out = pd.concat(parts, ignore_index=True)                         # Stack the 12 subsamples
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)   # Shuffle rows
    X = out[fc].to_numpy(dtype=float)                                 # Feature matrix
    y = (out[lbc].to_numpy() == "OIM").astype(int)                    # OIM = 1, WT = 0
    groups = out[idc].to_numpy()                                      # Animal ID per spectrum
    return X, y, groups, fc                                           # Ready for analysis


def _zs_models(seed):
    """Models zone SHAP is computed for - all have exact, fast explainers."""
    m = {  # Same hyper-parameters as the main application, so results stay comparable
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=seed),
    }
    try:                                       # XGBoost may not be installed
        m["XGBoost"] = XGBClassifier(eval_metric="logloss", random_state=seed, n_jobs=-1)
    except Exception:                          # Skip silently if unavailable
        pass
    return m                                   # Consumed by both methods


def _zs_pos_proba(model, X):
    """Probability of the positive (OIM) class, with a fallback for margin-only models."""
    if hasattr(model, "predict_proba"):                     # Nearly all models here
        return model.predict_proba(X)[:, 1]                 # y is encoded so column 1 is OIM
    if hasattr(model, "decision_function"):                 # Squash a margin into (0, 1)
        return 1.0 / (1.0 + np.exp(-np.asarray(model.decision_function(X)).ravel()))
    return model.predict(X).astype(float)                   # Last resort
# -------------------------- END: Data loading, balancing, models -----------------------------


# ------------- START: METHOD A - per-wavenumber SHAP, plus the editable exports ---------------
def _zs_perwavenumber_shap(X, y, groups, model_proto, seed, cap):
    """Out-of-fold per-wavenumber SHAP. Feeds BOTH Method A and the full-spectrum Excel export.

    This is ordinary SHAP computed under the same animal-bound design as everything else: for
    each of the 12 folds the model is fitted on 11 animals and only the held-out animal is
    explained. Nothing about the existing SHAP figures changes - this is a separate computation
    whose purpose is to produce the editable full-range tables the professor asked for.
    """
    logo = LeaveOneGroupOut()                       # Animal-bound splitter
    rng = np.random.RandomState(seed)               # Reproducible subsampling
    sv_all, val_all, lab_all, ani_all = [], [], [], []   # Accumulators across folds
    for tr, te in logo.split(X, y, groups):         # One held-out animal per iteration
        sc = StandardScaler().fit(X[tr])            # CRITICAL: fitted on TRAINING animals only
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])   # Apply the same statistics
        mdl = clone(model_proto)                    # Fresh model per fold
        mdl.fit(Xtr, y[tr])                         # Train on the 11 training animals
        take = min(cap, Xte.shape[0])               # Cap explained spectra for runtime
        sel = rng.choice(Xte.shape[0], size=take, replace=False)   # Reproducible subset
        _sh = _zs_get_shap()                        # Lazy import - raises a clear error if absent
        if isinstance(mdl, LogisticRegression):     # Exact and fast for linear models
            sv = _sh.LinearExplainer(mdl, Xtr).shap_values(Xte[sel])
        else:                                       # Exact and fast for tree ensembles
            sv = _sh.TreeExplainer(mdl).shap_values(Xte[sel])
        sv = np.asarray(sv)                         # Normalise the shapes shap returns
        if sv.ndim == 3:                            # (samples, features, classes)
            sv = sv[:, :, 1]                        # Positive-class slice
        sv_all.append(sv)                           # This fold's SHAP matrix
        val_all.append(Xte[sel])                    # Matching scaled feature values
        lab_all.append(y[te][sel])                  # Matching true labels
        ani_all.append(groups[te][sel])             # Matching animal IDs
    return (np.vstack(sv_all), np.vstack(val_all),
            np.concatenate(lab_all), np.concatenate(ani_all))


def _zs_editable_tables(sv, vals, labels, animals, feat_cols, zone_names, zone_idx):
    """Build the EDITABLE spreadsheets: SHAP for every wavenumber, not a top-20 list.

    On the call the professor's objection to the current SHAP export was precise: a ranked list
    of 20 wavenumbers is mostly the same band repeated, because adjacent channels in a spectrum
    are strongly correlated. What he needs is the value at EVERY wavenumber so it can be plotted
    across the spectrum and overlaid with the mean spectra. These tables are exactly that.
    """
    wn = np.array([_zs_wavenumber(c) for c in feat_cols], dtype=float)   # Wavenumber axis
    zone_of = {}                                                        # column index -> zone
    for name, ii in zip(zone_names, zone_idx):                          # Invert the mapping
        for i in ii:                                                    # Every column in the zone
            zone_of[i] = name                                           # Tag it

    # ---- Table 1: full-spectrum importance, one row per wavenumber ----
    full = pd.DataFrame({
        "Wavenumber (cm-1)": wn,                                        # The x-axis for his plot
        "Zone": [zone_of.get(i, _ZS_UNASSIGNED) for i in range(len(feat_cols))],
        "Mean |SHAP|": np.abs(sv).mean(axis=0),                         # Global importance
        "Mean SHAP (signed)": sv.mean(axis=0),                          # Direction of the effect
        "Mean SHAP - WT spectra": sv[labels == 0].mean(axis=0),         # Direction within WT
        "Mean SHAP - OIM spectra": sv[labels == 1].mean(axis=0),        # Direction within OIM
        "Std |SHAP|": np.abs(sv).std(axis=0),                           # Spread across spectra
    }).sort_values("Wavenumber (cm-1)", ascending=False)                # 1800 -> 800 convention

    # ---- Table 2: representative waterfall AS VALUES, not only a picture ----
    impact = np.abs(sv).sum(axis=1)                                     # Total impact per spectrum
    rep = int(np.argsort(impact)[len(impact) // 2])                     # Median-impact spectrum
    waterfall = pd.DataFrame({
        "Wavenumber (cm-1)": wn,                                        # Every channel, in order
        "Zone": [zone_of.get(i, _ZS_UNASSIGNED) for i in range(len(feat_cols))],
        "SHAP value": sv[rep],                                          # Signed contribution
        "Feature value (scaled)": vals[rep],                            # What the model saw
    }).sort_values("Wavenumber (cm-1)", ascending=False)                # Same axis convention
    meta = pd.DataFrame({"Item": ["Representative spectrum", "Animal", "True class",
                                  "Selection rule"],
                         "Value": [rep, animals[rep],
                                   "OIM" if labels[rep] == 1 else "WT",
                                   "median total |SHAP| - a typical spectrum, not an outlier"]})

    # ---- Table 3: per-spectrum matrix, wavenumbers as columns ----
    per_spec = pd.DataFrame(sv, columns=[str(c) for c in feat_cols])    # One row per spectrum
    per_spec.insert(0, "True class", np.where(labels == 1, "OIM", "WT"))   # Context column
    per_spec.insert(0, "Animal", animals)                               # Context column

    return full, waterfall, meta, per_spec                              # Four editable sheets
# -------------- END: METHOD A - per-wavenumber SHAP, plus the editable exports ----------------


# -------------- START: METHOD B - exact zone-perturbation Shapley (the paper) -----------------
def _zs_shapley_weights(n):
    """Pre-compute |S|!(n-|S|-1)!/n! for every coalition size."""
    w = np.zeros(n)                                          # One weight per coalition size
    for s in range(n):                                       # s = zones already in the coalition
        w[s] = _zs_math.factorial(s) * _zs_math.factorial(n - s - 1) / _zs_math.factorial(n)
    return w                                                 # Used by the enumeration below


def _zs_exact_shapley(model, x, background, zone_idx, masks, weights):
    """EXACT Shapley values over zones, enumerating every coalition. The paper's method.

    A zone is "removed" by replacing ALL of its wavenumbers at once with the training-set mean -
    a chemically plausible spectrum, rather than a spike of zeros. With ~10 zones there are only
    2^10 = 1024 coalitions, so the exact values are enumerated: no sampling, no approximation.

    Efficiency axiom: sum(phi) == f(x) - f(reference), which the caller checks numerically.
    """
    n = len(zone_idx)                                        # Number of zone super-features
    variants = np.repeat(background[None, :], masks.shape[0], axis=0)   # Start from the reference
    for j, ii in enumerate(zone_idx):                        # For each zone
        present = masks[:, j].astype(bool)                   # Coalitions including this zone
        variants[np.ix_(present, ii)] = x[ii]                # Restore the real values there
    f = _zs_pos_proba(model, variants)                       # ONE batched prediction call
    powers = (1 << np.arange(n))                             # Bit weights for packing
    keys = masks @ powers                                    # Integer id of each coalition
    fs = f[np.argsort(keys)]                                 # Predictions indexed by bitmask
    phi = np.zeros(n)                                        # Shapley value per zone
    ids = np.arange(masks.shape[0])                          # Every bitmask
    for j in range(n):                                       # Accumulate zone j's contributions
        bit = 1 << j                                         # Bit for zone j
        without = ids[(ids & bit) == 0]                      # Coalitions excluding zone j
        sizes = np.array([bin(s).count("1") for s in without])   # |S| for each
        phi[j] = np.sum(weights[sizes] * (fs[without | bit] - fs[without]))   # Weighted sum
    return phi, float(fs[-1]), float(fs[0])                  # phi, f(all present), f(reference)


def _zs_zone_perturbation(X, y, groups, zone_idx, model_proto, seed, cap):
    """Run the exact zone-perturbation Shapley calculation out-of-fold, animal by animal."""
    n = len(zone_idx)                                        # Number of zones
    masks = np.array(list(_zs_itertools.product([0, 1], repeat=n)), dtype=np.int8)  # Coalitions
    weights = _zs_shapley_weights(n)                         # Shapley weights by size
    logo = LeaveOneGroupOut()                                # Animal-bound splitter
    rng = np.random.RandomState(seed)                        # Reproducible subsampling
    phis, labs, anis, checks = [], [], [], []                # Accumulators
    for tr, te in logo.split(X, y, groups):                  # One held-out animal per iteration
        sc = StandardScaler().fit(X[tr])                     # Fitted on TRAINING animals only
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])  # Scale both sides
        mdl = clone(model_proto)                             # Fresh model
        mdl.fit(Xtr, y[tr])                                  # Train on 11 animals
        bg = Xtr.mean(axis=0)                                # Reference = training mean spectrum
        take = min(cap, Xte.shape[0])                        # Method B is ~1000x heavier
        for i in rng.choice(Xte.shape[0], size=take, replace=False):   # Reproducible subset
            phi, f_full, f_ref = _zs_exact_shapley(mdl, Xte[i], bg, zone_idx, masks, weights)
            phis.append(phi)                                 # Zone contributions
            labs.append(y[te][i])                            # True class
            anis.append(groups[te][i])                       # Animal
            checks.append({"Animal": groups[te][i], "sum(phi)": float(phi.sum()),
                           "f(x) - f(reference)": f_full - f_ref,
                           "abs error": abs(float(phi.sum()) - (f_full - f_ref))})
    return np.vstack(phis), np.asarray(labs), np.asarray(anis), pd.DataFrame(checks)
# --------------- END: METHOD B - exact zone-perturbation Shapley (the paper) ------------------


# --------------------------------- START: Zone figures ---------------------------------------
def _zs_figure_guide():
    """Static, plain-language guide to every figure this section produces.

    Written as a table rather than generated by a language model on purpose: this text goes into
    a manuscript workbook, so it has to be stable, checkable and identical between runs. A
    generated narrative would vary run to run and could state something the numbers do not
    support - which is the last thing you want in a supplementary file a reviewer reads.
    """
    rows = [
        ("ZoneMap",
         "Which part of the spectrum each of the nine zones covers.",
         "Orientation only - no results. Read it first so the zone names in the other figures "
         "map onto real wavenumbers.",
         "That the zones you expect are present, and how much of 1800-800 falls outside them."),
        ("ZoneImportance",
         "Mean |contribution| of each zone to the OIM/WT prediction, pooled over all explained "
         "spectra.",
         "Longer bar = the model relies on that band more. Values are absolute, so they show "
         "strength but not direction.",
         "Which chemical band drives the classification. For OIM - a type I collagen mutation - "
         "the protein bands (Amide I/II) leading is biologically the expected result."),
        ("ZoneImportance_perWavenumber",
         "The same importance divided by the number of channels in each zone.",
         "Corrects for zone width. A wide band accumulates contribution simply by being wide; "
         "this asks how informative each band is per unit of spectrum.",
         "Zones that rise sharply between this and the previous figure. A narrow band with high "
         "density (e.g. CO3, 16 channels) is punching above its width and may be a real finding."),
        ("ZoneBeeswarm",
         "Signed contribution of every zone for every explained spectrum, split by true class.",
         "Each dot is one spectrum. Left of zero pushes the prediction towards WT, right towards "
         "OIM. Blue = WT spectra, red = OIM spectra.",
         "Whether a zone pushes the two classes in OPPOSITE directions. A band that is important "
         "but pushes both classes the same way is not discriminating - it is a baseline offset."),
        ("ZoneWaterfall",
         "How one representative spectrum's prediction is built up, zone by zone.",
         "Starts at the reference prediction and each bar adds that zone's contribution until "
         "the model's final P(OIM) is reached. Red adds towards OIM, blue towards WT.",
         "The spectrum shown is the MEDIAN-impact one, deliberately typical rather than extreme. "
         "It shows the mechanism for a single sample, not an average."),
        ("SHAP_FullSpectrum_Overlay",
         "Mean |SHAP| plotted against wavenumber across the whole fingerprint region, with the "
         "zones shaded behind it.",
         "Read it like a spectrum: peaks mark influential regions. The shading shows which zone "
         "each peak belongs to.",
         "This is the figure that replaces a top-20 wavenumber list. Adjacent channels are "
         "correlated, so a ranked list repeats one band many times; this shows the bands."),
    ]
    return pd.DataFrame(rows, columns=["Figure", "What it shows", "How to read it",
                                       "What to look for"])


def _zs_interpretation(per_zone, labels, method_label, model_name, n_explained,
                       _zs_n_per_animal_used=250):
    """Interpretation of THIS run, derived arithmetically from the computed numbers.

    Every statement below is a direct readout of the table - a ratio, a rank or a sign. Nothing
    is inferred or written by a model, so the sheet cannot disagree with the data it describes.
    """
    d = per_zone.reset_index(drop=True)                       # Already sorted by importance
    real = d[d["Zone"] != _ZS_UNASSIGNED].reset_index(drop=True)   # Ignore the gap bucket
    top, second = real.iloc[0], real.iloc[1]                  # Two most influential zones
    dens = real.sort_values("Mean |contribution| per wavenumber", ascending=False).iloc[0]
    unassigned = d[d["Zone"] == _ZS_UNASSIGNED]               # The gaps bucket, if present
    tot = float(d["Mean |contribution|"].sum())               # Denominator for the shares

    rows = [("Model / method", f"{model_name} - {method_label}"),
            ("Training design",
             f"{_zs_n_per_animal_used} spectra per animal x 12 animals = "
             f"{_zs_n_per_animal_used * 12} balanced spectra; each fold trains on "
             f"{_zs_n_per_animal_used * 11} spectra from 11 animals and tests on the 12th "
             f"(the 1 June 2026 animal-bound design)"),
            ("Spectra explained", f"{n_explained} "
             f"({int((labels == 1).sum())} OIM, {int((labels == 0).sum())} WT), out-of-fold. "
             f"This is an explanation subset only - it does not affect any model or accuracy"),
            ("Most influential zone",
             f"{top['Zone']} at {top['Mean |contribution|']:.4f}, "
             f"{top['Mean |contribution|'] / max(second['Mean |contribution|'], 1e-12):.2f}x the "
             f"next zone ({second['Zone']}) and "
             f"{100 * top['Mean |contribution|'] / max(tot, 1e-12):.1f}% of total contribution"),
            ("Direction of the top zone",
             f"{top['Zone']} averages {top['Mean signed - WT']:+.4f} on WT spectra and "
             f"{top['Mean signed - OIM']:+.4f} on OIM spectra - "
             + ("it pushes the two classes in OPPOSITE directions, which is what a genuinely "
                "discriminating band looks like."
                if top['Mean signed - WT'] * top['Mean signed - OIM'] < 0 else
                "it pushes both classes the SAME way, so it acts more as an offset than as a "
                "discriminator. Worth checking against the beeswarm.")),
            ("Most information-dense zone",
             f"{dens['Zone']} at {dens['Mean |contribution| per wavenumber']:.5f} per wavenumber "
             f"across only {int(dens['Channels'])} channels"
             + ("" if dens['Zone'] == top['Zone'] else
                f" - note this is NOT the top zone by raw total ({top['Zone']}), so "
                f"{dens['Zone']} is contributing more than its width would suggest.")),
            ("Zone ranking",
             " > ".join(f"{r['Zone']} ({r['Mean |contribution|']:.4f})"
                        for _, r in real.head(5).iterrows()))]
    if len(unassigned):                                       # Only when gaps exist
        u = unassigned.iloc[0]
        rows.append(("Outside the nine zones",
                     f"{int(u['Channels'])} channels ({100 * u['Channels'] / d['Channels'].sum():.0f}% "
                     f"of the spectrum) fall in no zone and carry "
                     f"{u['Mean |contribution|']:.4f} "
                     f"({100 * u['Mean |contribution|'] / max(tot, 1e-12):.1f}% of total). "
                     f"If this share is large, the zone definitions may need widening."))
    rows.append(("Caveat",
                 "Contributions describe how THIS model uses the spectrum. They are evidence "
                 "about the model's behaviour, not direct measurement of tissue chemistry."))
    return pd.DataFrame(rows, columns=["Item", "Finding from this run"])


def _zs_embed_charts_on_data_sheets(wb, safe_name):
    """Anchor charts DIRECTLY on the existing data sheets, not only on separate CHART sheets.

    The separate '<Model> CHART ...' sheets are easy to miss in a 50-sheet workbook - you have to
    scroll the tab bar to find them. Putting a chart on the same sheet as its numbers means the
    graph is visible the moment that sheet is opened, which is what 'embedded' should mean in
    practice. The charts reference the cells already written by to_excel, so they stay live: edit
    a value and the graph follows.
    """
    from openpyxl.chart import BarChart, LineChart, Reference   # Imported lazily

    # ---- Zone summary sheet: three charts down the right-hand side ----
    zs_name = f"{safe_name} zones"[:31]                          # Sheet written earlier by to_excel
    if zs_name in wb.sheetnames:
        ws = wb[zs_name]                                         # Row 1 = header, rows 2..n = zones
        last = ws.max_row                                        # Last populated row
        if last >= 3:                                            # Need at least two zones to plot
            # Column layout from the DataFrame: A Zone | B Channels | C Mean |contribution|
            # D per wavenumber | E signed | F signed WT | G signed OIM
            c1 = BarChart(); c1.type = "bar"; c1.style = 10      # Headline importance
            c1.title = f"Zone importance - {safe_name}"
            c1.y_axis.title = "Mean |contribution|"; c1.x_axis.title = "Spectral zone"
            c1.add_data(Reference(ws, min_col=3, min_row=1, max_row=last), titles_from_data=True)
            c1.set_categories(Reference(ws, min_col=1, min_row=2, max_row=last))
            c1.height, c1.width = 9, 17
            ws.add_chart(c1, "J2")                               # To the right of the data

            c2 = BarChart(); c2.type = "bar"; c2.style = 12      # Width-normalised
            c2.title = f"Zone importance per wavenumber - {safe_name}"
            c2.y_axis.title = "Mean |contribution| per wavenumber"; c2.x_axis.title = "Spectral zone"
            c2.add_data(Reference(ws, min_col=4, min_row=1, max_row=last), titles_from_data=True)
            c2.set_categories(Reference(ws, min_col=1, min_row=2, max_row=last))
            c2.height, c2.width = 9, 17
            ws.add_chart(c2, "J20")                              # Below the first

            c3 = BarChart(); c3.type = "bar"; c3.grouping = "clustered"; c3.style = 10
            c3.title = f"Direction by class - {safe_name}"       # WT vs OIM, the beeswarm's message
            c3.y_axis.title = "Mean signed contribution"; c3.x_axis.title = "Spectral zone"
            c3.add_data(Reference(ws, min_col=6, max_col=7, min_row=1, max_row=last),
                        titles_from_data=True)
            c3.set_categories(Reference(ws, min_col=1, min_row=2, max_row=last))
            c3.height, c3.width = 9, 17
            ws.add_chart(c3, "J38")                              # Below the second

    # ---- Full-spectrum sheet: the SHAP curve across 1800-800 ----
    fr_name = f"{safe_name} SHAP full range"[:31]                # Written earlier by to_excel
    if fr_name in wb.sheetnames:
        ws2 = wb[fr_name]                                        # A Wavenumber | B Zone | C Mean |SHAP|
        last2 = ws2.max_row
        if last2 >= 3:
            c4 = LineChart(); c4.style = 2
            c4.title = f"Mean |SHAP| across 1800-800 cm-1 - {safe_name}"
            c4.x_axis.title = "Wavenumber (cm-1)"; c4.y_axis.title = "Mean |SHAP|"
            c4.add_data(Reference(ws2, min_col=3, min_row=1, max_row=last2), titles_from_data=True)
            c4.set_categories(Reference(ws2, min_col=1, min_row=2, max_row=last2))
            c4.height, c4.width = 10, 26                         # Wide, like a spectral panel
            for ser in c4.series:                                # Plain line, no markers
                ser.smooth = False
            ws2.add_chart(c4, "J2")                              # Right of the 501 data rows


def _zs_add_excel_charts(wb, model_name, per_zone, full_tbl, waterfall_zone_values, zone_names):
    """Write NATIVE Excel charts into the workbook, with their data on the same sheet.

    These are real Excel chart objects, not pictures: the professor can open the workbook, click a
    chart, and change fonts, colours, axis titles and series - and the chart stays linked to the
    numbers beside it, so editing a value updates the graph. This is what he asked for when he
    said he needed the figures in a form he could adjust for a journal figure panel.

    The SVG exports remain the better choice for the final manuscript figure (vector, publication
    resolution). These charts are for HIS editing workflow inside Excel.
    """
    from openpyxl.chart import BarChart, LineChart, Reference   # Imported lazily, only when needed
    from openpyxl.styles import Font                            # For the small headings

    safe = model_name.replace(" ", "")[:14]                     # Excel caps sheet names at 31

    # ---------- Chart 1 + 2: zone importance, raw total and per wavenumber ----------
    ws = wb.create_sheet(f"{safe} CHART zones"[:31])            # One sheet per chart group
    ws["A1"] = f"Zone importance - {model_name}"                # Human-readable title
    ws["A1"].font = Font(bold=True, size=12)                    # Make the heading stand out
    ws["A2"] = ("Edit any number below and both charts update. Charts are native Excel objects: "
                "right-click to change fonts, colours or axis titles.")
    hdr = ["Zone", "Mean |contribution|", "Mean |contribution| per wavenumber"]
    for c, h in enumerate(hdr, start=1):                        # Write the header row
        ws.cell(row=4, column=c, value=h).font = Font(bold=True)
    pz = per_zone.reset_index(drop=True)                        # Already sorted by importance
    for r in range(len(pz)):                                    # Then the data rows
        ws.cell(row=5 + r, column=1, value=str(pz.loc[r, "Zone"]))
        ws.cell(row=5 + r, column=2, value=float(pz.loc[r, "Mean |contribution|"]))
        ws.cell(row=5 + r, column=3,
                value=float(pz.loc[r, "Mean |contribution| per wavenumber"]))
    last = 4 + len(pz)                                          # Last data row

    ch1 = BarChart(); ch1.type = "bar"; ch1.style = 10          # Horizontal bars, like the figure
    ch1.title = f"Zone importance - {model_name}"               # Editable title
    ch1.x_axis.title = "Spectral zone"; ch1.y_axis.title = "Mean |contribution|"
    ch1.add_data(Reference(ws, min_col=2, min_row=4, max_row=last), titles_from_data=True)
    ch1.set_categories(Reference(ws, min_col=1, min_row=5, max_row=last))
    ch1.height, ch1.width = 10, 18                              # Roughly a half-page panel
    ws.add_chart(ch1, "E4")                                     # Anchor it beside the data

    ch2 = BarChart(); ch2.type = "bar"; ch2.style = 12          # Width-normalised version
    ch2.title = f"Zone importance per wavenumber - {model_name}"
    ch2.x_axis.title = "Spectral zone"; ch2.y_axis.title = "Mean |contribution| per wavenumber"
    ch2.add_data(Reference(ws, min_col=3, min_row=4, max_row=last), titles_from_data=True)
    ch2.set_categories(Reference(ws, min_col=1, min_row=5, max_row=last))
    ch2.height, ch2.width = 10, 18
    ws.add_chart(ch2, "E24")                                    # Below the first chart

    # ---------- Chart 3: signed direction, WT vs OIM (the beeswarm's message) ----------
    ws2 = wb.create_sheet(f"{safe} CHART direction"[:31])
    ws2["A1"] = f"Direction of each zone by true class - {model_name}"
    ws2["A1"].font = Font(bold=True, size=12)
    ws2["A2"] = ("Bars on opposite sides of zero mean the zone genuinely separates the two "
                 "groups. Same side means it shifts both predictions the same way.")
    for c, h in enumerate(["Zone", "Mean signed - WT", "Mean signed - OIM"], start=1):
        ws2.cell(row=4, column=c, value=h).font = Font(bold=True)
    for r in range(len(pz)):
        ws2.cell(row=5 + r, column=1, value=str(pz.loc[r, "Zone"]))
        ws2.cell(row=5 + r, column=2, value=float(pz.loc[r, "Mean signed - WT"]))
        ws2.cell(row=5 + r, column=3, value=float(pz.loc[r, "Mean signed - OIM"]))
    ch3 = BarChart(); ch3.type = "bar"; ch3.grouping = "clustered"; ch3.style = 10
    ch3.title = f"Zone contribution by class - {model_name}"
    ch3.x_axis.title = "Spectral zone"
    ch3.y_axis.title = "Mean signed contribution (negative = WT, positive = OIM)"
    ch3.add_data(Reference(ws2, min_col=2, max_col=3, min_row=4, max_row=4 + len(pz)),
                 titles_from_data=True)
    ch3.set_categories(Reference(ws2, min_col=1, min_row=5, max_row=4 + len(pz)))
    ch3.height, ch3.width = 11, 19
    ws2.add_chart(ch3, "E4")

    # ---------- Chart 4: full-spectrum SHAP curve, 1800 -> 800 ----------
    if full_tbl is not None and len(full_tbl):
        ft = full_tbl.dropna(subset=["Wavenumber (cm-1)"]).sort_values(
            "Wavenumber (cm-1)", ascending=False).reset_index(drop=True)
        ws3 = wb.create_sheet(f"{safe} CHART spectrum"[:31])
        ws3["A1"] = f"Mean |SHAP| across the spectrum - {model_name}"
        ws3["A1"].font = Font(bold=True, size=12)
        ws3["A2"] = ("The full fingerprint region, one row per wavenumber. This is the curve that "
                     "replaces a ranked top-20 list.")
        for c, h in enumerate(["Wavenumber (cm-1)", "Mean |SHAP|", "Zone"], start=1):
            ws3.cell(row=4, column=c, value=h).font = Font(bold=True)
        for r in range(len(ft)):                                # 501 rows - Excel handles this fine
            ws3.cell(row=5 + r, column=1, value=float(ft.loc[r, "Wavenumber (cm-1)"]))
            ws3.cell(row=5 + r, column=2, value=float(ft.loc[r, "Mean |SHAP|"]))
            ws3.cell(row=5 + r, column=3, value=str(ft.loc[r, "Zone"]))
        lastr = 4 + len(ft)
        ch4 = LineChart(); ch4.style = 2                        # A plain line, like the figure
        ch4.title = f"Mean |SHAP| across 1800-800 cm-1 - {model_name}"
        ch4.x_axis.title = "Wavenumber (cm-1)"; ch4.y_axis.title = "Mean |SHAP|"
        ch4.add_data(Reference(ws3, min_col=2, min_row=4, max_row=lastr), titles_from_data=True)
        ch4.set_categories(Reference(ws3, min_col=1, min_row=5, max_row=lastr))
        ch4.height, ch4.width = 10, 26                          # Wide, to match a spectral panel
        for ser in ch4.series:                                  # Thin line, no markers
            ser.smooth = False
        ws3.add_chart(ch4, "E4")

    # ---------- Chart 5: representative waterfall, as a bar of zone contributions ----------
    if waterfall_zone_values is not None:
        ws4 = wb.create_sheet(f"{safe} CHART waterfall"[:31])
        ws4["A1"] = f"Representative spectrum - zone contributions - {model_name}"
        ws4["A1"].font = Font(bold=True, size=12)
        ws4["A2"] = ("One typical (median-impact) spectrum, decomposed by zone. Positive pushes "
                     "the prediction towards OIM, negative towards WT.")
        for c, h in enumerate(["Zone", "Contribution"], start=1):
            ws4.cell(row=4, column=c, value=h).font = Font(bold=True)
        order = np.argsort(np.abs(waterfall_zone_values))[::-1]    # Largest first
        for r, j in enumerate(order):
            ws4.cell(row=5 + r, column=1, value=str(zone_names[j]))
            ws4.cell(row=5 + r, column=2, value=float(waterfall_zone_values[j]))
        ch5 = BarChart(); ch5.type = "bar"; ch5.style = 11
        ch5.title = f"Representative spectrum, zone contributions - {model_name}"
        ch5.x_axis.title = "Spectral zone"; ch5.y_axis.title = "Contribution to P(OIM)"
        ch5.add_data(Reference(ws4, min_col=2, min_row=4, max_row=4 + len(order)),
                     titles_from_data=True)
        ch5.set_categories(Reference(ws4, min_col=1, min_row=5, max_row=4 + len(order)))
        ch5.height, ch5.width = 10, 18
        ws4.add_chart(ch5, "E4")


def _zs_what_is_shap():
    """Introductory sheet: what SHAP is, what it does and does not tell you, and why it was used.

    Placed first in the workbook so a reader meets the concept before any figure. Kept factual and
    conservative - in particular the 'what SHAP does NOT indicate' section, because the most common
    misreading of an explainability result is to treat it as a causal or statistical claim.
    """
    rows = [
        ("What SHAP is",
         "SHAP (SHapley Additive exPlanations) is an explainable-AI method that quantifies why a "
         "model made a particular prediction. It comes from cooperative game theory: each feature "
         "is treated as a 'player' contributing to an outcome, and the Shapley value is that "
         "player's mathematically fair share of the result. Unlike ordinary feature-importance "
         "rankings, SHAP gives a contribution for EVERY feature on EVERY individual prediction."),

        ("Why it is needed here",
         "Random Forest, XGBoost and Logistic Regression can classify spectra accurately, but they "
         "do not say WHY a spectrum was called WT or OIM. A model that is accurate for the wrong "
         "reason - an instrument artefact, a batch effect, a quirk of one animal - looks identical "
         "to one that is accurate for the right reason, until you open it up. SHAP opens it up."),

        ("What the 'players' are in THIS study",
         "This is the key design choice. In ordinary SHAP each of the 501 individual wavenumbers "
         "is a player. In the zone-based method used here (Method B) each of the 10 SPECTRAL "
         "ZONES is a single player, so contributions are attributed to biochemical bands - Amide "
         "I, PO4, CO3 - rather than to individual wavenumbers that a spectroscopist cannot "
         "interpret in isolation."),

        ("How to read a SHAP value",
         "POSITIVE pushes the prediction towards OIM. NEGATIVE pushes it towards WT. A LARGE "
         "ABSOLUTE value means strong influence; NEAR ZERO means the feature barely mattered for "
         "that prediction. Values are additive: they sum to the difference between this "
         "prediction and the model's average prediction."),

        ("Global versus local interpretation",
         "GLOBAL - which regions matter across all spectra (zone importance, beeswarm, "
         "full-spectrum overlay). LOCAL - why one particular spectrum was classified as it was "
         "(waterfall). Both are provided, because a region can be important on average while "
         "acting differently on an individual sample."),

        ("Out-of-fold explanation",
         "Every explanation in this workbook comes from a model that never saw that animal during "
         "training, under leave-one-animal-out cross-validation. This matters: explanations from "
         "a model that has memorised an animal describe the memorisation, not the biology."),

        ("Which mathematical guarantees are checked",
         "Shapley values satisfy four axioms: efficiency (contributions sum to prediction minus "
         "baseline), symmetry, dummy (an unused feature receives exactly zero) and additivity. "
         "Because Method B enumerates all 2^10 coalitions rather than sampling them, the values "
         "are exact, and the efficiency identity is verified numerically for every explained "
         "spectrum - see the 'exactness' sheets, where the error is around 1e-16."),

        ("What SHAP DOES tell you here",
         "Which biochemical regions of the FTIR spectrum the trained model relies on to separate "
         "WT from OIM bone, how strongly, in which direction, and whether a region pushes the two "
         "groups apart or merely shifts every prediction the same way."),

        ("What SHAP does NOT tell you",
         "It does NOT test statistical significance, does NOT prove causation, does NOT measure "
         "biochemical concentrations, and does NOT by itself identify a disease mechanism. It "
         "describes the MODEL's behaviour. When the regions the model relies on coincide with the "
         "known biology - as they do here, with collagen bands leading in a collagen disease - "
         "that agreement is meaningful supporting evidence, but it is an interpretation placed on "
         "the result, not something SHAP established."),

        ("Why SHAP rather than other methods",
         "Permutation importance, Gini importance and regression coefficients give only an overall "
         "ranking, are model-specific, and cannot explain an individual prediction. SHAP provides "
         "a fair attribution with proven axioms, works consistently across all three model types "
         "used here so their explanations are directly comparable, and supports both global and "
         "local interpretation."),

        ("What it contributes to this manuscript",
         "It moves the study from 'the models classify WT and OIM' to 'the models classify WT and "
         "OIM using the collagen bands'. Accuracy alone (0.87-0.90 under animal-bound validation) "
         "says the task is learnable; SHAP says what is being learned, and links it to the "
         "biochemistry of the tissue."),

        ("How to read this workbook",
         "Start with 'plain summary' for the headline result in non-technical language, then "
         "'Method A vs Method B' for how zone importance is computed, 'Zone chemistry' for what "
         "each band is, and 'figure notes' for what each figure shows. 'Glossary' defines every "
         "term. The numeric sheets - zones, per-spectrum, SHAP full range, waterfall - hold the "
         "underlying values, all editable."),
    ]
    return pd.DataFrame(rows, columns=["Topic", "Explanation"])


def _zs_method_comparison():
    """Full explanation of Method A vs Method B - what each computes and when to use which.

    Written out because the distinction is the methodological core of this section and a reviewer
    will ask about it. The channel counts quoted are the real ones from the professor's zone
    definitions on the 1800-800 cm-1 grid at 2 cm-1 spacing.
    """
    rows = [
        ("The problem both methods solve",
         "Ordinary SHAP assigns a contribution to each of the 501 individual wavenumbers. But "
         "neighbouring channels in a spectrum are not independent: the ~51 channels between 1700 "
         "and 1600 cm-1 all measure the same Amide I vibration. Reporting them as 51 separate "
         "features produces a ranked list in which the same chemical band appears many times, "
         "which is why a top-20 wavenumber table is uninformative in spectroscopy. Both methods "
         "answer the more useful question: which BIOCHEMICAL REGION does the model rely on?"),

        ("Method A - Zone-aggregated SHAP: what it does",
         "Compute ordinary per-wavenumber SHAP, then add up the values inside each zone. "
         "Example: Amide I = SHAP(1700) + SHAP(1698) + ... + SHAP(1600). 501 numbers become 10. "
         "Because SHAP values are additive by construction, this sum is exact - it is a "
         "re-grouping of the same numbers, not a new estimate."),

        ("Method A - what it measures",
         "'How much total attribution sits inside this zone?' It reconciles perfectly with the "
         "per-wavenumber SHAP figures, so the two views can never contradict each other."),

        ("Method A - the theoretical weakness",
         "Wide zones accumulate more simply by containing more channels. In these zone "
         "definitions PO4 spans 85 channels while CO3 spans 16 - a 5.3x difference in width "
         "before any chemistry is considered. If every channel contributed equally, PO4 would "
         "score 5.3x CO3 while being no more informative. This is why the "
         "'per wavenumber' figure exists: it divides by channel count and removes the effect."),

        ("Method B - Zone perturbation: what it does",
         "The approach proposed by Contreras et al. Treat each zone as ONE feature. To measure "
         "Amide I, replace ALL of its wavenumbers at once with the training-set mean spectrum and "
         "see how far the prediction moves. Repeat for every possible combination of zones "
         "present or absent - with 10 zones that is 2^10 = 1024 combinations - and average the "
         "marginal effect using the Shapley weighting. Because every coalition is enumerated, the "
         "result is EXACT, not sampled."),

        ("Method B - what it measures",
         "'How much does the model actually depend on this band?' If a zone's information is also "
         "available elsewhere in the spectrum, removing it changes the prediction very little, "
         "and Method B correctly assigns it a small value. Method A cannot detect that redundancy, "
         "because it only adds up attributions without ever removing anything."),

        ("Method B - why it suits spectroscopy",
         "Perturbing a single wavenumber asks the model a question that cannot physically occur: "
         "no real sample has one channel change while its neighbours stay fixed. Perturbing a "
         "whole band is a question a spectroscopist can answer - what if this vibration were "
         "absent? It is also immune to zone width, since the whole band moves together."),

        ("Method B - the cost",
         "Roughly 1024 model evaluations per explained spectrum instead of one, so it is about "
         "three orders of magnitude slower. This is why the 'spectra explained per animal' "
         "setting is kept small for Method B and larger for Method A."),

        ("Do they disagree in THIS dataset?",
         "Barely. Comparing the two methods on the same runs gives Spearman rank correlations of "
         "0.976 to 1.000 across all three models and both preprocessing routes (raw and second "
         "derivative), with identical top-three zones in five of six comparisons. So the "
         "width-bias concern is real in principle but is NOT materially distorting these results. "
         "That agreement is itself a useful robustness check: two methods sharing no code reach "
         "the same biological conclusion."),

        ("Which to report",
         "MAIN MANUSCRIPT: Method B. It is the published, spectroscopy-specific approach, it is "
         "exact rather than sampled, and the Shapley efficiency identity is verified numerically "
         "(see the exactness sheets). SUPPLEMENTARY: Method A, as a sensitivity analysis showing "
         "the conclusions do not depend on which attribution method is used."),

        ("One-line summary",
         "Method A asks how much attribution LIES INSIDE a band; Method B asks how much the model "
         "would LOSE WITHOUT that band. Method B is the stronger question for correlated spectral "
         "data, and in this dataset both give the same answer."),
    ]
    return pd.DataFrame(rows, columns=["Topic", "Explanation"])


def _zs_zone_chemistry():
    """What each spectral zone IS, chemically and biologically, in plain language.

    Included so that someone reading the workbook without a spectroscopy background can tell what
    'Amide II' actually refers to and why it matters for this particular disease model.
    """
    rows = [
        ("Amide I", "1700-1600", "Protein backbone, C=O (carbonyl) stretch",
         "The strongest collagen signal in bone. Collagen is the flexible protein scaffold that "
         "gives bone its toughness - the rope, as opposed to the mineral chalk.",
         "DIRECTLY relevant: OIM is a type I collagen mutation, so this band reports on the "
         "defective molecule itself."),
        ("Amide II", "1600-1500", "Protein backbone, N-H bend combined with C-N stretch",
         "A second, independent signature of the same collagen protein. Sensitive to how the "
         "protein chains are folded and hydrogen-bonded.",
         "DIRECTLY relevant: changes here indicate altered collagen structure, not just amount."),
        ("CH (1490-1430)", "1490-1430", "CH2 / CH3 bending in organic side chains",
         "Generic organic-matrix signal from the non-collagen protein and lipid content.",
         "Indirect: reports on the organic matrix broadly rather than collagen specifically."),
        ("CH/CO3", "1430-1400", "Overlapping CH bending and carbonate (CO3) vibration",
         "A mixed band where organic and mineral signals sit on top of each other.",
         "Ambiguous by nature - a change here could come from either matrix or mineral, so it is "
         "hard to interpret on its own."),
        ("CH (1350-1330)", "1350-1330", "CH wagging vibration",
         "A narrow secondary organic-matrix band.",
         "Minor: rarely a primary marker in bone studies."),
        ("Amide III", "1300-1200", "C-N stretch with N-H bend",
         "A third collagen signature, particularly sensitive to the secondary structure - whether "
         "the collagen triple helix is properly formed.",
         "Relevant: often used to assess collagen maturity and cross-linking."),
        ("HPO4", "1200-1100", "Acid phosphate (HPO4) vibration",
         "Marks immature, newly deposited mineral that has not yet fully crystallised.",
         "Relevant: OIM bone often shows altered mineral maturity, so this band can indicate "
         "abnormal mineralisation dynamics."),
        ("PO4", "1100-930", "Phosphate v1/v3 stretching",
         "The dominant bone MINERAL band - hydroxyapatite, the hard mineral phase that provides "
         "stiffness.",
         "Secondary: in OIM the mineral abnormality is a downstream consequence of the collagen "
         "defect, because mineral is deposited on a faulty collagen template."),
        ("CO3", "890-860", "Carbonate (CO3) v2 vibration",
         "Carbonate that has substituted into the mineral crystal in place of phosphate. Higher "
         "substitution generally means less mature, more soluble mineral.",
         "Relevant: carbonate substitution is a recognised marker of bone quality, and this is a "
         "narrow band, so its per-wavenumber intensity matters more than its raw total."),
        (_ZS_UNASSIGNED, "gaps", "Regions outside the nine defined zones",
         "Includes 1800-1700, 1500-1490, 1400-1350, 1330-1300, 930-890 and 860-800 cm-1.",
         "Kept explicitly so that the zone contributions still sum to the model's total output. A "
         "large value here would suggest the zone definitions need widening."),
    ]
    return pd.DataFrame(rows, columns=["Zone", "Range (cm-1)", "Chemical assignment",
                                       "What it means in plain language",
                                       "Relevance to OIM vs WT"])


def _zs_glossary():
    """Every column and term used anywhere in this workbook, explained without jargon."""
    rows = [
        ("Wavenumber (cm-1)", "The position along the infrared spectrum - effectively 'which "
         "colour of infrared light'. Different chemical bonds absorb at different positions, so "
         "the wavenumber tells you which molecule is responsible. Runs 1800 down to 800 by "
         "spectroscopy convention."),
        ("Zone", "A named band of wavenumbers grouped together because they come from the same "
         "chemical bond. See the 'Zone chemistry' sheet."),
        ("Channels", "How many individual wavenumber measurements fall inside a zone. Wider zones "
         "have more channels, which matters when comparing totals between zones."),
        ("SHAP value", "How much one feature pushed the model's prediction for one spectrum, and "
         "in which direction. Positive pushes towards OIM, negative towards WT. SHAP values are "
         "additive: they sum to the difference between this prediction and the average one."),
        ("Mean |contribution| / Mean |SHAP|", "Average size of the push, ignoring direction. A "
         "measure of how much the model RELIES on that zone. Bigger = more influential."),
        ("Mean signed contribution", "Average push WITH direction kept. Near zero can mean the "
         "zone is unimportant, OR that it pushes different spectra opposite ways and they cancel."),
        ("Mean signed - WT / - OIM", "The average push computed separately within each class. If "
         "these have OPPOSITE signs, the zone genuinely separates the two groups. If they have "
         "the same sign, the zone shifts every prediction the same way and does not discriminate."),
        ("Mean |contribution| per wavenumber", "Importance divided by the number of channels. "
         "Removes the advantage wide zones get simply by covering more spectrum."),
        ("Zone perturbation (Method B)", "The published approach (Contreras et al. 2024). Each "
         "zone is treated as ONE feature and the whole band is replaced at once by an average "
         "spectrum, measuring what the model loses without that band. More realistic than altering "
         "single wavenumbers, because neighbouring wavenumbers never vary independently in a real "
         "sample."),
        ("Zone-aggregated (Method A)", "Ordinary per-wavenumber SHAP, summed within each zone. "
         "Exact and consistent with the per-wavenumber figures, but favours wide zones."),
        ("Exactness check", "A mathematical guarantee test. Shapley values must satisfy "
         "sum(contributions) = prediction - reference. An error near 1e-16 means these are exact "
         "values, not statistical estimates."),
        ("Out-of-fold", "Every explanation comes from a model that never saw that animal during "
         "training. Prevents the model from simply recognising an animal it has memorised."),
        ("Leave-One-Group-Out (LOGO)", "The validation scheme: train on 11 animals, test on the "
         "12th, repeat 12 times so every animal is held out exactly once. The grouping variable is "
         "Animal ID, so no animal's spectra ever appear in both training and testing."),
        ("Reference / baseline", "The average training spectrum. A zone is 'removed' by replacing "
         "it with this, which keeps every perturbed spectrum chemically plausible."),
        ("WT / OIM", "WT = wild type, the healthy control mice. OIM = osteogenesis imperfecta "
         "murine, a genetic model of brittle bone disease caused by a type I collagen mutation."),
    ]
    return pd.DataFrame(rows, columns=["Term", "Plain-language explanation"])


def _zs_plain_summary(per_zone, labels, model_name, n_explained, n_per_animal):
    """The whole result written out for a reader with no spectroscopy or ML background."""
    d = per_zone.reset_index(drop=True)
    real = d[d["Zone"] != _ZS_UNASSIGNED].reset_index(drop=True)
    top, second = real.iloc[0], real.iloc[1]
    tot = float(d["Mean |contribution|"].sum())
    collagen = real[real["Zone"].isin(["Amide I", "Amide II", "Amide III"])]
    mineral = real[real["Zone"].isin(["PO4", "HPO4", "CO3"])]
    col_share = 100 * collagen["Mean |contribution|"].sum() / max(tot, 1e-12)
    min_share = 100 * mineral["Mean |contribution|"].sum() / max(tot, 1e-12)
    disc = real[real["Mean signed - WT"] * real["Mean signed - OIM"] < 0]

    rows = [
        ("What was the question?",
         "Can a computer tell a bone sample from a mouse with brittle-bone disease (OIM) apart "
         "from a healthy mouse (WT), using only an infrared scan - and if so, WHICH part of the "
         "chemistry is it using?"),
        ("How was it tested?",
         f"{n_per_animal} spectra were taken from each of 12 mice (6 healthy, 6 diseased). The "
         f"model was trained on 11 mice and tested on the 12th, repeated so every mouse was held "
         f"out once. This matters: it proves the model recognises the DISEASE, not the individual "
         f"mouse. {n_explained} held-out spectra were then explained in detail."),
        ("What is being measured?",
         "For each region of the infrared spectrum, how much the model's decision depended on it. "
         "A large value means the model leaned on that chemistry heavily."),
        ("The headline result",
         f"{top['Zone']} matters most - {top['Mean |contribution|']:.4f}, which is "
         f"{top['Mean |contribution|'] / max(second['Mean |contribution|'], 1e-12):.1f} times the "
         f"next region ({second['Zone']}) and about {100 * top['Mean |contribution|'] / max(tot, 1e-12):.0f}% "
         f"of everything the model used."),
        ("Collagen versus mineral",
         f"Collagen regions (Amide I, II, III) account for {col_share:.0f}% of the model's "
         f"attention; mineral regions (PO4, HPO4, CO3) account for {min_share:.0f}%. "
         + ("Collagen dominates." if col_share > min_share else "Mineral dominates.")),
        ("Why that is the RIGHT answer",
         "OIM is caused by a mutation in type I collagen - the protein scaffold of bone. The "
         "model was given no biological hints, only numbers. It taught itself to look at the "
         "collagen bands, which is precisely where the disease lives. Had it keyed on something "
         "chemically meaningless, that would suggest it had latched onto a measurement artefact."),
        ("Which regions actually separate the two groups?",
         ", ".join(disc["Zone"].tolist()) + ". In these regions the model pushes healthy and "
         "diseased samples in OPPOSITE directions, which is what real discrimination looks like. "
         "A region can be influential without separating the groups, so this is a separate check."),
        ("What this does NOT prove",
         "These numbers describe how the MODEL uses the spectrum. They are strong evidence that "
         "the disease signal lives in collagen, but they are not a direct chemical measurement of "
         "the tissue. Only 12 animals were available, so individual values carry wide uncertainty "
         "- the RANKING of regions is reliable, the exact decimals are not."),
        ("One-sentence version",
         f"Using only infrared scans, the model separates diseased from healthy bone, and it does "
         f"so mainly by reading the {top['Zone']} region - a collagen signal - which matches the "
         f"known cause of the disease."),
    ]
    return pd.DataFrame(rows, columns=["Question", f"Answer ({model_name})"])


def _zs_figure_descriptions(per_zone, phi, labels, animals, zone_names, channels,
                            full_tbl, model_name, method_label):
    """One complete description PER FIGURE, combining the method with THIS run's actual numbers.

    Every "What this result shows" sentence is computed from the data being described - a rank, a
    ratio, a sign or a wavenumber lookup. Nothing is written by a language model, so a description
    can never contradict the figure it sits next to. That matters because these captions go into a
    supplementary file a reviewer reads alongside the figure.
    """
    d = per_zone.reset_index(drop=True)                       # Already ordered by importance
    real = d[d["Zone"] != _ZS_UNASSIGNED].reset_index(drop=True)      # Ignore the gaps bucket
    top, second = real.iloc[0], real.iloc[1]                  # Two strongest zones
    tot = float(d["Mean |contribution|"].sum())               # Denominator for shares
    dens = real.sort_values("Mean |contribution| per wavenumber", ascending=False).iloc[0]
    unas = d[d["Zone"] == _ZS_UNASSIGNED]                     # Gaps row, if present
    ch_tot = int(d["Channels"].sum())                         # Total channels in the spectrum

    # Which zones actually SEPARATE the classes: WT and OIM means must have opposite signs.
    sep = real[real["Mean signed - WT"] * real["Mean signed - OIM"] < 0]
    nonsep = real[(real["Mean signed - WT"] * real["Mean signed - OIM"] >= 0) &
                  (real["Mean |contribution|"] > 0.1 * top["Mean |contribution|"])]

    # Representative spectrum used by the waterfall - same rule as the figure itself.
    rep = int(np.argsort(np.abs(phi).sum(axis=1))[len(phi) // 2])
    rep_cls = "OIM" if labels[rep] == 1 else "WT"             # Its true class
    rep_top = zone_names[int(np.argmax(np.abs(phi[rep])))]    # Its biggest single contributor
    rep_val = float(phi[rep][int(np.argmax(np.abs(phi[rep])))])   # And that contribution

    # Peak of the full-spectrum curve, and how concentrated the importance is.
    ft = full_tbl.dropna(subset=["Wavenumber (cm-1)"]) if full_tbl is not None else None
    if ft is not None and len(ft):
        pk = ft.loc[ft["Mean |SHAP|"].idxmax()]                # Single most important wavenumber
        srt = ft.sort_values("Mean |SHAP|", ascending=False)   # Ranked channels
        n10 = max(1, int(0.10 * len(ft)))                      # Top 10% of channels
        conc = 100 * srt.head(n10)["Mean |SHAP|"].sum() / max(srt["Mean |SHAP|"].sum(), 1e-12)
        peak_txt = (f"The single most influential wavenumber is {pk['Wavenumber (cm-1)']:.0f} cm-1 "
                    f"(zone: {pk['Zone']}). The most important 10% of channels carry {conc:.0f}% "
                    f"of the total importance, so the signal is "
                    f"{'highly concentrated in a few bands' if conc > 50 else 'spread fairly broadly across the spectrum'}.")
    else:
        peak_txt = "Full-spectrum table not generated for this run."

    rows = [
        ("ZoneMap",
         "Reference map showing which wavenumbers belong to each of the nine zones.",
         "Orientation only - it carries no result. Read it first so the zone names in every other "
         "figure map onto real wavenumbers. Unshaded stretches belong to no zone.",
         f"The nine zones cover {ch_tot - (int(unas['Channels'].iloc[0]) if len(unas) else 0)} "
         f"of {ch_tot} channels. "
         + (f"{int(unas['Channels'].iloc[0])} channels "
            f"({100 * int(unas['Channels'].iloc[0]) / ch_tot:.0f}% of the fingerprint region) fall "
            f"between the defined zones and are grouped as '{_ZS_UNASSIGNED}'."
            if len(unas) else "Every channel falls inside a defined zone."),
         "Zones sharing an endpoint (1600, 1430, 1200, 1100 cm-1) are assigned to the first zone "
         "in the professor's listed order, so each channel belongs to exactly one zone."),

        ("ZoneImportance",
         f"Mean |contribution| of each spectral zone to the OIM-vs-WT prediction, for {model_name}, "
         f"computed by {method_label}, pooled over all explained spectra.",
         "Longer bar = the model relies on that band more. Values are absolute, so the bars show "
         "strength but not direction - use the beeswarm for direction.",
         f"{top['Zone']} is the dominant zone at {top['Mean |contribution|']:.4f}, "
         f"{top['Mean |contribution|'] / max(second['Mean |contribution|'], 1e-12):.2f}x the next "
         f"zone ({second['Zone']} at {second['Mean |contribution|']:.4f}) and "
         f"{100 * top['Mean |contribution|'] / max(tot, 1e-12):.0f}% of all contribution. "
         f"Full order: " + " > ".join(real.head(4)["Zone"].tolist()) + ".",
         "OIM is a type I collagen mutation, so protein bands (Amide I/II) leading is the "
         "biologically expected result. Report the ranking, not the third decimal: with 12 animals "
         "the individual values carry roughly +/-40% confidence intervals."),

        ("ZoneImportance_perWavenumber",
         "The same importance divided by the number of channels in each zone.",
         "Corrects for zone width. A wide band accumulates contribution simply by covering more "
         "spectrum; this asks how informative each band is per unit of spectrum.",
         f"{dens['Zone']} is the most information-dense zone at "
         f"{dens['Mean |contribution| per wavenumber']:.5f} per wavenumber across only "
         f"{int(dens['Channels'])} channels"
         + ("" if dens['Zone'] == top['Zone'] else
            f", even though {top['Zone']} leads on the raw total. {dens['Zone']} therefore "
            f"contributes more than its width would suggest.") + ".",
         "Compare this figure with the previous one. Zones that climb sharply between the two are "
         "narrow but informative - a genuine finding that raw totals hide."),

        ("ZoneBeeswarm",
         "Signed contribution of every zone for every explained spectrum, split by true class.",
         "Each dot is one spectrum. Left of zero pushes the prediction towards WT, right towards "
         "OIM. Blue = WT spectra, red = OIM spectra.",
         (f"{len(sep)} zone(s) push the two classes in OPPOSITE directions, listed strongest "
          f"first: "
          + "; ".join(f"{r['Zone']} (WT {r['Mean signed - WT']:+.4f}, OIM {r['Mean signed - OIM']:+.4f})"
                      for _, r in sep.iterrows())          # ALL of them, not just the first three
          + ". These are the genuinely discriminating bands. Note that the size of the separation "
            "matters as well as its existence - a zone can have opposite signs that are both very "
            "close to zero, which is technically discriminating but practically negligible."
          + (f" By contrast {', '.join(nonsep['Zone'].tolist())} push both classes the same way, "
             f"acting as an offset rather than a discriminator." if len(nonsep) else "")),
         "A zone can be important yet not discriminating. Importance bars alone cannot show this - "
         "only the signed split can."),

        ("ZoneWaterfall",
         "How one representative spectrum's prediction is assembled, zone by zone.",
         "Starts at the reference prediction; each bar adds that zone's contribution until the "
         "model's final P(OIM) is reached. Red pushes towards OIM, blue towards WT.",
         f"The spectrum shown is from animal {animals[rep]}, true class {rep_cls}. Its largest "
         f"single contribution comes from {rep_top} at {rep_val:+.4f}.",
         "This is the MEDIAN-impact spectrum, chosen deliberately to be typical rather than "
         "extreme. It illustrates the mechanism for one sample and is not an average over samples."),

        ("SHAP_FullSpectrum_Overlay",
         "Mean |SHAP| plotted against wavenumber across the whole fingerprint region, with the "
         "nine zones shaded behind the curve.",
         "Read it like a spectrum: peaks mark influential regions, and the shading shows which "
         "zone each peak sits in. The axis runs 1800 -> 800 cm-1 by spectroscopy convention.",
         peak_txt,
         "This figure replaces a ranked top-20 wavenumber list. Because neighbouring channels in a "
         "spectrum are strongly correlated, a ranked list repeats the same band many times; "
         "plotting the full range shows bands instead of individual channels."),
    ]
    return pd.DataFrame(rows, columns=["Figure", "What it shows", "How to read it",
                                       "What THIS result shows", "Caveat / note"])


def _zs_fig_map(feat_cols, names, idx):
    """Orientation figure: which part of the spectrum each zone covers."""
    wn = np.array([_zs_wavenumber(c) for c in feat_cols], dtype=float)   # Wavenumber axis
    fig, ax = plt.subplots(figsize=(9.2, 2.6))                           # Wide, short strip
    for n, ii in zip(names, idx):                                        # Shade each zone
        if n == _ZS_UNASSIGNED or not len(ii):                           # Leave gaps unshaded
            continue
        lo, hi = np.nanmin(wn[ii]), np.nanmax(wn[ii])                    # Extent of the zone
        ax.axvspan(lo, hi, color=_ZS_COLORS.get(n, "#999"), alpha=0.55)  # Coloured band
        ax.text((lo + hi) / 2, 0.5, n, rotation=90, ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")            # Zone label
    ax.set_xlim(np.nanmax(wn), np.nanmin(wn))                            # High to low, as usual
    ax.set_yticks([])                                                    # No information on y
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")                              # Axis label
    ax.set_title("Spectral zones used for grouped SHAP")                 # Title
    fig.tight_layout()                                                   # Tidy
    return fig


def _zs_fig_importance(vals, names, title, per_channel=False, channels=None):
    """Horizontal bar chart of mean |contribution| per zone - the headline zone figure."""
    imp = np.abs(vals).mean(axis=0)                                      # Mean absolute effect
    xlab = "Mean |zone contribution|"                                    # Default label
    if per_channel and channels is not None:                             # Width-normalised view
        imp = imp / np.maximum(np.asarray(channels, float), 1)           # Per-wavenumber density
        xlab = "Mean |contribution| per wavenumber"                      # Comparable across widths
    o = np.argsort(imp)                                                  # Largest ends up on top
    fig, ax = plt.subplots(figsize=(8.2, 5.2))                           # Panel size
    ax.barh([names[i] for i in o], imp[o],                               # One bar per zone
            color=[_ZS_COLORS.get(names[i], "#777") for i in o])
    for i, v in enumerate(imp[o]):                                       # Annotate the values
        ax.text(v, i, f" {v:.4f}", va="center", fontsize=8)
    ax.set_xlabel(xlab); ax.set_title(title); ax.grid(axis="x", alpha=0.25)   # Labels and grid
    fig.tight_layout()                                                   # Tidy
    return fig


def _zs_fig_spectrum_overlay(full_tbl, title):
    """Mean |SHAP| plotted across the WHOLE spectrum, with the zones shaded behind it.

    This is the figure the professor described on the call: importance along the spectral axis
    rather than a ranked list, so bands rather than individual wavenumbers can be read off.
    """
    d = full_tbl.dropna(subset=["Wavenumber (cm-1)"]).sort_values("Wavenumber (cm-1)")
    fig, ax = plt.subplots(figsize=(10.5, 4.6))                          # Wide spectral panel

    # Draw the data FIRST, so the y-axis is scaled to the real values before anything is placed
    # relative to it.
    ax.plot(d["Wavenumber (cm-1)"], d["Mean |SHAP|"], lw=1.1, color="#222")   # Importance trace

    for n, lo, hi in _ZS_ZONES:                                          # Then shade each zone
        ax.axvspan(lo, hi, color=_ZS_COLORS.get(n, "#999"), alpha=0.16)  # Light background band
        # CRITICAL: position the label in AXES-FRACTION coordinates on y via
        # get_xaxis_transform() - x stays in data units (wavenumbers), y becomes 0-1 across the
        # axes. Using a data coordinate here (e.g. ax.get_ylim()[1]) is a trap: read before
        # plotting it returns the default limit of 1.0, whereas real mean |SHAP| values are
        # around 0.004, so the labels land far above the data and bbox_inches="tight" then
        # expands the saved image to include them - producing a figure tens of thousands of
        # pixels tall that is almost entirely blank.
        ax.text((lo + hi) / 2, 0.985, n, rotation=90, va="top", ha="center",
                fontsize=7, color="#444", transform=ax.get_xaxis_transform())

    ax.set_xlim(d["Wavenumber (cm-1)"].max(), d["Wavenumber (cm-1)"].min())   # High to low
    ax.set_ylim(0, float(d["Mean |SHAP|"].max()) * 1.18)                 # Headroom for the labels
    ax.set_xlabel("Wavenumber (cm$^{-1}$)"); ax.set_ylabel("Mean |SHAP|")     # Axis labels
    ax.set_title(title)                                                  # Title
    fig.tight_layout()                                                   # Tidy
    return fig


def _zs_fig_beeswarm(vals, names, labels, title):
    """Per-zone strip plot of signed contributions, split by true class."""
    o = np.argsort(np.abs(vals).mean(axis=0))                            # Least to most important
    fig, ax = plt.subplots(figsize=(8.6, 5.6))                           # Panel size
    rng = np.random.RandomState(0)                                       # Fixed jitter
    for row, j in enumerate(o):                                          # One band per zone
        for cv, cn in [(0, "WT"), (1, "OIM")]:                           # Two classes per band
            m = labels == cv                                             # Mask for this class
            if not m.any():                                              # Defensive
                continue
            off = -0.16 if cv == 0 else 0.16                             # Separate the classes
            ax.scatter(vals[m, j], np.full(m.sum(), row) + off + rng.uniform(-.07, .07, m.sum()),
                       s=5, alpha=0.35, c=_ZS_CLASS_COLORS[cn], edgecolors="none",
                       label=cn if row == 0 else None)                   # Legend drawn once
    ax.axvline(0, color="grey", lw=0.9, ls="--")                         # Zero reference
    ax.set_yticks(range(len(o))); ax.set_yticklabels([names[j] for j in o])   # Zone labels
    ax.set_xlabel("Zone contribution to P(OIM)  -  negative = towards WT, positive = towards OIM")
    ax.set_title(title); ax.legend(frameon=False, loc="lower right")     # Title and legend
    fig.tight_layout()                                                   # Tidy
    return fig


def _zs_fig_waterfall(phi, names, base, title):
    """Waterfall for one representative spectrum, built from zone contributions."""
    o = np.argsort(np.abs(phi))[::-1]                                    # Largest first
    v, nm = phi[o], [names[i] for i in o]                                # Values and labels
    fig, ax = plt.subplots(figsize=(8.4, 5.4))                           # Panel size
    run = base                                                           # Running total
    for i, (n_, val) in enumerate(zip(nm, v)):                           # Draw each step
        ax.barh(len(v) - i - 1, val, left=run,                           # Bar from run to run+val
                color="#D7263D" if val > 0 else "#2E86AB", height=0.68)  # Red = towards OIM
        ax.text(run + val + (0.004 if val > 0 else -0.004), len(v) - i - 1, f"{val:+.3f}",
                va="center", ha="left" if val > 0 else "right", fontsize=8)   # Value label
        run += val                                                       # Accumulate
    ax.set_yticks(range(len(v))); ax.set_yticklabels(nm[::-1])           # Zone labels
    ax.axvline(base, color="grey", ls="--", lw=1.0, label=f"reference = {base:.3f}")
    ax.axvline(run, color="black", lw=1.2, label=f"final P(OIM) = {run:.3f}")
    ax.set_xlabel("P(OIM)"); ax.set_title(title)                         # Labels
    ax.legend(frameon=False, fontsize=8, loc="lower right")              # Explain the lines
    fig.tight_layout()                                                   # Tidy
    return fig
# ---------------------------------- END: Zone figures ----------------------------------------


# ------------------------------- START: Streamlit section ------------------------------------
st.sidebar.markdown("---")                       # Divider from the sections above
_zs_show = st.sidebar.checkbox(                  # This section is opt-in and off by default
    "🧭 Spectral Zone-Based SHAP (NEW)",
    value=False,
    help="Grouped-feature SHAP following Contreras et al., Anal. Chem. 2024, plus fully editable "
         "full-spectrum SHAP tables. Does not affect any existing result.",
)

if _zs_show:                                     # Everything below runs only when ticked
    st.markdown("---")                           # Visual separation from the section above
    st.markdown("<div id='zoneshap'></div>", unsafe_allow_html=True)   # Nav anchor
    st.header("🧭 Spectral Zone-Based SHAP")     # Section heading
    st.caption(
        "Grouped-feature explainability following Contreras, Winterfeld, Popp & Bocklitz, "
        "*Analytical Chemistry* 2024, 96(39), 15588-15597 (DOI 10.1021/acs.analchem.4c02329). "
        "Runs alongside — and does not replace — the existing per-wavenumber SHAP results."
    )
    st.info(
        "**Why zones.** Neighbouring wavenumbers in a spectrum are strongly correlated and never "
        "vary independently in a real sample, so perturbing one channel at a time asks the model "
        "a question that physically cannot happen. Perturbing a whole band asks something a "
        "spectroscopist can act on: *what does the model lose if the entire Amide I region goes?*",
        icon="🧪")

    with st.sidebar:                             # This section's own controls
        st.markdown("**Zone SHAP settings**")    # Sub-heading
        _zs_up = st.file_uploader("Dataset (.xlsx)", type=["xlsx"], key="zs_upload")
        _zs_sheet = st.selectbox("Sheet", ["Raw", "2nd derivative"], key="zs_sheet",
                                 help="Zone analysis needs a wavenumber axis, so the peak-ratio "
                                      "sheet does not apply.")
        _zs_n = int(st.number_input(
            "Spectra per animal (TRAINING - the professor's 250)", 50, 2000, 250, 50, key="zs_n",
            help="The balanced design from the 1 June 2026 request. Each model is trained on "
                 "this many spectra from each of the 11 training animals in every fold. Leave "
                 "at 250 to match the accuracy table and every other result in the paper."))
        _zs_seed = int(st.number_input("Random seed", 0, 10000, 42, 1, key="zs_seed"))
        _zs_method = st.radio("Method",
                              ["B - Zone perturbation (paper's method)",
                               "A - Zone-aggregated SHAP"], key="zs_method")[0]
        _zs_cap = int(st.number_input(
            "Spectra EXPLAINED per animal (SHAP cost only)", 2, 250,
            10 if _zs_method == "B" else 50, 1, key="zs_cap",
            help="How many of the held-out spectra get an explanation computed. This does NOT "
                 "change the models or any accuracy - it only affects how many spectra the SHAP "
                 "values are averaged over. Method B evaluates 2^n_zones coalitions per "
                 "spectrum, so raising it is expensive. With 12 animals the uncertainty is "
                 "dominated by between-animal variation, so going from 10 to 250 narrows the "
                 "confidence interval by only about 4%."))
        _zs_editable = st.checkbox("Also export editable full-spectrum SHAP tables", value=True,
                                   key="zs_editable",
                                   help="SHAP value at EVERY wavenumber 1800-800, not a top-20 "
                                        "list - what Prof. Querido asked for on the call.")
        _zs_pick = st.multiselect("Models", list(_zs_models(_zs_seed).keys()),
                                  default=list(_zs_models(_zs_seed).keys())[:1], key="zs_models")
        _zs_go = st.button("▶️ Run zone analysis", type="primary", key="zs_go")
        st.caption("**Two different numbers.** *Spectra per animal* = the professor's balanced "
                   "250, used to TRAIN. *Spectra explained* = how many held-out spectra get a "
                   "SHAP explanation. Changing the second never changes a model or an accuracy.")

    if not _zs_go:                               # Idle state: show the zone table, not a blank page
        _n0, _i0, _ = _zs_assign([str(w) for w in range(1800, 798, -2)])   # Nominal 1800-800 grid
        st.subheader("Zone definitions")         # Section heading
        st.dataframe(_zs_zone_table(_n0, _i0), width="stretch", hide_index=True)
        st.caption("Overlapping endpoints (1600, 1430, 1200, 1100) are assigned to the first "
                   "zone in the professor's listed order. Channels in no zone are collected as "
                   "'Unassigned (gaps)' so contributions still reconcile to the model output.")
        st.info("Upload the dataset and press **Run zone analysis**.", icon="👈")
    elif _zs_up is None:                         # Cannot proceed without data
        st.error("Upload the dataset workbook first.", icon="🚫")
    else:                                        # Run the analysis
        _zs_tmp = os.path.join(".", "_zone_shap_upload.xlsx")   # Loader needs a real path
        with open(_zs_tmp, "wb") as _fh:         # Persist the upload
            _fh.write(_zs_up.getbuffer())
        _zs_prog = st.progress(0.0, text="Loading and balancing ...")   # Visible progress
        try:
            _zsX, _zsy, _zsg, _zsfc = _zs_load_balanced(_zs_tmp, _zs_sheet, _zs_n, _zs_seed)
            _zsnames, _zsidx, _zsaudit = _zs_assign(_zsfc)        # Partition the spectrum
            _zsch = [len(i) for i in _zsidx]                      # Channels per zone
            st.success(f"{_zsX.shape[0]:,} spectra x {_zsX.shape[1]} channels, "
                       f"{len(np.unique(_zsg))} animals", icon="✅")

            _zsmods = _zs_models(_zs_seed)                        # Instantiate the classifiers
            _zsres, _zssheets = {}, {}                            # Results and workbook sheets
            for _k, _mn in enumerate(_zs_pick):                   # One run per chosen model
                _zs_prog.progress((_k + 0.1) / max(len(_zs_pick), 1), text=f"{_mn} ...")
                _tag = f"{_mn} - {'zone perturbation' if _zs_method == 'B' else 'zone-aggregated'}"

                if _zs_method == "B":                             # The paper's method
                    _phi, _lab, _ani, _chk = _zs_zone_perturbation(
                        _zsX, _zsy, _zsg, _zsidx, _zsmods[_mn], _zs_seed, _zs_cap)
                    _base = 0.5                                   # Waterfall reference
                else:                                             # Aggregate per-wavenumber SHAP
                    _sv, _vals, _lab, _ani = _zs_perwavenumber_shap(
                        _zsX, _zsy, _zsg, _zsmods[_mn], _zs_seed, _zs_cap)
                    _phi = np.column_stack([_sv[:, i].sum(axis=1) for i in _zsidx])   # Exact sum
                    _chk = None                                   # No axiom check for Method A
                    _base = 0.0                                   # Deviation from expected value

                _per_zone = pd.DataFrame({                        # Zone summary table
                    "Zone": _zsnames, "Channels": _zsch,
                    "Mean |contribution|": np.abs(_phi).mean(axis=0),
                    "Mean |contribution| per wavenumber":
                        np.abs(_phi).mean(axis=0) / np.maximum(_zsch, 1),
                    "Mean signed contribution": _phi.mean(axis=0),
                    "Mean signed - WT": _phi[_lab == 0].mean(axis=0),
                    "Mean signed - OIM": _phi[_lab == 1].mean(axis=0),
                }).sort_values("Mean |contribution|", ascending=False)

                _rep = int(np.argsort(np.abs(_phi).sum(axis=1))[len(_phi) // 2])   # Median impact
                _figs = {                                          # Figures for this model
                    "ZoneMap": _zs_fig_map(_zsfc, _zsnames, _zsidx),
                    "ZoneImportance": _zs_fig_importance(_phi, _zsnames, f"Zone importance - {_tag}"),
                    "ZoneImportance_perWavenumber": _zs_fig_importance(
                        _phi, _zsnames, f"Zone importance per wavenumber - {_tag}",
                        per_channel=True, channels=_zsch),
                    "ZoneBeeswarm": _zs_fig_beeswarm(_phi, _zsnames, _lab,
                                                     f"Zone contributions - {_tag}"),
                    "ZoneWaterfall": _zs_fig_waterfall(_phi[_rep], _zsnames, _base,
                                                       f"Representative zone waterfall - {_tag}"),
                }

                _safe = _mn.replace(" ", "")[:16]                  # Excel sheet-name budget
                _zssheets[f"{_safe} zones"[:31]] = _per_zone       # Zone summary
                _raw = pd.DataFrame(_phi, columns=_zsnames)        # Per-spectrum zone values
                _raw.insert(0, "True class", np.where(_lab == 1, "OIM", "WT"))
                _raw.insert(0, "Animal", _ani)
                _zssheets[f"{_safe} per-spectrum"[:31]] = _raw     # Per-spectrum table
                if _chk is not None:                               # Method B exactness audit
                    _zssheets[f"{_safe} exactness"[:31]] = _chk

                # ---- EDITABLE full-spectrum SHAP tables (the professor's call request) ----
                if _zs_editable:                                   # Optional but on by default
                    _zs_prog.progress((_k + 0.6) / max(len(_zs_pick), 1),
                                      text=f"{_mn} - full-spectrum SHAP export ...")
                    if _zs_method == "B":                          # Method A already computed them
                        _sv, _vals, _lab2, _ani2 = _zs_perwavenumber_shap(
                            _zsX, _zsy, _zsg, _zsmods[_mn], _zs_seed, _zs_cap)
                    else:                                          # Reuse what we already have
                        _lab2, _ani2 = _lab, _ani
                    _full, _wf, _wfmeta, _perspec = _zs_editable_tables(
                        _sv, _vals, _lab2, _ani2, _zsfc, _zsnames, _zsidx)
                    _zssheets[f"{_safe} SHAP full range"[:31]] = _full        # Every wavenumber
                    _zssheets[f"{_safe} SHAP waterfall"[:31]] = _wf           # Waterfall AS VALUES
                    _zssheets[f"{_safe} waterfall info"[:31]] = _wfmeta       # Which spectrum
                    _zssheets[f"{_safe} SHAP by spectrum"[:31]] = _perspec    # Full matrix
                    _figs["SHAP_FullSpectrum_Overlay"] = _zs_fig_spectrum_overlay(
                        _full, f"Mean |SHAP| across the spectrum - {_mn}")

                _zs_interp = _zs_interpretation(                       # Grounded in the numbers
                    _per_zone, _lab,
                    "zone perturbation (Contreras et al. 2024)" if _zs_method == "B"
                    else "zone-aggregated per-wavenumber SHAP", _mn, len(_phi),
                    _zs_n_per_animal_used=_zs_n)
                _zssheets[f"{_safe} interpretation"[:31]] = _zs_interp   # Editable, per model
                # Per-FIGURE descriptions: what each image is, and what THIS run shows in it.
                _zs_desc = _zs_figure_descriptions(
                    _per_zone, _phi, _lab, _ani, _zsnames, _zsch,
                    _full if (_zs_editable and "_full" in dir()) else None, _mn,
                    "zone perturbation (Contreras et al. 2024)" if _zs_method == "B"
                    else "zone-aggregated per-wavenumber SHAP")
                _zssheets[f"{_safe} figure notes"[:31]] = _zs_desc      # Editable, per model
                _zssheets[f"{_safe} plain summary"[:31]] = _zs_plain_summary(
                    _per_zone, _lab, _mn, len(_phi), _zs_n)            # Non-technical write-up
                _zsres[_mn] = {"per_zone": _per_zone, "figs": _figs, "checks": _chk,
                               "interp": _zs_interp, "desc": _zs_desc,
                               "plain": _zs_plain_summary(_per_zone, _lab, _mn, len(_phi), _zs_n),
                               # Kept so native Excel charts can be built from the same numbers
                               "chart_full": (_full if _zs_editable else None),
                               "chart_wf": _phi[_rep], "chart_zn": list(_zsnames)}

            _zs_prog.progress(1.0, text="Complete")               # Finished

            # ---- Build the workbook ONCE, here, while the results are in hand ----
            _zsbuf = io.BytesIO()                                 # Workbook built in memory
            with pd.ExcelWriter(_zsbuf, engine="openpyxl") as _xw:
                pd.DataFrame({                                    # README documenting the run
                    "Item": ["Generated", "Dataset", "Sheet", "Spectra per animal", "Random seed",
                             "Validation", "Method", "Spectra explained per animal",
                             "Reference", "Zone assignment rule", "Editable tables"],
                    "Value": [datetime.now().strftime("%Y-%m-%d %H:%M"), _zs_up.name, _zs_sheet,
                              _zs_n, _zs_seed,
                              "Leave-One-Group-Out CV, grouping variable = Animal ID",
                              "Zone perturbation (Contreras et al. 2024)" if _zs_method == "B"
                              else "Zone-aggregated per-wavenumber SHAP", _zs_cap,
                              "Contreras et al., Anal. Chem. 2024, 96(39), 15588-15597; "
                              "DOI 10.1021/acs.analchem.4c02329",
                              "First matching zone in the professor's listed order wins; "
                              "channels in no zone are collected as 'Unassigned (gaps)'",
                              "SHAP value at every wavenumber 1800-800, plus waterfall values"],
                }).to_excel(_xw, sheet_name="README", index=False)
                pd.DataFrame({
                    "Item": ["Native Excel charts", "Where", "How to edit", "Also provided"],
                    "Value": [
                        "Yes - sheets named '<Model> CHART ...' contain real Excel chart objects",
                        "CHART zones (importance + per wavenumber), CHART direction (WT vs OIM), "
                        "CHART spectrum (full 1800-800 curve), CHART waterfall (representative "
                        "spectrum)",
                        "Click any chart in Excel to change fonts, colours, axis titles or size. "
                        "The data sits beside the chart on the same sheet, so editing a number "
                        "updates the graph.",
                        "The same figures as 300 dpi PNG and vector SVG in the figures ZIP. SVG "
                        "is the better format for the final journal figure; these charts are for "
                        "editing inside Excel."],
                }).to_excel(_xw, sheet_name="README", index=False)
                _zs_what_is_shap().to_excel(_xw, sheet_name="What is SHAP", index=False)
                _zs_zone_table(_zsnames, _zsidx).to_excel(_xw, sheet_name="Zone definitions",
                                                          index=False)
                _zsaudit.to_excel(_xw, sheet_name="Channel assignment", index=False)
                _zs_figure_guide().to_excel(_xw, sheet_name="Figure guide", index=False)
                _zs_zone_chemistry().to_excel(_xw, sheet_name="Zone chemistry", index=False)
                _zs_glossary().to_excel(_xw, sheet_name="Glossary", index=False)
                _zs_method_comparison().to_excel(_xw, sheet_name="Method A vs Method B",
                                                 index=False)
                for _sn, _tbl in _zssheets.items():               # Every model's tables
                    _tbl.to_excel(_xw, sheet_name=_sn[:31], index=False)
                # NATIVE EXCEL CHARTS - added last, once every data sheet exists. These are real
                # chart objects the professor can restyle inside Excel, with the numbers beside
                # them so edits flow straight through to the graph.
                for _cm, _cr in _zsres.items():
                    try:
                        _zs_add_excel_charts(_xw.book, _cm, _cr["per_zone"],
                                             _cr.get("chart_full"), _cr.get("chart_wf"),
                                             _cr.get("chart_zn"))
                        # ALSO put charts straight onto the data sheets, so they are visible
                        # without hunting through the tab bar for a separate CHART sheet.
                        _zs_embed_charts_on_data_sheets(_xw.book, _cm.replace(" ", "")[:16])
                    except Exception as _ce:      # A chart failure must never lose the data
                        print(f"chart build skipped for {_cm}: {_ce}")

            # ---- Render every figure to PNG + SVG bytes, then release the figure objects ----
            # WHY BYTES: a matplotlib Figure cannot be kept in st.session_state reliably, and
            # Streamlit re-executes the whole script on EVERY widget interaction - including a
            # download-button click. Storing rendered bytes lets the results survive those
            # reruns, so downloading no longer wipes the analysis and force a re-run.
            _zs_png, _zs_zipbytes = {}, io.BytesIO()              # PNG per figure, and the archive
            with _zs_zipfile.ZipFile(_zs_zipbytes, "w", _zs_zipfile.ZIP_DEFLATED) as _zf:
                for _mn, _r in _zsres.items():                    # Every model
                    _zs_png[_mn] = {}                             # Its rendered figures
                    for _fn, _f in _r["figs"].items():            # Every figure of that model
                        _p = io.BytesIO(); _f.savefig(_p, format="png", dpi=300,
                                                      bbox_inches="tight", facecolor="white")
                        _zs_png[_mn][_fn] = _p.getvalue()         # Keep for on-screen display
                        _zf.writestr(f"{_mn.replace(' ', '_')}/{_fn}.png", _p.getvalue())
                        _s2 = io.BytesIO(); _f.savefig(_s2, format="svg", bbox_inches="tight",
                                                       facecolor="white")
                        _zf.writestr(f"{_mn.replace(' ', '_')}/{_fn}.svg", _s2.getvalue())
                        plt.close(_f)                             # Release immediately

            # ---- Persist everything, so reruns re-display instead of recomputing ----
            st.session_state["zs_results"] = {
                "png": _zs_png,                                   # model -> figure -> PNG bytes
                "tables": {_mn: {"per_zone": _r["per_zone"], "interp": _r["interp"],
                                 "checks": _r["checks"], "desc": _r.get("desc"),
                                 "plain": _r.get("plain")}
                           for _mn, _r in _zsres.items()},
                "workbook": _zsbuf.getvalue(),                    # The .xlsx bytes
                "zip": _zs_zipbytes.getvalue(),                   # The figures archive
                "stamp": datetime.now().strftime("%Y%m%d_%H%M%S"),   # Used in the file names
                "sheet": _zs_sheet,                               # Shown in the caption
                "method": "Zone perturbation (Contreras et al. 2024)" if _zs_method == "B"
                          else "Zone-aggregated per-wavenumber SHAP",
                "n_per_animal": _zs_n, "seed": _zs_seed, "cap": _zs_cap,   # Run settings
            }
        except Exception as _zs_err:                              # Surface errors in the UI
            st.error("Zone SHAP analysis failed.")
            st.exception(_zs_err)
        finally:                                                  # Always clean up
            if os.path.exists(_zs_tmp):                           # Only if it was written
                os.remove(_zs_tmp)                                # Leave nothing on disk

    # ---------------------------------------------------------------------------------------
    # PERSISTENT RESULTS BLOCK - deliberately OUTSIDE the "Run" branch above.
    # Streamlit re-executes this script top to bottom on every widget interaction, and a
    # download-button click counts as one. Anything rendered inside "if _zs_go:" therefore
    # disappears the moment a download is clicked, which is why the results used to vanish and
    # have to be recomputed. Reading from st.session_state instead means downloads, tab
    # switches and slider changes all re-display the SAME results with no recomputation.
    # ---------------------------------------------------------------------------------------
    _zs_saved = st.session_state.get("zs_results")     # Results from the most recent run, if any
    if _zs_saved:                                      # Render them on every rerun
        st.markdown("---")                             # Separate from the controls above
        st.subheader("Zone SHAP results")              # Section heading
        st.caption(f"{_zs_saved['sheet']} · {_zs_saved['method']} · "
                   f"{_zs_saved['n_per_animal']} spectra/animal · seed {_zs_saved['seed']} · "
                   f"{_zs_saved['cap']} explained per animal · run {_zs_saved['stamp']}")
        st.info("These results are held in the session, so downloading a file no longer clears "
                "them. Press **Run zone analysis** again only when you change a setting.",
                icon="💾")

        _zs_dl1, _zs_dl2, _zs_dl3 = st.columns([2, 2, 1])   # Downloads first, so they are reachable
        with _zs_dl1:                                       # The editable workbook
            st.download_button(
                "📥 Zone-SHAP + editable full-spectrum workbook (.xlsx)",
                data=_zs_saved["workbook"],
                file_name=f"Zone_SHAP_{_zs_saved['sheet'].replace(' ', '_')}_"
                          f"{_zs_saved['stamp']}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="zs_dl_workbook", use_container_width=True)
        with _zs_dl2:                                       # The figure archive
            st.download_button(
                "📦 Zone-SHAP figures (PNG + editable SVG)",
                data=_zs_saved["zip"],
                file_name=f"Zone_SHAP_Figures_{_zs_saved['stamp']}.zip",
                mime="application/zip", key="zs_dl_figs", use_container_width=True)
        with _zs_dl3:                                       # Explicit way to discard the results
            if st.button("🗑️ Clear", key="zs_clear", use_container_width=True,
                         help="Discard these results and start fresh."):
                del st.session_state["zs_results"]           # Drop them
                st.rerun()                                   # Redraw the empty state

        _zs_w = int(st.slider("Figure width on screen (px)", 300, 1400, 900, 50,
                              key="zs_disp_width",
                              help="Display only - downloads are always full 300 dpi / vector."))

        for _tb, _mn in zip(st.tabs(list(_zs_saved["png"].keys())),
                            list(_zs_saved["png"].keys())):      # One tab per model
            with _tb:
                _t = _zs_saved["tables"][_mn]                    # That model's tables
                st.dataframe(_t["per_zone"].round(5), width="stretch", hide_index=True)
                with st.expander("📝 Plain-language summary — start here", expanded=True):
                    st.caption("Written for a reader with no spectroscopy or machine-learning "
                               "background. Also in the workbook.")
                    st.dataframe(_t.get("plain") if _t.get("plain") is not None
                                 else pd.DataFrame(), width="stretch", hide_index=True)
                with st.expander("📖 Detailed interpretation and figure guide", expanded=False):
                    st.dataframe(_t["interp"], width="stretch", hide_index=True)
                    st.dataframe(_zs_figure_guide(), width="stretch", hide_index=True)
                with st.expander("❓ What is SHAP, and what does it tell us here?",
                                 expanded=False):
                    st.caption("Start here if you are new to SHAP. Also the first sheet of the "
                               "workbook.")
                    st.dataframe(_zs_what_is_shap(), width="stretch", hide_index=True)
                with st.expander("⚖️ Method A vs Method B — how zone importance is computed",
                                 expanded=False):
                    st.caption("The methodological core of this section. Also in the workbook.")
                    st.dataframe(_zs_method_comparison(), width="stretch", hide_index=True)
                with st.expander("🧪 What each spectral zone IS (chemistry)", expanded=False):
                    st.dataframe(_zs_zone_chemistry(), width="stretch", hide_index=True)
                with st.expander("📚 Glossary — every term explained", expanded=False):
                    st.dataframe(_zs_glossary(), width="stretch", hide_index=True)
                if _t["checks"] is not None:                     # Method B exactness audit
                    _e = _t["checks"]["abs error"].max()         # Worst deviation
                    st.success(
                        f"Exactness check: max |sum(zone Shapley) − (f(x) − f(reference))| "
                        f"= {_e:.2e} over {len(_t['checks'])} spectra. The Shapley efficiency "
                        f"axiom holds, so these values are exact, not estimated.", icon="✅")
                _zs_d = _t.get("desc")                               # Per-figure descriptions
                for _fn, _bytes in _zs_saved["png"][_mn].items():    # Every figure, as bytes
                    st.markdown(f"### {_fn.replace('_', ' ')}")      # Figure title
                    st.image(_bytes, width=_zs_w)                   # Display at the chosen width
                    if _zs_d is not None and _fn in set(_zs_d["Figure"]):   # Matching description
                        _row = _zs_d[_zs_d["Figure"] == _fn].iloc[0]        # Its row
                        st.markdown(
                            f"**What it shows.** {_row['What it shows']}\n\n"
                            f"**How to read it.** {_row['How to read it']}\n\n"
                            f"**What THIS result shows.** {_row['What THIS result shows']}\n\n"
                            f"*{_row['Caveat / note']}*")            # Full description under the image
                    st.download_button(f"📥 {_fn}.png", data=_bytes,
                                       file_name=f"{_mn.replace(' ', '_')}_{_fn}.png",
                                       mime="image/png",
                                       key=f"zs_dl_{_mn}_{_fn}")    # Unique key per figure
                    st.markdown("---")                               # Separate the figures
# -------------------------------- END: Streamlit section -------------------------------------
# =============================================================================================
# ####################  END SECTION: SPECTRAL ZONE-BASED SHAP  ################################
# =============================================================================================


# =============================================================================================
# ################  SECTION: FIG 2a - CONVENTIONAL PEAK-RATIO PARAMETERS  #####################
# =============================================================================================
# Self-contained, appended. Produces the "standard quantitative analysis" panel the professor
# listed as Fig. 2a: the conventional peak-ratio parameters compared between WT and OIM, which is
# how this field has always presented bone FTIR before any multivariate modelling.
#
# It deliberately uses the SAME balanced design as everything else (250 spectra per animal, fixed
# seed), so the panel is directly comparable with the PCA, PLS-DA and ML results beside it.
# =============================================================================================

def _q2a_stats(v_wt, v_oim):
    """Descriptive statistics plus a Mann-Whitney U test computed from ranks, using scipy."""
    out = {"WT n": len(v_wt), "OIM n": len(v_oim),
           "WT mean": float(np.mean(v_wt)), "OIM mean": float(np.mean(v_oim)),
           "WT SD": float(np.std(v_wt, ddof=1)), "OIM SD": float(np.std(v_oim, ddof=1)),
           "WT median": float(np.median(v_wt)), "OIM median": float(np.median(v_oim))}
    out["Difference (OIM - WT)"] = out["OIM mean"] - out["WT mean"]                # Raw effect
    pooled = np.sqrt(((len(v_wt) - 1) * out["WT SD"]**2 +                          # Cohen's d
                      (len(v_oim) - 1) * out["OIM SD"]**2) / (len(v_wt) + len(v_oim) - 2))
    out["Cohen's d"] = out["Difference (OIM - WT)"] / pooled if pooled > 0 else np.nan
    try:                                                       # Non-parametric: no normality claim
        from scipy import stats                                 # Imported INSIDE the try, so a
        u, p = stats.mannwhitneyu(v_oim, v_wt, alternative="two-sided")   # missing scipy only
                                                                # drops the p-value, not the table
        out["Mann-Whitney U"] = float(u); out["p-value"] = float(p)
    except Exception:
        out["Mann-Whitney U"] = np.nan; out["p-value"] = np.nan
    return out


def _q2bc_pca_plsda(X, y, groups, feat_names, seed):
    """Fig 2b + 2c: PCA and PLS-DA on the QUANTITATIVE peak-ratio parameters.

    Same animal-bound design as every other result: leave-one-animal-out, scaler and model fitted
    on the training animals only. Two accuracies are produced and they mean different things:

      * PCA is UNSUPERVISED - the score plot itself has no accuracy. What is reported is a
        logistic regression on PC1+PC2 under the same LOGO scheme, which is how this field
        summarises "how separable are the groups in PCA space".
      * PLS-DA is SUPERVISED, so its scores are computed OUT-OF-FOLD: each animal is projected by
        a model fitted on the other 11. Anything else would show optimistically clean separation.
    """
    logo = LeaveOneGroupOut()                                  # Animal-bound splitter
    n_pc = min(3, X.shape[1], max(1, X.shape[0] - 1))          # Up to 3 components
    pca_oof = np.full(len(y), -1, dtype=int)                   # PCA+LR out-of-fold predictions
    pls_scores = np.full((len(y), max(2, n_pc)), np.nan)       # PLS-DA out-of-fold scores
    pls_pred = np.full(len(y), -1, dtype=int)                  # PLS-DA out-of-fold predictions

    for tr, te in logo.split(X, y, groups):                    # One held-out animal per fold
        sc = StandardScaler().fit(X[tr])                       # Fitted on TRAIN animals only
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        pf = PCA(n_components=n_pc, random_state=42).fit(Xtr)  # PCA on train only
        lr = LogisticRegression(max_iter=3000, random_state=seed).fit(pf.transform(Xtr), y[tr])
        pca_oof[te] = lr.predict(pf.transform(Xte))            # PCA+LR prediction
        pls = PLSRegression(n_components=n_pc, scale=False).fit(Xtr, y[tr].astype(float).reshape(-1, 1))
        _t = pls.transform(Xte); pls_scores[te, :_t.shape[1]] = _t     # Out-of-fold projection
        pls_pred[te] = (pls.predict(Xte).ravel() >= 0.5).astype(int)   # 0.5 decision threshold

    # Display PCA scores come from a full-data fit: PCA is unsupervised, so this is descriptive
    # and carries no leakage risk. The ACCURACY above is the out-of-fold quantity.
    sc_all = StandardScaler().fit(X); Xz = sc_all.transform(X)
    pca_all = PCA(n_components=n_pc, random_state=42).fit(Xz)
    pca_scores = pca_all.transform(Xz)
    pls_all = PLSRegression(n_components=n_pc, scale=False).fit(Xz, y.astype(float).reshape(-1, 1))
    vip = _zs_vip_like(pls_all, X.shape[1])                    # VIP over the five parameters

    return {
        "pca_scores": pca_scores, "pca_evr": pca_all.explained_variance_ratio_,
        "pca_loadings": pca_all.components_, "pca_accuracy": float(accuracy_score(y, pca_oof)),
        "pca_cm": confusion_matrix(y, pca_oof, labels=[0, 1]),
        "plsda_scores": pls_scores, "plsda_accuracy": float(accuracy_score(y, pls_pred)),
        "plsda_cm": confusion_matrix(y, pls_pred, labels=[0, 1]),
        "vip": vip, "features": list(feat_names),
    }


def _zs_vip_like(pls_model, n_feat):
    """VIP scores for a fitted PLS model. Mean of squared VIP equals 1 by construction."""
    t = pls_model.x_scores_; w = pls_model.x_weights_; q = pls_model.y_loadings_
    p, h = w.shape                                             # p features, h components
    ssy = np.sum(t ** 2, axis=0) * np.sum(q ** 2, axis=0)      # y-variance explained per component
    tot = np.sum(ssy)                                          # Denominator
    out = np.zeros(p)
    for i in range(p):                                         # Standard VIP formula
        wgt = np.array([(w[i, j] / np.linalg.norm(w[:, j])) ** 2 for j in range(h)])
        out[i] = np.sqrt(p * (ssy @ wgt) / tot) if tot > 0 else 0.0
    return out


def _q2bc_figure(res, y, title_prefix):
    """Two score plots side by side: PCA (2b) and out-of-fold PLS-DA (2c)."""
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.0))         # Side-by-side panels
    cols = {"WT": "#2E86AB", "OIM": "#D7263D"}                  # Consistent palette
    mk = {"WT": "o", "OIM": "^"}                                # Consistent markers
    for ax, key, lbl, acc in [
            (axes[0], "pca_scores", "PCA", res["pca_accuracy"]),
            (axes[1], "plsda_scores", "PLS-DA (out-of-fold)", res["plsda_accuracy"])]:
        S = res[key]
        for cls, val in [("WT", 0), ("OIM", 1)]:                # Plot each class
            m = (y == val)
            ax.scatter(S[m, 0], S[m, 1], s=9, alpha=0.45, c=cols[cls],
                       marker=mk[cls], label=cls, edgecolors="none")
        ax.axhline(0, color="grey", lw=0.6, ls="--"); ax.axvline(0, color="grey", lw=0.6, ls="--")
        if key == "pca_scores":                                 # Variance only applies to PCA
            ev = res["pca_evr"]
            ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}% variance)")
            ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}% variance)")
            ax.set_title(f"{lbl} — PCA(PC1+PC2)+LR LOGO accuracy = {acc*100:.1f}%", fontsize=10)
        else:
            ax.set_xlabel("LV1 (out-of-fold)"); ax.set_ylabel("LV2 (out-of-fold)")
            ax.set_title(f"{lbl} — LOGO accuracy = {acc*100:.1f}%", fontsize=10)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle(f"{title_prefix}\nquantitative peak-ratio parameters, "
                 f"250 spectra per animal, animal-bound validation", y=1.03, fontsize=11)
    fig.tight_layout()
    return fig


def _q2a_figure(df_bal, ratio_cols, label_col, id_col):
    """Scatter/strip panel: every spectrum plotted per ratio, WT beside OIM, with mean +/- SD."""
    n = len(ratio_cols)                                        # One panel per peak ratio
    ncols = min(3, max(1, n)); nrows = int(np.ceil(n / ncols)) # Compact grid
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.9 * nrows), squeeze=False)
    rng = np.random.RandomState(0)                             # Fixed jitter - reproducible figure
    colors = {"WT": "#2E86AB", "OIM": "#D7263D"}               # Same palette as every other figure

    for k, col in enumerate(ratio_cols):                       # Draw each ratio
        ax = axes[k // ncols][k % ncols]
        for xpos, cls in [(0, "WT"), (1, "OIM")]:              # Two groups side by side
            v = df_bal.loc[df_bal[label_col] == cls, col].to_numpy(dtype=float)
            v = v[np.isfinite(v)]                              # Drop any non-finite values
            if not len(v):
                continue
            ax.scatter(xpos + rng.uniform(-0.13, 0.13, len(v)), v,   # Jitter avoids overplotting
                       s=5, alpha=0.25, c=colors[cls], edgecolors="none")
            m, sd = v.mean(), v.std(ddof=1)                     # Mean and SD overlay
            ax.errorbar(xpos, m, yerr=sd, fmt="_", ms=34, mew=2.4,
                        c="black", capsize=8, elinewidth=1.6, zorder=5)
        # ROBUST Y-LIMITS, APPLIED ONLY WHEN GENUINELY NEEDED.
        # A peak ratio is a division, so when the denominator band is near zero a few spectra
        # produce enormous values - 1030/1020 reaches -73.9 here - which compresses every real
        # point into a flat line. But clipping at a fixed percentile would ALWAYS remove the same
        # 1% of points, and annotating that count would be meaningless: with 3000 spectra it reads
        # "30 extreme values" on every panel, including panels with no outliers at all.
        # Instead: measure how heavy the tails actually are, and only clip when they genuinely
        # dominate the axis. Whiskers are the standard Tukey 1.5*IQR fences.
        _allv = df_bal[col].to_numpy(dtype=float); _allv = _allv[np.isfinite(_allv)]
        _q1, _q3 = np.percentile(_allv, [25, 75]); _iqr = _q3 - _q1
        _flo, _fhi = _q1 - 1.5 * _iqr, _q3 + 1.5 * _iqr            # Tukey fences
        _inl = _allv[(_allv >= _flo) & (_allv <= _fhi)]            # The bulk of the distribution
        _full_span = _allv.max() - _allv.min()                     # What the axis would need
        _inl_span = (_inl.max() - _inl.min()) if len(_inl) else _full_span
        # Clip only if the extremes stretch the axis to more than 3x the span of the bulk.
        if _inl_span > 0 and _full_span > 3.0 * _inl_span:
            _pad = 0.10 * _inl_span                                # Breathing room around the bulk
            _lo, _hi = _inl.min() - _pad, _inl.max() + _pad        # The applied window
            _n_out = int(((_allv < _lo) | (_allv > _hi)).sum())    # Points ACTUALLY outside it
            ax.set_ylim(_lo, _hi)                                  # Apply
            ax.text(0.5, 0.97,
                    f"{_n_out} of {len(_allv)} extreme value(s) outside view "
                    f"(range {_allv.min():.3g} to {_allv.max():.3g})",
                    transform=ax.transAxes, ha="center", va="top", fontsize=7, color="#8a6d3b")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["WT", "OIM"])   # Group labels
        ax.set_title(str(col), fontsize=10)                    # The ratio name
        ax.set_ylabel("Ratio value", fontsize=9)               # Y-axis meaning
        ax.grid(axis="y", alpha=0.22)                          # Light guide lines
        ax.set_xlim(-0.55, 1.55)                               # A little breathing room

    for k in range(n, nrows * ncols):                          # Hide unused panels
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle("Conventional peak-ratio parameters, WT vs OIM\n"
                 "(balanced design: 250 spectra per animal, 12 animals; black bars = mean +/- SD)",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    return fig


st.sidebar.markdown("---")                        # Divider from the sections above
_q2a_show = st.sidebar.checkbox(                  # Opt-in, off by default
    "📊 Fig 2a — Conventional peak-ratio parameters (NEW)",
    value=False,
    help="The standard quantitative analysis panel: each peak ratio compared between WT and OIM, "
         "on the same balanced 250-spectra-per-animal dataset.")

if _q2a_show:
    st.markdown("---")
    st.header("📊 Fig 2a — Conventional peak-ratio parameters")
    st.caption("How this field presents bone FTIR before any multivariate modelling: each "
               "quantitative peak-ratio parameter compared directly between the two groups. "
               "Uses the same balanced design as every other figure, so it is directly comparable.")

    with st.sidebar:
        _q2a_up = st.file_uploader("Dataset (.xlsx)", type=["xlsx"], key="q2a_up")
        _q2a_n = int(st.number_input("Spectra per animal", 50, 2000, 250, 50, key="q2a_n"))
        _q2a_seed = int(st.number_input("Random seed", 0, 10000, 42, 1, key="q2a_seed"))
        _q2a_go = st.button("▶️ Generate Fig 2a", type="primary", key="q2a_go")

    if _q2a_go and _q2a_up is not None:
        _q2a_tmp = os.path.join(".", "_q2a_upload.xlsx")        # Loader needs a real path
        with open(_q2a_tmp, "wb") as _fh:
            _fh.write(_q2a_up.getbuffer())
        try:
            _q2a_raw = _zs_read_sheet(_q2a_tmp, "Quant peak ratios")     # Reuse the streaming reader
            _q2a_raw.columns = [str(c).strip() for c in _q2a_raw.columns]
            _idc, _lbc = _q2a_raw.columns[0], _q2a_raw.columns[1]
            _q2a_raw[_idc] = _q2a_raw[_idc].astype(str).str.strip()
            _q2a_raw[_lbc] = _q2a_raw[_lbc].astype(str).str.strip().str.upper()
            _q2a_raw = _q2a_raw[_q2a_raw[_idc].isin(_ZS_ALL_IDS)].copy()  # The 12 known animals
            _rat = [c for c in _q2a_raw.columns if c not in (_idc, _lbc)] # The five ratios
            _q2a_raw[_rat] = _q2a_raw[_rat].apply(pd.to_numeric, errors="coerce")
            _q2a_raw = _q2a_raw.dropna(subset=_rat).reset_index(drop=True)

            _parts = []                                          # Same balancing rule as elsewhere
            for _an, _g in _q2a_raw.groupby(_idc, sort=True):
                _rs = (_q2a_seed + int.from_bytes(
                    _zs_hashlib.blake2b(str(_an).encode("utf-8"), digest_size=4).digest(),
                    "big")) % (2**32 - 1)
                _parts.append(_g.sample(n=min(_q2a_n, len(_g)), random_state=_rs))
            _bal = pd.concat(_parts, ignore_index=True)
            st.success(f"{len(_bal):,} spectra · {_bal[_idc].nunique()} animals · "
                       f"{len(_rat)} peak ratios", icon="✅")

            _rows = []                                           # Per-ratio statistics table
            for _c in _rat:
                _st = _q2a_stats(_bal.loc[_bal[_lbc] == "WT", _c].to_numpy(float),
                                 _bal.loc[_bal[_lbc] == "OIM", _c].to_numpy(float))
                _st = {"Peak ratio": _c, **_st}
                _rows.append(_st)
            _stats_df = pd.DataFrame(_rows)

            _per_animal = (_bal.groupby([_lbc, _idc])[_rat].mean()      # Animal-level means: the
                           .reset_index())                             # statistically honest unit

            _fig2a = _q2a_figure(_bal, _rat, _lbc, _idc)                # Fig 2a panel

            # ---- Fig 2b + 2c: PCA and PLS-DA on the quantitative parameters ----
            _Xq = _bal[_rat].to_numpy(dtype=float)                      # The five ratio features
            _yq = (_bal[_lbc].to_numpy() == "OIM").astype(int)          # OIM = 1, WT = 0
            _gq = _bal[_idc].to_numpy()                                 # Animal = the CV group
            _res2bc = _q2bc_pca_plsda(_Xq, _yq, _gq, _rat, _q2a_seed)
            _fig2bc = _q2bc_figure(_res2bc, _yq, "Fig 2b/2c — PCA and PLS-DA scores")

            _cls = ["WT", "OIM"]                                        # Row/column order
            _pca_cm_df = pd.DataFrame(_res2bc["pca_cm"], index=[f"True {c}" for c in _cls],
                                      columns=[f"Pred {c}" for c in _cls]).reset_index()
            _pls_cm_df = pd.DataFrame(_res2bc["plsda_cm"], index=[f"True {c}" for c in _cls],
                                      columns=[f"Pred {c}" for c in _cls]).reset_index()
            _load_df = pd.DataFrame(_res2bc["pca_loadings"][:2].T, columns=["PC1 loading",
                                                                            "PC2 loading"])
            _load_df.insert(0, "Parameter", _rat)
            _vip_df = pd.DataFrame({"Parameter": _rat, "VIP": _res2bc["vip"]}).sort_values(
                "VIP", ascending=False)
            _scores_df = pd.DataFrame({"Animal": _gq,
                                       "Phenotype": np.where(_yq == 1, "OIM", "WT"),
                                       "PC1": _res2bc["pca_scores"][:, 0],
                                       "PC2": _res2bc["pca_scores"][:, 1],
                                       "PLSDA LV1 (OOF)": _res2bc["plsda_scores"][:, 0],
                                       "PLSDA LV2 (OOF)": _res2bc["plsda_scores"][:, 1]})
            _acc2bc = pd.DataFrame({
                "Quantity": ["PCA(PC1+PC2)+LR LOGO accuracy", "PLS-DA LOGO accuracy",
                             "PCA accuracy definition", "PLS-DA scores definition"],
                "Value": [_res2bc["pca_accuracy"], _res2bc["plsda_accuracy"],
                          "PCA itself is unsupervised; this is logistic regression on PC1+PC2 "
                          "under leave-one-animal-out",
                          "Out-of-fold: each animal projected by a model fitted on the other 11"]})

            _q2a_buf = io.BytesIO()                                     # Editable workbook
            with pd.ExcelWriter(_q2a_buf, engine="openpyxl") as _xw:
                pd.DataFrame({"Item": ["Generated", "Dataset", "Sheet", "Spectra per animal",
                                       "Random seed", "Figure", "Note"],
                              "Value": [datetime.now().strftime("%Y-%m-%d %H:%M"), _q2a_up.name,
                                        "Quant peak ratios", _q2a_n, _q2a_seed,
                                        "Fig 2a - conventional peak-ratio parameters",
                                        "Statistics across spectra are optimistic; the per-animal "
                                        "sheet treats the animal as the independent unit."]
                              }).to_excel(_xw, sheet_name="README", index=False)
                _stats_df.to_excel(_xw, sheet_name="Ratio statistics", index=False)
                _acc2bc.to_excel(_xw, sheet_name="Fig2bc accuracy", index=False)
                _scores_df.to_excel(_xw, sheet_name="Fig2bc scores", index=False)
                _load_df.to_excel(_xw, sheet_name="Fig2b PCA loadings", index=False)
                _vip_df.to_excel(_xw, sheet_name="Fig2c PLSDA VIP", index=False)
                _pca_cm_df.to_excel(_xw, sheet_name="Fig2b PCA CM", index=False)
                _pls_cm_df.to_excel(_xw, sheet_name="Fig2c PLSDA CM", index=False)
                _per_animal.to_excel(_xw, sheet_name="Per-animal means", index=False)
                _bal[[_idc, _lbc] + _rat].to_excel(_xw, sheet_name="All values", index=False)
                try:
                    # NATIVE EXCEL SCATTER CHARTS - the actual Fig 2a panel, editable in Excel.
                    # One sheet per ratio holding jittered x and the y values split by group, so
                    # each chart has a true WT series and OIM series that can be restyled.
                    from openpyxl.chart import ScatterChart, Reference, Series
                    _rng2 = np.random.RandomState(0)                # Fixed jitter - reproducible
                    for _c in _rat:
                        _sn = f"PLOT {_c}".replace("/", "-")[:31]   # '/' is illegal in sheet names
                        _wsp = _xw.book.create_sheet(_sn)
                        _wsp["A1"] = f"Fig 2a scatter data - {_c}"
                        _wsp["A2"] = ("Columns A/B are the WT points, C/D the OIM points. X values "
                                      "carry a small fixed jitter so the points do not overlap. "
                                      "Edit any value and the chart updates.")
                        for _j, _h in enumerate(["WT x", "WT value", "OIM x", "OIM value"], start=1):
                            _wsp.cell(row=4, column=_j, value=_h)
                        _vw = _bal.loc[_bal[_lbc] == "WT", _c].to_numpy(dtype=float)
                        _vo = _bal.loc[_bal[_lbc] == "OIM", _c].to_numpy(dtype=float)
                        _vw = _vw[np.isfinite(_vw)]; _vo = _vo[np.isfinite(_vo)]
                        _xw_ = 1 + _rng2.uniform(-0.16, 0.16, len(_vw))    # WT clustered around 1
                        _xo_ = 2 + _rng2.uniform(-0.16, 0.16, len(_vo))    # OIM clustered around 2
                        for _r in range(max(len(_vw), len(_vo))):
                            if _r < len(_vw):
                                _wsp.cell(row=5 + _r, column=1, value=float(_xw_[_r]))
                                _wsp.cell(row=5 + _r, column=2, value=float(_vw[_r]))
                            if _r < len(_vo):
                                _wsp.cell(row=5 + _r, column=3, value=float(_xo_[_r]))
                                _wsp.cell(row=5 + _r, column=4, value=float(_vo[_r]))
                        _sc = ScatterChart(); _sc.style = 13
                        _sc.title = f"{_c} - WT vs OIM"
                        _sc.x_axis.title = "Group (1 = WT, 2 = OIM)"; _sc.y_axis.title = "Ratio value"
                        for _col, _nm in ((1, "WT"), (3, "OIM")):   # Two independent series
                            _ser = Series(Reference(_wsp, min_col=_col + 1, min_row=5,
                                                    max_row=4 + max(len(_vw), len(_vo))),
                                          Reference(_wsp, min_col=_col, min_row=5,
                                                    max_row=4 + max(len(_vw), len(_vo))),
                                          title=_nm)
                            _ser.marker.symbol = "circle"; _ser.marker.size = 4
                            _ser.graphicalProperties.line.noFill = True    # Points only, no line
                            _sc.series.append(_ser)
                        _sc.height, _sc.width = 11, 16
                        _wsp.add_chart(_sc, "G4")

                    from openpyxl.chart import BarChart
                    _ws = _xw.book["Per-animal means"]; _last = _ws.max_row
                    for _j, _c in enumerate(_rat):
                        _ch = BarChart(); _ch.type = "col"; _ch.style = 10
                        _ch.title = f"{_c} - mean per animal"
                        _ch.y_axis.title = "Ratio value"; _ch.x_axis.title = "Animal"
                        _ch.add_data(Reference(_ws, min_col=3 + _j, min_row=1, max_row=_last),
                                     titles_from_data=True)
                        _ch.set_categories(Reference(_ws, min_col=2, min_row=2, max_row=_last))
                        _ch.height, _ch.width = 8, 16
                        _ws.add_chart(_ch, f"J{2 + _j * 16}")
                except Exception as _ce:
                    print("Fig 2a chart skipped:", _ce)

            _q2a_zip = io.BytesIO()                                     # PNG + editable SVG
            _q2a_png = io.BytesIO(); _q2bc_png = io.BytesIO()           # Also kept for display
            _fig2a.savefig(_q2a_png, format="png", dpi=300, bbox_inches="tight", facecolor="white")
            _fig2bc.savefig(_q2bc_png, format="png", dpi=300, bbox_inches="tight", facecolor="white")
            with _zs_zipfile.ZipFile(_q2a_zip, "w", _zs_zipfile.ZIP_DEFLATED) as _zf:
                _zf.writestr("Fig2a_PeakRatios.png", _q2a_png.getvalue())
                _v = io.BytesIO(); _fig2a.savefig(_v, format="svg",
                                                  bbox_inches="tight", facecolor="white")
                _zf.writestr("Fig2a_PeakRatios.svg", _v.getvalue())
                _zf.writestr("Fig2bc_PCA_PLSDA_QuantParams.png", _q2bc_png.getvalue())
                _v2 = io.BytesIO(); _fig2bc.savefig(_v2, format="svg",
                                                    bbox_inches="tight", facecolor="white")
                _zf.writestr("Fig2bc_PCA_PLSDA_QuantParams.svg", _v2.getvalue())
            plt.close(_fig2a); plt.close(_fig2bc)                       # Release both figures

            # PERSIST: Streamlit re-runs this whole script on every widget interaction, and a
            # download click is one. Anything rendered inside "if button:" therefore vanishes the
            # moment a file is downloaded. Storing rendered bytes in session_state means downloads
            # re-display the same results instead of forcing a recomputation.
            st.session_state["q2a_results"] = {
                "png": _q2a_png.getvalue(), "png2bc": _q2bc_png.getvalue(),
                "acc2bc": _acc2bc, "vip2bc": _vip_df,
                "zip": _q2a_zip.getvalue(),
                "xlsx": _q2a_buf.getvalue(), "stats": _stats_df,
                "per_animal": _per_animal, "n": len(_bal),
                "animals": int(_bal[_idc].nunique()), "ratios": len(_rat),
                "stamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            }
        except Exception as _q2a_err:
            st.error("Could not generate Fig 2a."); st.exception(_q2a_err)
        finally:
            if os.path.exists(_q2a_tmp):
                os.remove(_q2a_tmp)
    elif _q2a_go:
        st.error("Upload the dataset workbook first.", icon="🚫")

    # ---- Persistent results block, deliberately OUTSIDE the "Generate" branch above ----
    _q2a_saved = st.session_state.get("q2a_results")     # Results from the most recent run
    if _q2a_saved:                                       # Re-displayed on every rerun
        st.markdown("---")
        st.subheader("Fig 2a results")
        st.caption(f"{_q2a_saved['n']:,} spectra · {_q2a_saved['animals']} animals · "
                   f"{_q2a_saved['ratios']} peak ratios · run {_q2a_saved['stamp']}")
        st.info("Held in the session, so downloading a file no longer clears the results.",
                icon="💾")

        _qc1, _qc2, _qc3 = st.columns([2, 2, 1])         # Downloads first, always reachable
        with _qc1:
            st.download_button("📥 Data + statistics (editable .xlsx)",
                               data=_q2a_saved["xlsx"],
                               file_name=f"Fig2a_PeakRatios_{_q2a_saved['stamp']}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument."
                                    "spreadsheetml.sheet",
                               key="q2a_dl_xlsx", use_container_width=True)
        with _qc2:
            st.download_button("📦 Figure (PNG + editable SVG)", data=_q2a_saved["zip"],
                               file_name=f"Fig2a_PeakRatios_{_q2a_saved['stamp']}.zip",
                               mime="application/zip", key="q2a_dl_zip",
                               use_container_width=True)
        with _qc3:
            if st.button("🗑️ Clear", key="q2a_clear", use_container_width=True):
                del st.session_state["q2a_results"]      # Discard and start fresh
                st.rerun()

        _q2a_w = int(st.slider("Figure width on screen (px)", 300, 1600, 1000, 50,
                               key="q2a_width",
                               help="Display only - downloads are full 300 dpi / vector."))
        st.markdown("#### Fig 2a — conventional peak-ratio parameters")
        st.image(_q2a_saved["png"], width=_q2a_w)        # The 2a panel
        if _q2a_saved.get("png2bc") is not None:         # Fig 2b/2c, when present
            st.markdown("#### Fig 2b / 2c — PCA and PLS-DA on the quantitative parameters")
            st.image(_q2a_saved["png2bc"], width=_q2a_w)
            st.dataframe(_q2a_saved["acc2bc"], width="stretch", hide_index=True)
            st.caption("VIP over the five quantitative parameters:")
            st.dataframe(_q2a_saved["vip2bc"].round(4), width="stretch", hide_index=True)
        st.dataframe(_q2a_saved["stats"].round(4), width="stretch", hide_index=True)
        st.warning("Statistics above are computed across SPECTRA. With 12 animals the independent "
                   "unit is the ANIMAL, not the spectrum, so these p-values are optimistic. The "
                   "'Per-animal means' sheet in the workbook gives the honest comparison.",
                   icon="⚠️")
        with st.expander("Per-animal means (the statistically independent unit)", expanded=False):
            st.dataframe(_q2a_saved["per_animal"].round(4), width="stretch", hide_index=True)
# =============================================================================================
# ################  END SECTION: FIG 2a  ######################################################
# =============================================================================================
