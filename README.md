# BoneSpec AI

### AI-Powered Infrared Spectral Analysis of Bone

**BoneSpec AI** is an interactive, Python-based platform for exploring bone infrared spectral data using chemometrics, machine learning, grouped cross-validation, and interpretable AI. The Streamlit interface provides visualizations and downloadable analysis outputs without requiring users to write Python.

**[Launch the application](https://bonespec-ai.streamlit.app/)** · **[User guide and Excel format](USER_GUIDE.md)**

## Features

- **Spectral data exploration:** Load compatible Excel workbooks and inspect spectral measurements and metadata.
- **Preprocessing and spectral regions:** Analyze raw and second-derivative spectra, select predefined fingerprint and subregions, and explore quantitative peak-ratio features.
- **Chemometrics:** Principal component analysis (PCA), partial least squares discriminant analysis (PLS-DA), and variable importance in projection (VIP).
- **Machine learning:** Compare logistic regression, support vector machines, k-nearest neighbors, random forests, and XGBoost.
- **Grouped validation:** Leave-One-Group-Out (LOGO) cross-validation using specimen/sample IDs, with reproducible sampling and training-fold preprocessing to reduce within-specimen leakage.
- **Explainable AI:** Wavenumber-level and spectral-zone SHAP visualizations for supported workflows.
- **Exports:** Download supported plots, metrics, and Excel/CSV reports.

## Getting started

### Use online

Open **https://bonespec-ai.streamlit.app/**, select **Start Analysis**, and upload a compatible Excel file. Follow the in-app guide for individual workflows. An example dataset option will appear when an approved shareable workbook is added to the application.

### Run locally

Python **3.12** is recommended for the current deployment environment.

```bash
git clone https://github.com/wavhalkomal/bonespec-ai.git
cd bonespec-ai

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On Windows, activate the environment using `.venv\Scripts\activate` before installing dependencies.

## Excel input format

For **Sample-ID Grouped Cross-Validation**, provide an `.xlsx` workbook containing these worksheet names:

| Worksheet | Expected data |
| --- | --- |
| `Raw` | Raw spectra, one spectrum per row |
| `2nd derivative` | Second-derivative spectra in the same tabular structure |
| `Quant peak ratios` | Optional quantitative spectral features for relevant analyses |

Spectral worksheets should include:

- **`Sample ID`**: Independent specimen/group identifier, repeated across spectra from that specimen.
- **`Type`**: Class label associated with each spectrum (consistent within a specimen).
- **Wavenumber columns**: Numeric or numeric-like headers (e.g., `800`, `802`, `804`) with numeric spectral values.

Illustrative structure (example values are not experimental observations):

| Sample ID | Type | 800 | 802 | 804 |
| --- | --- | ---: | ---: | ---: |
| Sample_A | WT | 0.21 | 0.24 | 0.27 |
| Sample_A | WT | 0.22 | 0.25 | 0.28 |
| Sample_B | OIM | 0.30 | 0.33 | 0.35 |

The grouped-validation module accepts `Sample ID` / `Animal ID` and `Type` / `Class` / `Label` aliases. Some other sections have stricter worksheet and column assumptions. See the **[full user guide](USER_GUIDE.md)** before preparing a new dataset.

## Validation methodology

For datasets with multiple spectra from the same specimen, random spectrum-level splitting can place correlated measurements in both training and testing sets. The optional **Sample-ID Grouped Cross-Validation** module uses **Leave-One-Group-Out** folds: each fold holds out all spectra from one sample ID, trains on the remaining groups, and generates held-out predictions. The module also supports a fixed random seed, per-group spectral sampling, and evaluation with classification metrics and confusion matrices.

For supported models, preprocessing that requires fitting (such as standardization) is fitted on training groups inside each fold. The app also provides PCA, PLS-DA, and SHAP-based interpretation tools.

## Technology

Python · Streamlit · pandas · NumPy · SciPy · scikit-learn · XGBoost · SHAP · Plotly · Matplotlib · Seaborn · openpyxl · XlsxWriter

**Main entry point:** `app.py`  
**Dependencies:** `requirements.txt`  
**Application settings:** `.streamlit/config.toml`  
**User documentation:** `USER_GUIDE.md`

## Scope and limitations

This is research software, **not a clinical diagnostic tool**. The current application includes legacy workflows designed around specific bone spectral regions, workbook layouts, and WT/OIM class labels. While the grouped-validation module reads sample IDs from input data, not every feature is yet dataset-agnostic. Validate the expected inputs and model behavior before applying it to a different dataset. Large model-comparison and SHAP jobs can be slow on free hosting.

The repository does not distribute the unpublished study dataset or manuscript findings. Research-specific experimental design, results, and biological interpretations belong in the associated paper.

## Publication and citation

A citation for the associated research paper will be added once it is available.

---

Developed by **Komal Wavhal** in collaboration with **William Querido, PhD**, Stevens Institute of Technology.
