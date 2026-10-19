# Explainable Machine Learning for Infrared Spectral Analysis of Bone

### An interactive Python research platform for vibrational spectroscopy, chemometrics, machine learning, and chemically interpretable AI

**Research:** Stevens Institute of Technology — Department of Biomedical Engineering and Department of Computer Science  
**Researchers:** Komal Wavhal and William Querido, PhD  
**Technology:** Python · Streamlit · NumPy · pandas · scikit-learn · XGBoost · SHAP · Plotly · Matplotlib  
**Status:** Research software / deployment preparation. Public hosted application URL pending verification and research-team approval.

> **Research scope:** This platform supports exploratory and reproducible analysis of infrared spectra from bone specimens. It is not a clinical diagnostic device, does not establish human disease biomarkers, and has not been validated for patient-level predictions.

---

## 1. Project overview

Bone strength depends on more than bone quantity. The molecular composition and organization of the collagen-rich organic matrix and apatite mineral contribute to tissue quality and mechanical integrity. Infrared (IR) spectroscopy measures molecular vibrational signatures associated with these constituents and can reveal compositional differences that are difficult to capture using only a few predefined measurements.

**The problem:** Traditional bone IR analysis frequently compresses an entire spectrum into a handful of peak-intensity ratios. Although these parameters are biochemically meaningful, they discard information in spectral shapes, neighboring wavenumbers, correlated bands, and broad molecular signatures.

**Our approach:** Build a Python-based interactive analytical platform that retains and examines the broader IR spectral fingerprint. The software combines data visualization, spectral preprocessing and selection, conventional peak-ratio analysis, multivariate chemometrics, supervised machine learning, animal-grouped validation, and explainable AI.

The associated study investigates whether full-spectrum analysis can distinguish **wild-type (WT)** mouse bone, representing a typical bone phenotype, from **osteogenesis imperfecta model (OIM)** mouse bone, representing a fragile bone phenotype. WT/OIM is an experimentally defined contrast used to test the analytical framework; the study is **not** intended to establish osteogenesis imperfecta-specific diagnostic biomarkers.

### Research questions

1. Does the full IR fingerprint capture phenotype-associated information that five conventional peak ratios miss?
2. Are collagen-rich or mineral-rich spectral regions more informative?
3. How do raw and second-derivative spectra compare?
4. Does model complexity matter as much as spectral representation?
5. Which chemically meaningful spectral regions drive predictions, and in which direction?

---

## 2. Experimental dataset and study design

The manuscript reports a reanalysis of previously collected **optical photothermal infrared (O-PTIR)** spectra from mouse tibial cortical bone.

| Property | Research dataset |
| --- | --- |
| Biological groups | Wild-type (WT) and osteogenesis imperfecta model (OIM) |
| Animals | 6 WT + 6 OIM = **12 animals** |
| Animal characteristics | Female, 26 weeks old |
| Tissue | Tibial mid-diaphysis cortical bone |
| Measurement modality | O-PTIR spectroscopy |
| Spectral acquisition | Linescans across cortex, endosteal to periosteal |
| Acquisition detail | Three linescans per sample; 500 nm sampling spacing |
| Balanced analytical dataset | **250 spectra per animal** |
| Total spectra in balanced analysis | **3,000 spectra** |
| Labels and grouping | Phenotype plus animal ID retained for every spectrum |
| Validation unit | **Animal**, not an individual spectrum |

The research reuses existing measurements; it did not involve new animal experiments. Spectral acquisition and sample preparation are described in the prior dataset publication cited by the manuscript.

**Why animal ID matters:** Spectra measured from the same mouse are correlated. Randomly splitting individual spectra across training and testing can leak animal-specific information and overstate generalization. The reported research therefore uses **leave-one-group-out (LOGO) cross-validation**, holding out one entire animal per fold.

> **Important distinction:** There are 3,000 spectral observations, but only 12 biological animals. The 3,000 spectra are **not** 3,000 independent biological replicates.

### Infrared spectral representations

| Representation | Wavenumber range | Purpose |
| --- | --- | --- |
| Full fingerprint | **1800–800 cm⁻¹** | Retain broad organic and mineral signatures |
| Collagen-rich | **1800–1200 cm⁻¹** | Focus on predominantly organic-matrix information |
| Mineral-rich | **1200–800 cm⁻¹** | Focus on predominantly mineral-associated information |
| Conventional peak ratios | Five derived parameters | Biochemically interpretable baseline |

Both **raw** and **second-derivative** spectral representations are compared. The second derivative helps resolve overlapping peaks, but improved visual separation does not necessarily improve classification.

---

## 3. Conventional bone compositional parameters

The conventional analysis uses inverted second-derivative peak intensities to calculate five established ratios:

| Parameter | Peak ratio (cm⁻¹) | Interpretation |
| --- | --- | --- |
| Mineral content | 1030 / 1650 | Mineral-to-organic matrix signature |
| Carbonate content | 875 / 1030 | Carbonate relative to phosphate |
| HPO₄ content | 1120 / 1030 | Acid phosphate signature; inversely associated with mineral maturity |
| Mineral crystallinity | 1030 / 1020 | Mineral structural organization |
| Collagen integrity | 1338 / 1550 | Collagen-related compositional signature |

These measures remain useful biochemical descriptors. The project's central argument is **not** that peak ratios are obsolete, but that they can be complemented by information retained in the full spectrum.

In the study, OIM bone showed higher mineral content and lower HPO₄ content at the animal level; other reported conventional parameters did not show significant between-group differences.

---

## 4. Platform capabilities

The repository contains a Streamlit application that brings several scientific analysis stages into one interface.

### A. Data import and exploration

- Import compatible spectroscopy Excel workbooks.
- Explore spectral data and associated metadata.
- Visualize spectral intensity against wavenumber.
- Compare WT and OIM spectra and inspect derived compositional measurements.
- Preserve phenotype and animal identifiers for grouped evaluation.

### B. Spectral preprocessing and representation

- Work with raw and second-derivative spectral representations.
- Select full-fingerprint, collagen-rich, or mineral-rich ranges.
- Examine chemically assigned spectral zones.
- Compare conventional peak-ratio features with high-dimensional spectra.

The underlying manuscript describes exclusion of spectra with substantial PMMA contamination and IQR-based outlier removal during preparation of the original analytical data. These steps should not be confused with a claim that every original laboratory preprocessing operation is reproduced automatically for arbitrary uploaded files.

### C. Chemometrics and dimensionality reduction

- **Principal component analysis (PCA)** for exploratory variation and visualization.
- **Partial least squares discriminant analysis (PLS-DA)** for supervised separation.
- **Variable importance in projection (VIP)** for identifying influential spectral variables.

In the reported research, PLS-DA used **three latent variables**.

### D. Machine-learning model comparison

The research benchmarks multiple complementary model families:

| Model | Modeling approach |
| --- | --- |
| Logistic regression (LR) | Linear classification |
| Support vector machine (SVM) | Margin/kernel-based classification |
| K-nearest neighbors (KNN) | Distance-based classification |
| Random forest (RF) | Tree ensemble |
| XGBoost (XGB) | Gradient-boosted tree ensemble |
| PLS-DA | Supervised chemometric latent-variable classifier |

Comparing these models helps separate the effect of **feature representation** from the effect of **classifier choice**.

### E. Animal-level validation and model evaluation

The manuscript's reported analytical workflow:

1. Balance the data to 250 spectra for each of the 12 animals.
2. Keep animal ID and WT/OIM phenotype with every observation.
3. Hold out **all spectra from one animal** for a LOGO validation fold.
4. Fit model preprocessing on training animals only when required; for logistic regression, `StandardScaler` is fitted within the training fold.
5. Train the model without any spectra from the held-out animal.
6. Predict the held-out animal's spectra.
7. Combine predictions from all folds into out-of-fold (OOF) results.
8. Compute classification metrics and confusion matrices.

The reported metrics include **accuracy, F1 score, ROC-AUC**, and row-normalized confusion matrices. These are performance estimates on held-out animals from the same study, **not external clinical validation**.

### F. Explainable AI: VIP and SHAP

Predictive accuracy alone does not explain *why* a spectral model distinguishes two phenotypes. The platform's interpretability work connects predictions to molecular regions:

- **VIP:** Identifies influential wavenumbers in PLS-DA.
- **SHAP:** Estimates feature contributions for logistic regression and XGBoost.
- **Spectral-zone-based SHAP:** Treats contiguous, chemically defined wavenumber groups as interpretable units instead of attributing every highly correlated neighboring wavenumber independently.
- **Directional explanations:** Show whether a zone contributes toward WT or OIM classification.

The manuscript's zone-based approach perturbs all wavenumbers in a zone together using corresponding training-set mean values. Explanations are generated within the animal-held-out framework. **Ten spectra per animal** were analyzed for the reported SHAP interpretation, totaling **120 explained spectra**.

### G. Visualizations and research exports

The application includes interactive visualizations and downloadable research outputs, including workflows for:

- Spectral and peak-ratio figures.
- PCA and PLS-DA visualizations.
- Model comparison summaries.
- Classification metrics and confusion matrices.
- VIP and SHAP interpretation.
- Excel workbooks, CSV tables, and figure exports where supported by the selected workflow.

The repository's application is a substantial research prototype. Individual functions and export paths should be validated against the live deployment before being described as production-tested.

---

## 5. Chemically meaningful spectral zones

The research uses defined spectral regions to connect model explanations to molecular vibrations:

| Zone | Range (cm⁻¹) | Assignment |
| --- | --- | --- |
| Amide I | 1700–1600 | C=O stretching |
| Amide II | 1600–1500 | N–H bending and C–N stretching |
| CH 1450 | 1490–1430 | CH₂/CH₃ bending |
| CH/CO₃ | 1430–1400 | Overlapping CH bending and carbonate vibrations |
| CH 1340 | 1350–1330 | CH wagging |
| Amide III | 1300–1200 | C–N stretching and N–H bending |
| HPO₄ | 1200–1100 | Acid phosphate vibration |
| PO₄ | 1100–930 | Phosphate stretching |
| CO₃ | 890–860 | Carbonate vibration |
| Unassigned gaps | Outside defined zones | Remaining portions of the analyzed fingerprint |

These groupings make explainability more useful to spectroscopy researchers by relating mathematical attributions to established chemical assignments.

---

## 6. Research findings

**The following are results reported in the accompanying research manuscript, not independently reproduced deployment benchmarks.**

| Analysis / representation | Reported classification accuracy |
| --- | ---: |
| PLS-DA — conventional five peak ratios | **63%** |
| PLS-DA — full raw fingerprint | **87%** |
| PLS-DA — collagen-rich raw spectrum | **85%** |
| PLS-DA — mineral-rich raw spectrum | **75%** |
| Logistic regression — full raw fingerprint | **90%** |
| SVM — full raw fingerprint | **88%** |
| KNN — full raw fingerprint | **82%** |
| Random forest — full raw fingerprint | **87%** |
| XGBoost — full raw fingerprint | **88%** |
| ML models — conventional peak ratios | **59–62%** |

### Key scientific insights

**1. Richer spectral representation improves discrimination.** Models using the full fingerprint outperform the five conventional peak-ratio features in this dataset.

**2. Collagen-associated information is particularly informative.** The collagen-rich region approaches full-fingerprint performance and consistently outperforms the mineral-rich region.

**3. More complex models are not necessarily better.** Logistic regression, PLS-DA, and tree-based methods show broadly similar performance when provided informative spectra. The representation of the data matters more than simply choosing a more complex classifier.

**4. Second-derivative spectra do not consistently improve predictive performance.** Although derivatives sharpen overlapping bands, the raw spectrum performs similarly or better in several comparisons.

**5. Predictions can be chemically interpreted.** VIP and spectral-zone SHAP highlight **Amide I**, **Amide II**, and **PO₄** as important contributors. Logistic regression exhibits broader attribution patterns, while XGBoost has more localized importance patterns; the Amide III region is also evident in XGBoost attribution.

Supplementary analyses report **F1 scores and ROC-AUC** across the same spectral representations and model families, supporting the overall classification-performance trends.

### Interpretation of the results

These results support the **analytical feasibility** of interpretable full-spectrum modeling for this specific WT/OIM bone dataset. They do **not** demonstrate a validated clinical test, a universal marker of skeletal fragility, or proven transfer to human bone.

---

## 7. End-to-end analytical workflow

```text
O-PTIR bone spectra + animal metadata
                 |
                 v
Data quality checks and prepared spectral dataset
                 |
                 v
Balanced sampling: 250 spectra × 12 animals
                 |
                 v
Choose spectral representation
  |-- Five conventional peak ratios
  |-- Full fingerprint (1800–800 cm⁻¹)
  |-- Collagen-rich (1800–1200 cm⁻¹)
  |-- Mineral-rich (1200–800 cm⁻¹)
  |-- Raw / second-derivative spectra
                 |
                 v
Exploration and chemometrics
  |-- Spectral visualization
  |-- PCA
  |-- PLS-DA + VIP
                 |
                 v
Animal-grouped leave-one-group-out validation
  |-- Training animals only
  |-- Hold out one complete animal
  |-- Fit preprocessing within training folds
  |-- Train LR / SVM / KNN / RF / XGBoost
                 |
                 v
Pooled out-of-fold predictions
  |-- Accuracy, F1, ROC-AUC
  |-- Confusion matrices
  |-- Model and spectral-range comparisons
                 |
                 v
Interpretability
  |-- VIP spectral profiles
  |-- Wavenumber-level SHAP
  |-- Chemically grouped spectral-zone SHAP
                 |
                 v
Interactive plots and downloadable research reports
```

---

## 8. Technical implementation

### Software stack

| Layer | Technology | Role |
| --- | --- | --- |
| Application UI | Streamlit | Interactive research workflows |
| Language | Python | Scientific analysis and application logic |
| Data manipulation | pandas, NumPy | Spectral tables and numerical arrays |
| Scientific computing | SciPy | Signal-processing and numerical utilities |
| Machine learning | scikit-learn | PCA, PLS-DA-related workflows, classifiers, validation, metrics |
| Gradient boosting | XGBoost | Ensemble classification |
| Explainability | SHAP | Feature and spectral-zone attribution |
| Visualization | Plotly, Matplotlib, Seaborn | Interactive and publication-oriented figures |
| Spreadsheet reporting | openpyxl, XlsxWriter | Excel analysis exports |
| Version control | Git / GitHub | Source management |
| Intended initial hosting | Streamlit Community Cloud | Browser-accessible research application |

### Repository entrypoint

```text
app_k_PROFESSOR_250_FULL_RESULTS_FINAL.py
```

The application is currently maintained primarily as a large Streamlit Python script. A future modularization may separate the UI, preprocessing, models, grouped validation, and explainability into individually tested packages; **that refactor is a future improvement, not a claim about the current codebase**.

### Repository configuration

```text
README.md
requirements.txt
runtime.txt
.gitignore
.streamlit/config.toml
DEPLOYMENT.md
app_k_PROFESSOR_250_FULL_RESULTS_FINAL.py
```

Additional source files or assets may be present in the repository. Some image references in the application (such as `assets/hero.jpg`, `assets/paper.png`, and `Background_1.jpg`) need to be checked during deployment.

---

## 9. Run locally

**Recommended environment:** Python 3.11.

Clone the repository:

```bash
git clone https://github.com/wavhalkomal/AI-Powered-Vibrational-Spectroscopy-for-Bone-Health_Research.git
cd AI-Powered-Vibrational-Spectroscopy-for-Bone-Health_Research
```

Create and activate a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

On Windows, activate with `.venv\\Scripts\\activate`.

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Launch the application:

```bash
streamlit run app_k_PROFESSOR_250_FULL_RESULTS_FINAL.py
```

Streamlit will provide a local browser URL, typically `http://localhost:8501`.

**Input data:** Use a compatible, authorized Excel workbook containing the spectral representations and metadata expected by the selected analysis. Different parts of the research application may require specific workbook sheets or columns; inspect the relevant upload instructions in the app. The research dataset is **not** automatically distributed by this README.

For deployment-specific instructions, see [DEPLOYMENT.md](DEPLOYMENT.md).

---

## 10. Streamlit Community Cloud deployment

The initial deployment strategy is **GitHub + Streamlit Community Cloud** to provide researchers with browser-based access without requiring local Python installation.

Use the following configuration:

| Setting | Value |
| --- | --- |
| Repository | `wavhalkomal/AI-Powered-Vibrational-Spectroscopy-for-Bone-Health_Research` |
| Branch | `main` |
| App file | `app_k_PROFESSOR_250_FULL_RESULTS_FINAL.py` |
| Dependencies | `requirements.txt` |

Deploy at [Streamlit Community Cloud](https://share.streamlit.io/).

**Current status:** The repository includes deployment preparation files, but a successful hosted build and full functional validation have not yet been confirmed. A public application URL will be added **only after deployment, testing, and approval**.

### Future engineering roadmap (not yet implemented)

Potential enhancements, depending on user demand and computational requirements:

- Refactor the scientific analysis engine into modular, independently tested Python components.
- Add automated tests for data schema, model reproducibility, animal-level grouping, and leakage prevention.
- Introduce reproducible dependency locking, release tags, and CI checks.
- Provide curated, permission-cleared demonstration datasets.
- Add asynchronous jobs and persistent experiment tracking for larger workloads.
- Consider Docker and AWS services only if hosting scale, security, compute, or storage needs justify the added complexity.

**AWS, Databricks, GenAI, and agentic workflows are not presented as existing platform capabilities.**

---

## 11. Scientific rigor and limitations

The platform and manuscript should be interpreted with these limitations in mind:

- **Small biological sample:** 12 animals total, despite 3,000 balanced spectral observations.
- **Correlated measurements:** Spectra within an animal are not independent biological replicates; animal-level LOGO validation is essential.
- **Dataset-specific results:** Reported accuracy reflects this particular WT/OIM experimental contrast.
- **No human validation:** No evidence is presented that the models work on human specimens or can diagnose human skeletal disease.
- **No independent external cohort:** Broader generalization across populations, conditions, instruments, or laboratories remains to be tested.
- **Modality transfer unverified:** The reported experiment uses O-PTIR; results cannot automatically be assumed to hold for FTIR or other modalities.
- **Model explanations are associations:** VIP and SHAP describe model attribution, not proof of biological causality.
- **Prototype status:** Successful public deployment and end-to-end automated tests remain to be verified.

These constraints are central to responsible interpretation of the research, not minor implementation details.

---

## 12. Research data, publication, and responsible use

The dataset analyzed in the associated manuscript was previously generated and reported by **Dev et al. (2024)**. The manuscript's data availability statement says that supporting data may be requested from the corresponding author on reasonable request.

**Do not assume that the full dataset, unpublished manuscript, supplementary files, or other laboratory materials are licensed for redistribution through this repository.** Obtain approval from the corresponding author and research team before publishing them.

The software is intended for scientific exploration and methodological research. It must not be used to make clinical diagnoses or treatment decisions.

### Publication

**Manuscript:** *Explainable machine learning for infrared spectral analysis of bone*  
**Authors in the supplied October 3, 2026 draft:** Komal Wavhal and William Querido  
**Institution:** Stevens Institute of Technology, Hoboken, New Jersey, USA  
**Publication status:** Manuscript draft; journal publication and DOI not yet supplied.

The manuscript acknowledges support from **NIH/NIAMS award R21AR082129**.

Author list, manuscript details, license, dataset permissions, and citation instructions should be updated following research-team approval and formal publication.

### Related scientific references

- Dev, I., et al. (2024). *Assessment of submicron bone tissue composition in plastic-embedded samples using optical photothermal infrared (O-PTIR) spectral imaging and machine learning.* **Journal of Structural Biology X**, 10, 100111.
- Contreras, J., et al. (2024). *Spectral Zones-Based SHAP/LIME: Enhancing Interpretability in Spectral Deep Learning Models Through Grouped Feature Analysis.* **Analytical Chemistry**, 96(39), 15588–15597.

---

## 13. Contributors and acknowledgements

**Komal Wavhal** — Computational analyses, interactive Python analytical platform development, data visualizations, and manuscript contributions.

**William Querido, PhD** — Study conception and supervision, data analysis and interpretation, manuscript writing and revision.

**Affiliation:** Stevens Institute of Technology, Departments of Biomedical Engineering and Computer Science.

**Research support:** NIH/NIAMS R21AR082129.

For questions about research data access or publication permissions, contact the corresponding author through the institution's official channels.

---

## 14. Project vision

The long-term goal is to provide a **reproducible, transparent, and interpretable research environment** for extracting chemically meaningful information from vibrational spectroscopy data. The immediate contribution is a unified workflow that allows researchers to compare traditional spectral metrics with multivariate and machine-learning approaches while maintaining appropriate animal-level validation.

**Core principle: Better spectral representations, rigorous validation, and chemically interpretable explanations matter more than model complexity alone.**
