# BoneSpec AI — User Guide

## AI-Powered Infrared Spectral Analysis of Bone

**Live application:** https://bonespec-ai.streamlit.app/

### Quick start
1. Open the live app and select **Start Analysis**.
2. Select **Upload your own Excel file**, then choose an `.xlsx` workbook. An **example dataset** option appears automatically if an approved sample workbook is installed in `sample_data/bone_spectra_example.xlsx`.
3. Explore the available preprocessing, PCA, PLS-DA, classification, and SHAP sections. Some advanced sections are opt-in from the sidebar and may request a workbook separately.
4. For grouped validation, enable **Sample-ID Grouped Cross-Validation** in the sidebar, upload your workbook there, choose the worksheet, model(s), random seed, and number of spectra per group, and click **Run validation**.
5. Download the available figures and Excel/CSV results from each section.

### Excel input requirements
For grouped-validation workflows, use these worksheet names:

| Worksheet | Contents |
| --- | --- |
| `Raw` | Original spectral measurements |
| `2nd derivative` | Second-derivative spectral measurements |
| `Quant peak ratios` | Numeric spectral peak-ratio features (optional for some workflows) |

**Raw and 2nd derivative:** One row per spectrum. Include `Sample ID` (independent specimen identifier), `Type` (class label), and numeric spectral wavenumber columns, for example `800`, `802`, `804`, ... `1800`. Wavenumber headers should be numeric or numeric-like strings; the values should be numeric spectral intensities.

| Sample ID | Type | 800 | 802 | 804 |
| --- | --- | ---: | ---: | ---: |
| Sample_A | WT | 0.21 | 0.24 | 0.27 |
| Sample_A | WT | 0.22 | 0.25 | 0.28 |
| Sample_B | OIM | 0.30 | 0.33 | 0.35 |

*The table above is an illustrative schema, not experimental data.*

**Quant peak ratios:** Include `Sample ID`, `Type`, and numeric feature columns. Consistent class labels and nonempty sample IDs are required. Each independent specimen should retain the same class label across all its spectra.

### Grouped validation
The grouped validation module uses Leave-One-Group-Out (LOGO) cross-validation so that spectra from the same Sample ID are not split between training and held-out testing. Select the number of spectra sampled per group and a fixed random seed for reproducibility. If a group has insufficient usable spectra, the run may fail or need different settings.

### Important limitations
- Several legacy workflows were built around WT/OIM binary labels and particular bone spectral regions; the platform is not yet fully generic for arbitrary class labels, wavenumber ranges, or workbook structures.
- Large workbooks, repeated model fits, and SHAP computations can be slow on Streamlit Community Cloud.
- Uploaded data are processed by the hosted application. Do not upload sensitive or confidential data unless authorized.
- The public repository intentionally does **not** bundle unpublished experimental research data. A shareable demonstration dataset should be added only after approval.

### Troubleshooting
- **Missing sheets:** Check worksheet names exactly, especially `Raw`, `2nd derivative`, and `Quant peak ratios`.
- **Missing group or class columns:** Use `Sample ID` and `Type`.
- **No usable spectral columns:** Use numeric wavenumber headers and numeric values in the spectral columns.
- **Long-running analyses:** Begin with one model and one worksheet, and run SHAP only when needed.

For code, installation instructions and updates, see the [GitHub repository](https://github.com/wavhalkomal/bonespec-ai).
