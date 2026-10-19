# Streamlit deployment guide

## App entrypoint

`app.py`

## Local launch

1. Install Python 3.11.
2. Create a virtual environment: `python3.11 -m venv .venv`
3. Activate it: `source .venv/bin/activate` (Windows: `.venv\\Scripts\\activate`)
4. Install dependencies: `python -m pip install -r requirements.txt`
5. Launch: `streamlit run app.py`

## Streamlit Community Cloud

1. Sign in to https://share.streamlit.io/ using GitHub.
2. Choose **Create app** > **Deploy a public app from GitHub**.
3. Repository: `wavhalkomal/AI-Powered-Vibrational-Spectroscopy-for-Bone-Health_Research`
4. Branch: `main`
5. Main file path: `app.py`
6. Deploy and inspect the build and application logs.
7. Upload a **non-sensitive, approved sample dataset** to exercise each workflow. Validate model outputs against locally verified results.

## Important release checks

- The repository is public. Obtain the research team's approval before distributing unpublished data, manuscripts, analysis outputs, or confidential material.
- Git ignore rules do not remove files already committed or published.
- Do not put credentials or private data in the repository. Configure secrets in Streamlit's app settings if needed.
- Community Cloud has resource and upload limits. The 200 MB upload configuration does **not** guarantee enough memory or compute for large ML workloads.
- This repository's deployment configuration is provided as a starting point; successful cloud build, end-to-end operation, and scientific reproducibility must be verified on the live app.
- Uploaded files are processed by a hosted third-party service; use only datasets authorized for that environment.
