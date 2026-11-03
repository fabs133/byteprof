# Colab notebook

- Used once for a Linux cross-check of the Windows measurements (does WDDM launch latency explain part of the decode cost?). See `FINDINGS.md`, section "Correction", for the context of the profile it re-runs.
- It was run once, ran several times slower than the local GPU, and is not maintained.
- Regenerate it from the repository root with `python tools/build_colab_notebook.py` (writes `notebooks/colab/colab_profile.ipynb`).
