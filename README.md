# hAChE Peptide Screening Colab

Colab notebook for screening peptide candidates as human acetylcholinesterase (hAChE) inhibitors using Protenix-guided peptide conformers and modular docking.

## Contents

- `hAChE_peptide_screening_modular_docking.ipynb`: main notebook.

## Notes

The notebook is designed to keep running in current Google Colab environments even when optional docking engines are unavailable. In Python 3.12 Colab runtimes, ProDy-dependent docking tools such as LightDock and HADDOCK3 may fail to build, so the notebook skips those engines and continues with available Protenix/idealized-conformer workflow pieces. Protenix is installed into `/content/protenix_env` so its heavy ML dependencies do not disturb the notebook kernel.

For LightDock/HADDOCK3-specific runs, use a Python 3.10/3.11 environment. If a previous install attempt upgraded Colab packages such as `numpy`, `pandas`, or `protobuf`, restart the Colab runtime before rerunning the corrected install cell. The Protenix environment is created with `virtualenv` because Colab's Python 3.12 `venv`/`ensurepip` bootstrap can fail. Protenix also needs `ninja`/build tools so PyTorch can compile its CUDA layer normalization extension on first run.
