# hAChE Peptide Screening Colab

Colab notebook for screening peptide candidates as human acetylcholinesterase (hAChE) inhibitors using Protenix-guided peptide conformers and modular docking.

## Contents

- `hAChE_peptide_screening_modular_docking.ipynb`: main notebook.

## Notes

The notebook is designed to keep running in current Google Colab environments even when optional docking engines are unavailable. In Python 3.12 Colab runtimes, ProDy-dependent docking tools such as LightDock and HADDOCK3 may fail to build, so the notebook skips those engines and continues with available Protenix/idealized-conformer workflow pieces.

For LightDock/HADDOCK3-specific runs, use a Python 3.10/3.11 environment. If a previous install attempt upgraded Colab packages such as `numpy`, `pandas`, or `protobuf`, restart the Colab runtime before rerunning the corrected install cell.
