# hAChE Peptide Screening Colab

Colab notebook for screening peptide candidates as human acetylcholinesterase (hAChE) inhibitors using Protenix-guided peptide conformers and modular docking.

## Contents

- `hAChE_peptide_screening_modular_docking.ipynb`: main notebook.

## Notes

The notebook is designed to keep running in current Google Colab environments even when optional docking engines are unavailable. In Python 3.12 Colab runtimes, HADDOCK3/ProDy installation may fail, so the notebook skips HADDOCK3 and continues with available engines such as LightDock.

For HADDOCK3-specific runs, use a Python 3.10/3.11 environment.
