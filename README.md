# hAChE Peptide Screening Colab

Colab notebook for screening peptide candidates as human acetylcholinesterase (hAChE) inhibitors using a PAS/no-PAS co-folding filter followed by docking/MM-GBSA ranking.

## Contents

- `hAChE_peptide_screening_modular_docking.ipynb`: main notebook.

## Notes

The notebook is designed to keep running in current Google Colab environments even when optional engines are unavailable. Co-folding outputs are treated as a stage-1 PAS localization filter, not as final affinity ranking evidence. Final prioritization is reserved for post-filter docking/rescoring metrics such as HADDOCK cluster score, MM-GBSA global energy, and Trp286 residue-level contribution.

For LightDock/HADDOCK3-specific runs, use a Python 3.10/3.11 environment. If a previous install attempt upgraded Colab packages such as `numpy`, `pandas`, or `protobuf`, restart the Colab runtime before rerunning the corrected install cell. The Protenix environment is created with `virtualenv` because Colab's Python 3.12 `venv`/`ensurepip` bootstrap can fail. Protenix also needs `ninja`/build tools so PyTorch can compile its CUDA layer normalization extension on first run.
