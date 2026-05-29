# Colab Bioinformatics Utilities

Small collection of Google Colab-ready notebooks and scripts for peptide and protein-structure workflows.

This repository is **not** the active peptide-screening protocol. Notebooks here are kept as practical, reusable utilities that can be run, adapted or inspected independently.

## Contents

- `protenix_colab_template.ipynb`: standalone Protenix notebook for Google Colab. It installs Protenix in an isolated environment, prepares protein-peptide JSON inputs, optionally runs predictions on a GPU runtime, inventories output files and exports the run folder.

## Protenix Notebook Scope

The Protenix notebook is a technical implementation example, not a ranking workflow. Its predicted complexes should be treated as exploratory structural hypotheses that require manual inspection and independent validation before they inform synthesis or screening decisions.

The notebook intentionally omits the previous docking, PAS-filter, HADDOCK, LightDock and MM/GBSA ranking sections. Those steps are not maintained here as part of the current screening strategy.

## Running In Colab

1. Open `protenix_colab_template.ipynb` in Google Colab.
2. Select `Runtime > Change runtime type > GPU`.
3. Run the install cell.
4. Replace the demo target sequence and peptide CSV with your own inputs.
5. Generate the Protenix JSON files.
6. Set `RUN_PROTENIX = True` only after the JSON inputs look correct.

Outputs are written under `/content/protenix_colab_jobs` and can be exported as `protenix_colab_jobs.zip`.
