# Peptide Screening Colab Utilities

Google Colab-ready notebooks and helper scripts for peptide and protein-structure workflows.

The main screening notebook is experimental and modular: Boltz-2 is used as an active-site structural filter, then site-directed docking and MM/GBSA-like rescoring are used to build a final ranking.

## Contents

- `ache_peptide_screening_boltz2_lightdock_mmgbsa.ipynb`: active hAChE peptide-screening notebook. It reads peptide sequences, maps hAChE active-site residues, runs Boltz-2-style protein-peptide cofolding against PAS/anionic/CAS site definitions, prepares LightDock docking around the selected site, performs OpenMM GBSA rescoring, and exports a weighted ranking table.
- `protenix_colab_template.ipynb`: standalone Protenix notebook for Google Colab. It installs Protenix in an isolated environment, prepares protein-peptide JSON inputs, optionally runs predictions on a GPU runtime, inventories output files and exports the run folder.
- `scripts/build_screening_notebook.py`: reproducible generator for the screening notebook.

## Screening Notebook Scope

The screening notebook follows this logic:

1. Read `peptide_id,sequence` candidates from CSV.
2. Prepare hAChE chain A and map functional residues by human AChE numbering.
3. Use Boltz-2 cofolding as a structural filter for PAS, anionic subsite and CAS engagement.
4. Dock peptides that pass the filter with LightDock around the selected site.
5. Rescore poses with a single-frame OpenMM GBSA approximation.
6. Combine Boltz geometry/confidence, docking geometry and MM/GBSA-like energy into a final ranking.

Boltz-2 outputs should be treated as structural hypotheses, not final affinity evidence. The GBSA stage is a practical Colab rescoring layer, not a replacement for a fully equilibrated Amber `MMPBSA.py` ensemble protocol.

## Protenix Notebook Scope

The Protenix notebook is a technical implementation example, not a ranking workflow. Its predicted complexes should be treated as exploratory structural hypotheses that require manual inspection and independent validation before they inform synthesis or screening decisions.

The notebook intentionally omits the previous docking, PAS-filter, HADDOCK, LightDock and MM/GBSA ranking sections. Those steps are not maintained here as part of the current screening strategy.

## Running In Colab

1. Open `ache_peptide_screening_boltz2_lightdock_mmgbsa.ipynb` in Google Colab.
2. Select `Runtime > Change runtime type > GPU`.
3. Run the install cell.
4. Replace the demo peptide CSV with your own inputs.
5. Generate Boltz-2 YAML jobs and inspect them.
6. Enable `RUN_BOLTZ2`, then inspect the active-site filter table.
7. Enable `RUN_DOCKING` and `RUN_MMGBSA` only after the upstream outputs look reasonable.

Screening outputs are written under `/content/hache_peptide_screening_boltz2` and can be exported as `hache_peptide_screening_boltz2_results.zip`.
