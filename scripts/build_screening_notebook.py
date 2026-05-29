from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ache_peptide_screening_boltz2_lightdock_mmgbsa.ipynb"


def md(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    }


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


cells = [
    md(
        r"""
# hAChE peptide screening: Boltz-2 site filter -> docking -> MM/GBSA ranking

This notebook implements the intended screening logic:

```text
peptide CSV
  -> sequence curation
  -> hAChE structure preparation and active-site residue mapping
  -> Boltz-2 protein-peptide cofolding against PAS / anionic subsite / CAS
  -> geometric + confidence filter for active-site engagement
  -> site-directed peptide docking with LightDock
  -> single-frame OpenMM GBSA rescoring
  -> weighted final ranking table
```

Interpretation rule: Boltz-2 is used as a structural hypothesis and active-site filter, not as final peptide affinity evidence. The final ranking is only meaningful when docking and MM/GBSA-like rescoring have run and the top poses have been inspected manually.

Default human AChE residue sets:

- `PAS`: Tyr72, Asp74, Tyr124, Trp286, Tyr341
- `ANIONIC_SUBSITE`: Trp86, Tyr133, Tyr337, Phe338
- `CAS`: Ser203, Glu334, His447
- `CATALYTIC_GORGE`: anionic subsite plus catalytic, oxyanion and acyl-pocket support residues
"""
    ),
    md(
        r"""
## 0. Install dependencies

Use `Runtime > Change runtime type > GPU` before installing Boltz-2. Boltz-2 is installed into `/content/boltz_env`; LightDock and OpenMM are installed in the notebook kernel.
"""
    ),
    code(
        r"""
%%bash
#@title Install Boltz-2, LightDock and OpenMM
set -u
export DEBIAN_FRONTEND=noninteractive

INSTALL_BOLTZ2=true
INSTALL_LIGHTDOCK=true
INSTALL_OPENMM_GBSA=true

apt-get update -qq || true
apt-get install -y -qq git wget curl zip build-essential ninja-build gawk bzip2
python -m pip -q install --upgrade pip wheel packaging virtualenv
python -m pip -q install 'numpy<2.2' pandas biopython pyyaml py3Dmol PeptideBuilder pdb-tools

if [ "$INSTALL_OPENMM_GBSA" = true ]; then
  python -m pip -q install openmm || echo "WARNING: OpenMM install failed."
fi

if [ "$INSTALL_LIGHTDOCK" = true ]; then
  # LightDock depends on ProDy, which often fails to build in Colab's Python 3.12.
  # Keep it in a Python 3.10 micromamba environment and call its scripts by path.
  mkdir -p /content/bin
  if [ ! -x /content/bin/micromamba ]; then
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /content/bin --strip-components=1 bin/micromamba
  fi
  /content/bin/micromamba create -y -p /content/lightdock_env -c conda-forge python=3.10 pip numpy scipy cython prody || echo "WARNING: micromamba environment creation failed."
  /content/lightdock_env/bin/python -m pip install lightdock==0.9.2post1 || echo "WARNING: LightDock install failed inside /content/lightdock_env."
fi

if [ "$INSTALL_BOLTZ2" = true ]; then
  BOLTZ_ENV=/content/boltz_env
  rm -rf "$BOLTZ_ENV"
  python -m virtualenv "$BOLTZ_ENV"
  "$BOLTZ_ENV/bin/python" -m pip -q install --upgrade pip setuptools wheel ninja
  "$BOLTZ_ENV/bin/python" -m pip install 'boltz[cuda]' -U || "$BOLTZ_ENV/bin/python" -m pip install boltz -U
fi

python - <<'PY'
from pathlib import Path
import importlib.util, shutil, sys
print('Notebook Python:', sys.version.split()[0])
for name, module in [('pandas','pandas'), ('Bio','Bio'), ('yaml','yaml'), ('py3Dmol','py3Dmol'), ('PeptideBuilder','PeptideBuilder'), ('openmm','openmm')]:
    print(f'{name:18s}:', 'OK' if importlib.util.find_spec(module) else 'missing')
ld_bin = Path('/content/lightdock_env/bin')
print('lightdock3_setup.py:', ld_bin / 'lightdock3_setup.py' if (ld_bin / 'lightdock3_setup.py').exists() else shutil.which('lightdock3_setup.py') or 'missing')
print('lightdock3.py      :', ld_bin / 'lightdock3.py' if (ld_bin / 'lightdock3.py').exists() else shutil.which('lightdock3.py') or 'missing')
print('lgd_generate       :', ld_bin / 'lgd_generate_conformations.py' if (ld_bin / 'lgd_generate_conformations.py').exists() else shutil.which('lgd_generate_conformations.py') or 'missing')
print('boltz             :', Path('/content/boltz_env/bin/boltz') if Path('/content/boltz_env/bin/boltz').exists() else shutil.which('boltz') or 'missing')
PY
"""
    ),
    md(
        r"""
## 1. Configuration

The default target is human AChE chain A from PDB `4EY7`. If you change the PDB ID or chain, verify that all residue mappings remain correct.
"""
    ),
    code(
        r"""
#@title Global configuration
from pathlib import Path
import glob, json, math, os, re, shutil, subprocess
import numpy as np
import pandas as pd
import yaml

ROOT = Path('/content/hache_peptide_screening_boltz2')
RAW = ROOT / 'raw'
INPUTS = ROOT / 'inputs'
BOLTZ_IN = ROOT / 'boltz_yaml'
BOLTZ_OUT = ROOT / 'boltz_outputs'
DOCKING_DIR = ROOT / 'docking'
MMGBSA_DIR = ROOT / 'mmgbsa_openmm'
REPORTS = ROOT / 'reports'
for p in [RAW, INPUTS, BOLTZ_IN, BOLTZ_OUT, DOCKING_DIR, MMGBSA_DIR, REPORTS]:
    p.mkdir(parents=True, exist_ok=True)

CONFIG = {
    'target_pdb_id': '4EY7',
    'target_chain': 'A',
    'model_receptor_chain': 'A',
    'model_peptide_chain': 'B',
    'active_sites_pdbnum': {
        'PAS': [72, 74, 124, 286, 341],
        'ANIONIC_SUBSITE': [86, 133, 337, 338],
        'CAS': [203, 334, 447],
        'CATALYTIC_GORGE': [86, 121, 122, 133, 203, 236, 295, 297, 334, 337, 338, 447],
    },
    'boltz_sites_to_run': ['PAS', 'ANIONIC_SUBSITE', 'CAS'],
    'include_unconstrained_boltz_run': True,
    'boltz_use_msa_server': False,
    'boltz_use_potentials': True,
    'boltz_force_pocket_constraint': False,
    'boltz_pocket_max_distance_A': 8.0,
    'boltz_method': 'boltz2',
    'boltz_diffusion_samples': 2,
    'boltz_recycling_steps': 3,
    'boltz_sampling_steps': 120,
    'boltz_output_format': 'pdb',
    'boltz_override': True,
    'boltz_pass_dmin_A': 8.0,
    'boltz_borderline_dmin_A': 12.0,
    'boltz_pass_contacts_5A': 3,
    'boltz_min_confidence_score': 0.35,
    'boltz_min_protein_iptm': 0.25,
    'max_peptides_for_docking': 20,
    'lightdock_swarms': 12,
    'lightdock_glowworms': 40,
    'lightdock_steps': 50,
    'lightdock_cores': 2,
    'lightdock_top_models_per_swarm': 3,
    'lightdock_score': 'fastdfire',
    'openmm_minimize': True,
    'openmm_minimize_max_iterations': 250,
    'openmm_ph': 7.4,
    'rank_weights': {
        'boltz_filter_score_norm': 0.30,
        'docking_score_norm': 0.30,
        'mmgbsa_dg_norm': 0.30,
        'site_contact_score_norm': 0.10,
    },
}

RUN_BOLTZ2 = False #@param {type:'boolean'}
RUN_DOCKING = False #@param {type:'boolean'}
RUN_MMGBSA = False #@param {type:'boolean'}

BOLTZ_CMD = str(Path('/content/boltz_env/bin/boltz')) if Path('/content/boltz_env/bin/boltz').exists() else shutil.which('boltz')
LIGHTDOCK_BIN = Path('/content/lightdock_env/bin')
LIGHTDOCK_SETUP = str(LIGHTDOCK_BIN / 'lightdock3_setup.py') if (LIGHTDOCK_BIN / 'lightdock3_setup.py').exists() else shutil.which('lightdock3_setup.py')
LIGHTDOCK_RUN = str(LIGHTDOCK_BIN / 'lightdock3.py') if (LIGHTDOCK_BIN / 'lightdock3.py').exists() else shutil.which('lightdock3.py')
LIGHTDOCK_GENERATE = str(LIGHTDOCK_BIN / 'lgd_generate_conformations.py') if (LIGHTDOCK_BIN / 'lgd_generate_conformations.py').exists() else shutil.which('lgd_generate_conformations.py')
if Path('/usr/local/cuda').exists() and not os.environ.get('CUDA_HOME'):
    os.environ['CUDA_HOME'] = '/usr/local/cuda'
    os.environ['PATH'] = '/usr/local/cuda/bin:' + os.environ.get('PATH', '')
    os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

TOOL_STATUS = {
    'boltz': BOLTZ_CMD is not None,
    'lightdock_setup': LIGHTDOCK_SETUP is not None,
    'lightdock_run': LIGHTDOCK_RUN is not None,
    'lightdock_generate': LIGHTDOCK_GENERATE is not None,
}
print('Workdir:', ROOT)
print('Boltz command:', BOLTZ_CMD or 'missing')
print('CUDA_HOME:', os.environ.get('CUDA_HOME', 'not set'))
print('Tool status:', TOOL_STATUS)
print('LightDock setup:', LIGHTDOCK_SETUP or 'missing')
print('LightDock run:', LIGHTDOCK_RUN or 'missing')
"""
    ),
    md(
        r"""
## 2. Peptide input

Replace `/content/hache_peptide_screening_boltz2/inputs/peptides.csv` with your own file. Required columns: `peptide_id`, `sequence`.
"""
    ),
    code(
        r"""
#@title Create example peptide CSV
example = pd.DataFrame({
    'peptide_id': ['pep_demo_LGWVSKGKLL', 'pep_demo_KLVFFAE'],
    'sequence': ['LGWVSKGKLL', 'KLVFFAE'],
    'notes': ['demo only', 'demo only'],
})
peptides_csv = INPUTS / 'peptides.csv'
example.to_csv(peptides_csv, index=False)
print('Replace this file for a real run:', peptides_csv)
example
"""
    ),
    code(
        r"""
#@title Validate peptide table
AA = set('ACDEFGHIKLMNPQRSTVWY')
HYDRO = {'A':1.8,'C':2.5,'D':-3.5,'E':-3.5,'F':2.8,'G':-0.4,'H':-3.2,'I':4.5,'K':-3.9,'L':3.8,'M':1.9,'N':-3.5,'P':-1.6,'Q':-3.5,'R':-4.5,'S':-0.8,'T':-0.7,'V':4.2,'W':-0.9,'Y':-1.3}

def clean_sequence(seq):
    return re.sub('[^A-Z]', '', str(seq).upper())

def safe_id(value):
    text = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value).strip())
    return text.strip('._-') or 'peptide'

def rough_charge(seq):
    return sum(seq.count(x) for x in 'KR') - sum(seq.count(x) for x in 'DE') + 0.1 * seq.count('H')

def mean_hydropathy(seq):
    vals = [HYDRO[a] for a in seq if a in HYDRO]
    return float(np.mean(vals)) if vals else np.nan

peptides = pd.read_csv(peptides_csv)
missing = {'peptide_id', 'sequence'} - set(peptides.columns)
if missing:
    raise ValueError(f'Missing required columns: {sorted(missing)}')
peptides['peptide_id'] = peptides['peptide_id'].map(safe_id)
peptides['sequence'] = peptides['sequence'].map(clean_sequence)
peptides['valid_natural_aa'] = peptides['sequence'].map(lambda s: bool(s) and set(s).issubset(AA))
peptides['length'] = peptides['sequence'].str.len()
peptides['rough_charge'] = peptides['sequence'].map(rough_charge)
peptides['hydropathy_mean'] = peptides['sequence'].map(mean_hydropathy)
peptides['ready_for_screening'] = peptides['valid_natural_aa'] & peptides['length'].between(4, 60)
if peptides['peptide_id'].duplicated().any():
    raise ValueError('Duplicate peptide_id values after cleaning; please make IDs unique.')
peptides.to_csv(REPORTS / '01_validated_peptides.csv', index=False)
peptides
"""
    ),
    md(
        r"""
## 3. Prepare hAChE receptor and residue mapping
"""
    ),
    code(
        r"""
#@title Download receptor, clean chain and map active residues
from Bio.PDB import PDBParser, MMCIFParser, PDBIO, Select
from Bio.PDB.Polypeptide import protein_letters_3to1_extended
import urllib.request

three_to_one = {k.upper(): v for k, v in protein_letters_3to1_extended.items()}
one_to_three = {'A':'ALA','C':'CYS','D':'ASP','E':'GLU','F':'PHE','G':'GLY','H':'HIS','I':'ILE','K':'LYS','L':'LEU','M':'MET','N':'ASN','P':'PRO','Q':'GLN','R':'ARG','S':'SER','T':'THR','V':'VAL','W':'TRP','Y':'TYR'}

def run_cmd(cmd, cwd=None, allow_fail=False):
    cmd = [str(x) for x in cmd]
    print('RUN:', ' '.join(cmd))
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout[-4000:])
    if result.stderr:
        print(result.stderr[-4000:])
    if result.returncode != 0 and not allow_fail:
        raise RuntimeError(f'Command failed with exit code {result.returncode}: {cmd}')
    return result

class CleanChainSelect(Select):
    def __init__(self, chain_id):
        self.chain_id = chain_id
    def accept_chain(self, chain):
        return chain.id == self.chain_id
    def accept_residue(self, residue):
        return residue.id[0] == ' ' and residue.get_resname().upper() in three_to_one
    def accept_atom(self, atom):
        return atom.element != 'H'

def download_pdb(pdb_id):
    out_path = RAW / f'{pdb_id.upper()}.pdb'
    if not out_path.exists():
        urllib.request.urlretrieve(f'https://files.rcsb.org/download/{pdb_id.upper()}.pdb', out_path)
    return out_path

def extract_chain_sequence_and_mapping(pdb_path, chain_id):
    structure = PDBParser(QUIET=True).get_structure('target', str(pdb_path))
    chain = structure[0][chain_id]
    seq, pdbnum_to_seqpos, seqpos_to_pdbnum, seqpos_to_resname = [], {}, {}, {}
    for residue in chain:
        if residue.id[0] != ' ':
            continue
        resname = residue.get_resname().upper()
        if resname not in three_to_one:
            continue
        seq.append(three_to_one[resname])
        seqpos = len(seq)
        pdbnum = int(residue.id[1])
        pdbnum_to_seqpos[pdbnum] = seqpos
        seqpos_to_pdbnum[seqpos] = pdbnum
        seqpos_to_resname[seqpos] = resname
    return ''.join(seq), pdbnum_to_seqpos, seqpos_to_pdbnum, seqpos_to_resname

pdb_path = download_pdb(CONFIG['target_pdb_id'])
target_seq, pdbnum_to_seqpos, seqpos_to_pdbnum, seqpos_to_resname = extract_chain_sequence_and_mapping(pdb_path, CONFIG['target_chain'])

receptor_clean = RAW / f"{CONFIG['target_pdb_id']}_{CONFIG['target_chain']}_clean_receptor.pdb"
structure = PDBParser(QUIET=True).get_structure('target', str(pdb_path))
io = PDBIO(); io.set_structure(structure)
io.save(str(receptor_clean), CleanChainSelect(CONFIG['target_chain']))

site_rows, active_sites_seqpos = [], {}
for site, pdbnums in CONFIG['active_sites_pdbnum'].items():
    active_sites_seqpos[site] = []
    for pdbnum in pdbnums:
        seqpos = pdbnum_to_seqpos.get(int(pdbnum))
        site_rows.append({'site': site, 'pdb_residue_number': pdbnum, 'boltz_sequence_position': seqpos})
        if seqpos is not None:
            active_sites_seqpos[site].append(seqpos)
site_map = pd.DataFrame(site_rows)
site_map.to_csv(REPORTS / '02_active_site_mapping.csv', index=False)
print('Downloaded PDB:', pdb_path)
print('Clean receptor:', receptor_clean)
print('Target sequence length:', len(target_seq))
display(site_map)
"""
    ),
    md(
        r"""
## 4. Generate and optionally run Boltz-2 YAML jobs

For each peptide this creates one unconstrained job plus one weak pocket-guided job for each requested site. Pocket contacts use Boltz sequence positions, not PDB numbering.
"""
    ),
    code(
        r"""
#@title Generate Boltz-2 YAML files
def protein_entry(chain_id, sequence, use_msa_server=False):
    entry = {'id': chain_id, 'sequence': sequence}
    if not use_msa_server:
        entry['msa'] = 'empty'
    return {'protein': entry}

def make_boltz_yaml(peptide_seq, site_name=None):
    doc = {
        'version': 1,
        'sequences': [
            protein_entry(CONFIG['model_receptor_chain'], target_seq, CONFIG['boltz_use_msa_server']),
            protein_entry(CONFIG['model_peptide_chain'], peptide_seq, False),
        ],
    }
    if site_name and site_name != 'UNCONSTRAINED':
        contacts = [[CONFIG['model_receptor_chain'], int(pos)] for pos in active_sites_seqpos.get(site_name, [])]
        doc['constraints'] = [{'pocket': {
            'binder': CONFIG['model_peptide_chain'],
            'contacts': contacts,
            'max_distance': float(CONFIG['boltz_pocket_max_distance_A']),
            'force': bool(CONFIG['boltz_force_pocket_constraint']),
        }}]
    return doc

sites_to_run = list(CONFIG['boltz_sites_to_run'])
if CONFIG['include_unconstrained_boltz_run']:
    sites_to_run = ['UNCONSTRAINED'] + sites_to_run

boltz_jobs = []
for _, row in peptides[peptides['ready_for_screening']].iterrows():
    for site_name in sites_to_run:
        job_name = f"{row['peptide_id']}__{site_name}"
        out_yaml = BOLTZ_IN / f'{job_name}.yaml'
        with open(out_yaml, 'w', encoding='utf-8') as f:
            yaml.safe_dump(make_boltz_yaml(row['sequence'], site_name), f, sort_keys=False)
        boltz_jobs.append({'peptide_id': row['peptide_id'], 'sequence': row['sequence'], 'site_tested': site_name, 'job_name': job_name, 'yaml': str(out_yaml)})
boltz_jobs = pd.DataFrame(boltz_jobs)
boltz_jobs.to_csv(REPORTS / '03_boltz_jobs.csv', index=False)
print(f'Wrote {len(boltz_jobs)} Boltz YAML jobs to {BOLTZ_IN}')
boltz_jobs.head(20)
"""
    ),
    code(
        r"""
#@title Run Boltz-2
if RUN_BOLTZ2 and not TOOL_STATUS['boltz']:
    raise RuntimeError('RUN_BOLTZ2=True but boltz is missing. Rerun installation.')

def boltz_help_text():
    if not BOLTZ_CMD:
        return ''
    result = subprocess.run([BOLTZ_CMD, 'predict', '--help'], text=True, capture_output=True)
    return (result.stdout or '') + '\n' + (result.stderr or '')

def build_boltz_cmd(input_path):
    help_text = boltz_help_text()
    cmd = [BOLTZ_CMD, 'predict', str(input_path), '--out_dir', str(BOLTZ_OUT)]
    if CONFIG['boltz_use_msa_server'] and '--use_msa_server' in help_text:
        cmd.append('--use_msa_server')
    if CONFIG['boltz_use_potentials'] and '--use_potentials' in help_text:
        cmd.append('--use_potentials')
    for opt, key in [('--diffusion_samples','boltz_diffusion_samples'),('--recycling_steps','boltz_recycling_steps'),('--sampling_steps','boltz_sampling_steps'),('--output_format','boltz_output_format')]:
        if opt in help_text:
            cmd += [opt, str(CONFIG[key])]
    if CONFIG.get('boltz_method') and '--method' in help_text:
        cmd += ['--method', str(CONFIG['boltz_method'])]
    if CONFIG['boltz_override'] and '--override' in help_text:
        cmd.append('--override')
    return cmd

if RUN_BOLTZ2:
    run_cmd(build_boltz_cmd(BOLTZ_IN))
else:
    print('RUN_BOLTZ2=False. YAML files are ready at:', BOLTZ_IN)
"""
    ),
    md(
        r"""
## 5. Score Boltz-2 models and select peptides for docking
"""
    ),
    code(
        r"""
#@title Boltz-2 geometry/confidence filter
from Bio.PDB import PDBParser, MMCIFParser, PDBIO, Select

def find_files(base_dir, suffixes):
    hits = []
    for suffix in suffixes:
        hits.extend(Path(base_dir).rglob(f'*{suffix}'))
    return sorted([p for p in hits if p.is_file()])

def load_structure_any(path):
    path = Path(path)
    if path.suffix.lower() in {'.cif', '.mmcif'}:
        return MMCIFParser(QUIET=True).get_structure(path.stem, str(path))
    return PDBParser(QUIET=True).get_structure(path.stem, str(path))

def heavy_atom_coords_chain(chain):
    coords = []
    if chain is None:
        return np.empty((0, 3))
    for residue in chain:
        if residue.id[0] != ' ':
            continue
        for atom in residue:
            if atom.element != 'H':
                coords.append(atom.coord)
    return np.array(coords, dtype=float) if coords else np.empty((0, 3))

def heavy_atom_coords_residues(chain, residue_numbers):
    coords, wanted = [], set(int(x) for x in residue_numbers if x is not None)
    if chain is None:
        return np.empty((0, 3))
    for residue in chain:
        if residue.id[0] != ' ' or int(residue.id[1]) not in wanted:
            continue
        for atom in residue:
            if atom.element != 'H':
                coords.append(atom.coord)
    return np.array(coords, dtype=float) if coords else np.empty((0, 3))

def min_distance(A, B):
    if A.size == 0 or B.size == 0:
        return np.nan
    diff = A[:, None, :] - B[None, :, :]
    return float(np.sqrt((diff * diff).sum(axis=2)).min())

def count_contacts(A, B, cutoff=5.0):
    if A.size == 0 or B.size == 0:
        return 0
    diff = A[:, None, :] - B[None, :, :]
    d2 = (diff * diff).sum(axis=2)
    return int((d2 <= cutoff * cutoff).sum())

def score_structure_against_sites(structure_file, site_residue_numbers, rec_chain_id='A', pep_chain_id='B'):
    structure = load_structure_any(structure_file)
    model = next(structure.get_models())
    chains = list(model.get_chains())
    rec = model[rec_chain_id] if rec_chain_id in model else (chains[0] if chains else None)
    pep = model[pep_chain_id] if pep_chain_id in model else (chains[1] if len(chains) > 1 else None)
    pep_coords = heavy_atom_coords_chain(pep)
    row = {}
    for site, residues in site_residue_numbers.items():
        site_coords = heavy_atom_coords_residues(rec, residues)
        row[f'dmin_{site}_A'] = min_distance(pep_coords, site_coords)
        row[f'contacts_{site}_5A'] = count_contacts(pep_coords, site_coords, cutoff=5.0)
    return row

def parse_confidence_json(prediction_dir):
    files = list(Path(prediction_dir).glob('confidence*.json'))
    if not files:
        return {}
    try:
        return json.loads(files[0].read_text())
    except Exception:
        return {}

def locate_prediction_dir(job_name):
    for c in [BOLTZ_OUT / 'predictions' / job_name, BOLTZ_OUT / job_name]:
        if c.exists():
            return c
    matches = [p for p in BOLTZ_OUT.rglob(job_name) if p.is_dir()]
    return matches[0] if matches else None

def norm_0_100(values, higher_is_better=True):
    s = pd.to_numeric(values, errors='coerce')
    out = pd.Series(np.nan, index=s.index, dtype=float)
    if s.notna().sum() == 0:
        return out
    lo, hi = float(s.min()), float(s.max())
    if math.isclose(lo, hi):
        out.loc[s.notna()] = 50.0
        return out
    z = (s - lo) / (hi - lo)
    if not higher_is_better:
        z = 1 - z
    return 100 * z

rows = []
for _, job in boltz_jobs.iterrows():
    pred_dir = locate_prediction_dir(job['job_name'])
    if pred_dir is None:
        rows.append({**job.to_dict(), 'prediction_dir': '', 'model_file': '', 'missing_output': True})
        continue
    model_files = find_files(pred_dir, ['.pdb', '.cif', '.mmcif'])
    if not model_files:
        rows.append({**job.to_dict(), 'prediction_dir': str(pred_dir), 'model_file': '', 'missing_output': True})
        continue
    conf = parse_confidence_json(pred_dir)
    for model_file in model_files:
        geom = score_structure_against_sites(model_file, active_sites_seqpos, CONFIG['model_receptor_chain'], CONFIG['model_peptide_chain'])
        tested = job['site_tested']
        row = {**job.to_dict(), 'prediction_dir': str(pred_dir), 'model_file': str(model_file), 'missing_output': False, **geom}
        row.update({k: conf.get(k, np.nan) for k in ['confidence_score','ptm','iptm','protein_iptm','complex_plddt','complex_iplddt','complex_ipde']})
        if tested in active_sites_seqpos:
            row['tested_site_dmin_A'] = row.get(f'dmin_{tested}_A', np.nan)
            row['tested_site_contacts_5A'] = row.get(f'contacts_{tested}_5A', 0)
        else:
            row['tested_site_dmin_A'] = np.nanmin([row.get(f'dmin_{s}_A', np.nan) for s in CONFIG['boltz_sites_to_run']])
            row['tested_site_contacts_5A'] = max([row.get(f'contacts_{s}_5A', 0) for s in CONFIG['boltz_sites_to_run']])
        rows.append(row)

boltz_scores = pd.DataFrame(rows)
if len(boltz_scores):
    boltz_scores['site_distance_norm'] = norm_0_100(boltz_scores['tested_site_dmin_A'], higher_is_better=False)
    boltz_scores['site_contacts_norm'] = norm_0_100(boltz_scores['tested_site_contacts_5A'], higher_is_better=True)
    boltz_scores['confidence_norm'] = norm_0_100(boltz_scores['confidence_score'], higher_is_better=True)
    boltz_scores['protein_iptm_norm'] = norm_0_100(boltz_scores['protein_iptm'], higher_is_better=True)
    boltz_scores['boltz_filter_score'] = 0.35*boltz_scores['site_distance_norm'].fillna(0) + 0.25*boltz_scores['site_contacts_norm'].fillna(0) + 0.20*boltz_scores['confidence_norm'].fillna(50) + 0.20*boltz_scores['protein_iptm_norm'].fillna(50)
    boltz_scores['boltz_geometry_call'] = np.select(
        [(boltz_scores['tested_site_dmin_A'] <= CONFIG['boltz_pass_dmin_A']) | (boltz_scores['tested_site_contacts_5A'] >= CONFIG['boltz_pass_contacts_5A']), boltz_scores['tested_site_dmin_A'] <= CONFIG['boltz_borderline_dmin_A']],
        ['active_site', 'borderline'],
        default='off_site'
    )
    confidence_ok = (boltz_scores['confidence_score'].fillna(0) >= CONFIG['boltz_min_confidence_score']) | (boltz_scores['protein_iptm'].fillna(0) >= CONFIG['boltz_min_protein_iptm'])
    boltz_scores['boltz_filter_pass'] = boltz_scores['site_tested'].isin(CONFIG['boltz_sites_to_run']) & boltz_scores['boltz_geometry_call'].isin(['active_site','borderline']) & confidence_ok & ~boltz_scores['missing_output'].fillna(True)

boltz_scores.to_csv(REPORTS / '04_boltz_site_filter_all_models.csv', index=False)
usable = boltz_scores[~boltz_scores.get('missing_output', True).fillna(True)].copy() if len(boltz_scores) else pd.DataFrame()
if len(usable):
    usable = usable.sort_values(['boltz_filter_pass', 'boltz_filter_score'], ascending=[False, False])
    boltz_best = usable.groupby('peptide_id', as_index=False).head(1).reset_index(drop=True)
    candidates_for_docking = boltz_best[boltz_best['boltz_filter_pass']].sort_values('boltz_filter_score', ascending=False).head(CONFIG['max_peptides_for_docking'])
else:
    boltz_best = pd.DataFrame()
    candidates_for_docking = pd.DataFrame()
boltz_best.to_csv(REPORTS / '05_boltz_best_by_peptide.csv', index=False)
candidates_for_docking.to_csv(REPORTS / '06_candidates_for_docking.csv', index=False)
print('Peptides passing Boltz filter:', len(candidates_for_docking))
candidates_for_docking.head(30)
"""
    ),
    md(
        r"""
## 6. Site-directed docking with LightDock

The receptor restraints restrict sampling around the site selected by the Boltz-2 filter. The peptide starting structure is extracted from the best Boltz-2 model when possible; otherwise an ideal peptide is generated.
"""
    ),
    code(
        r"""
#@title Prepare and run LightDock
from Bio.PDB import PDBIO, Select

class SingleChainSelect(Select):
    def __init__(self, chain_id):
        self.chain_id = chain_id
    def accept_chain(self, chain):
        return chain.id == self.chain_id
    def accept_atom(self, atom):
        return atom.element != 'H'

def extract_chain_to_pdb(structure_file, chain_id, out_pdb):
    structure = load_structure_any(structure_file)
    io = PDBIO(); io.set_structure(structure)
    io.save(str(out_pdb), SingleChainSelect(chain_id))
    # Force ligand chain ID B in PDB text.
    fixed = []
    for line in Path(out_pdb).read_text(errors='replace').splitlines():
        if line.startswith(('ATOM','HETATM')) and len(line) >= 22:
            line = line[:21] + 'B' + line[22:]
        fixed.append(line)
    Path(out_pdb).write_text('\n'.join(fixed) + '\n')
    return out_pdb

def build_ideal_peptide_pdb(sequence, out_pdb):
    import PeptideBuilder
    structure = PeptideBuilder.make_structure(sequence)
    for chain in structure.get_chains():
        chain.id = 'B'
    io = PDBIO(); io.set_structure(structure); io.save(str(out_pdb))
    return out_pdb

def resname_for_pdbnum(pdbnum):
    return seqpos_to_resname.get(pdbnum_to_seqpos.get(int(pdbnum)), 'UNK')

def write_lightdock_restraints(path, site_name, peptide_sequence):
    lines = [f"R {CONFIG['target_chain']}.{resname_for_pdbnum(r)}.{int(r)}" for r in CONFIG['active_sites_pdbnum'][site_name]]
    lines += [f"L B.{one_to_three.get(aa, 'GLY')}.{i}" for i, aa in enumerate(peptide_sequence, start=1)]
    Path(path).write_text('\n'.join(lines) + '\n')
    return path

def concatenate_complex(receptor_pdb, ligand_pdb, out_pdb):
    lines = []
    for src in [receptor_pdb, ligand_pdb]:
        for line in Path(src).read_text(errors='replace').splitlines():
            if line.startswith(('ATOM','HETATM')):
                lines.append(line)
    Path(out_pdb).write_text('\n'.join(lines + ['END','']))
    return out_pdb

prepared = []
for _, row in candidates_for_docking.iterrows() if len(candidates_for_docking) else []:
    dock_dir = DOCKING_DIR / row['peptide_id'] / row['site_tested']
    dock_dir.mkdir(parents=True, exist_ok=True)
    receptor_pdb = dock_dir / 'receptor.pdb'
    peptide_pdb = dock_dir / 'peptide_start.pdb'
    shutil.copyfile(receptor_clean, receptor_pdb)
    if row.get('model_file') and Path(row['model_file']).exists():
        try:
            extract_chain_to_pdb(row['model_file'], CONFIG['model_peptide_chain'], peptide_pdb)
        except Exception:
            build_ideal_peptide_pdb(row['sequence'], peptide_pdb)
    else:
        build_ideal_peptide_pdb(row['sequence'], peptide_pdb)
    restraints = write_lightdock_restraints(dock_dir / 'restraints.list', row['site_tested'], row['sequence'])
    prepared.append({'peptide_id': row['peptide_id'], 'sequence': row['sequence'], 'site': row['site_tested'], 'dock_dir': str(dock_dir), 'receptor_pdb': str(receptor_pdb), 'peptide_pdb': str(peptide_pdb), 'restraints': str(restraints), 'boltz_model_file': row.get('model_file','')})
prepared_docking = pd.DataFrame(prepared)
prepared_docking.to_csv(REPORTS / '07_prepared_docking_jobs.csv', index=False)

if RUN_DOCKING and (not TOOL_STATUS['lightdock_setup'] or not TOOL_STATUS['lightdock_run']):
    raise RuntimeError('RUN_DOCKING=True but LightDock commands are missing.')

if RUN_DOCKING:
    for _, job in prepared_docking.iterrows():
        d = Path(job['dock_dir'])
        run_cmd([LIGHTDOCK_SETUP, 'receptor.pdb', 'peptide_start.pdb', '-s', CONFIG['lightdock_swarms'], '-g', CONFIG['lightdock_glowworms'], '-r', 'restraints.list', '--noxt', '--noh', '--now', '-spr', '8'], cwd=d)
        run_cmd([LIGHTDOCK_RUN, 'setup.json', CONFIG['lightdock_steps'], '-s', CONFIG['lightdock_score'], '-c', CONFIG['lightdock_cores']], cwd=d)
        if LIGHTDOCK_GENERATE:
            for gso in sorted(d.glob('swarm_*/gso_*.out')):
                run_cmd([LIGHTDOCK_GENERATE, '../receptor.pdb', '../peptide_start.pdb', gso.name, str(CONFIG['lightdock_top_models_per_swarm']), '--setup', '../setup.json'], cwd=gso.parent, allow_fail=True)
        else:
            print('WARNING: lgd_generate_conformations.py is missing, so docking scores may exist but pose PDB files will not be generated.')
else:
    print('RUN_DOCKING=False. Prepared docking folders:', DOCKING_DIR)
prepared_docking
"""
    ),
    code(
        r"""
#@title Collect docking poses
pose_rows = []
if len(prepared_docking):
    for _, job in prepared_docking.iterrows():
        d = Path(job['dock_dir'])
        poses = sorted(d.glob('swarm_*/lightdock_*.pdb'))
        if not poses and job.get('boltz_model_file') and Path(job['boltz_model_file']).exists():
            poses = [Path(job['boltz_model_file'])]
        for i, pose in enumerate(poses):
            complex_pdb = d / f'complex_pose_{i:03d}.pdb'
            if pose.name.startswith('lightdock_'):
                concatenate_complex(job['receptor_pdb'], pose, complex_pdb)
                geom = score_structure_against_sites(complex_pdb, CONFIG['active_sites_pdbnum'], CONFIG['target_chain'], 'B')
            else:
                complex_pdb = pose
                geom = score_structure_against_sites(complex_pdb, active_sites_seqpos, CONFIG['model_receptor_chain'], CONFIG['model_peptide_chain'])
            row = {'peptide_id': job['peptide_id'], 'site': job['site'], 'pose_file': str(pose), 'complex_pdb': str(complex_pdb), **geom}
            row['docking_site_dmin_A'] = row.get(f"dmin_{job['site']}_A", np.nan)
            row['docking_site_contacts_5A'] = row.get(f"contacts_{job['site']}_5A", np.nan)
            pose_rows.append(row)
docking_poses = pd.DataFrame(pose_rows)
if len(docking_poses):
    docking_poses['docking_score_norm'] = norm_0_100(docking_poses['docking_site_dmin_A'], higher_is_better=False)
    docking_poses['site_contact_score_norm'] = norm_0_100(docking_poses['docking_site_contacts_5A'], higher_is_better=True)
docking_poses.to_csv(REPORTS / '08_docking_poses.csv', index=False)
docking_poses.head(30)
"""
    ),
    md(
        r"""
## 7. OpenMM GBSA rescoring

This computes a single-frame MM/GBSA-like interaction estimate:

```text
DeltaG_GBSA ~= E_complex_GBSA - E_receptor_GBSA - E_peptide_GBSA
```

It is useful as a consistent Colab rescoring layer, but not a replacement for a fully equilibrated Amber MMPBSA.py protocol with ensemble averaging.
"""
    ),
    code(
        r"""
#@title Run OpenMM GBSA
class KeepChainsSelect(Select):
    def __init__(self, chain_ids):
        self.chain_ids = set(chain_ids)
    def accept_chain(self, chain):
        return chain.id in self.chain_ids

def write_chain_subset_pdb(in_pdb, out_pdb, chain_ids):
    structure = load_structure_any(in_pdb)
    io = PDBIO(); io.set_structure(structure)
    io.save(str(out_pdb), KeepChainsSelect(chain_ids))
    return out_pdb

def openmm_gbsa_energy_kcal(pdb_path, minimize=True):
    from openmm import LangevinIntegrator, unit
    from openmm.app import PDBFile, ForceField, Modeller, Simulation, NoCutoff, HBonds
    pdb = PDBFile(str(pdb_path))
    forcefield = ForceField('amber14-all.xml', 'implicit/gbn2.xml')
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(forcefield, pH=float(CONFIG['openmm_ph']))
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=NoCutoff, constraints=HBonds, implicitSolventKappa=0.0)
    integrator = LangevinIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
    simulation = Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
    if minimize:
        simulation.minimizeEnergy(maxIterations=int(CONFIG['openmm_minimize_max_iterations']))
    state = simulation.context.getState(getEnergy=True)
    return float(state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole))

mmgbsa_rows = []
if RUN_MMGBSA and len(docking_poses):
    for i, row in docking_poses.iterrows():
        cpx = Path(row['complex_pdb'])
        job_dir = MMGBSA_DIR / row['peptide_id'] / row['site'] / f'pose_{i:03d}'
        job_dir.mkdir(parents=True, exist_ok=True)
        rec = write_chain_subset_pdb(cpx, job_dir / 'receptor_only.pdb', [CONFIG['target_chain'], CONFIG['model_receptor_chain']])
        lig = write_chain_subset_pdb(cpx, job_dir / 'peptide_only.pdb', ['B', CONFIG['model_peptide_chain']])
        try:
            e_c = openmm_gbsa_energy_kcal(cpx, CONFIG['openmm_minimize'])
            e_r = openmm_gbsa_energy_kcal(rec, CONFIG['openmm_minimize'])
            e_l = openmm_gbsa_energy_kcal(lig, CONFIG['openmm_minimize'])
            dg = e_c - e_r - e_l
            err = ''
        except Exception as e:
            e_c = e_r = e_l = dg = np.nan
            err = str(e)
        mmgbsa_rows.append({'peptide_id': row['peptide_id'], 'site': row['site'], 'complex_pdb': str(cpx), 'openmm_E_complex_kcal_mol': e_c, 'openmm_E_receptor_kcal_mol': e_r, 'openmm_E_peptide_kcal_mol': e_l, 'openmm_deltaG_gbsa_kcal_mol': dg, 'mmgbsa_error': err})
        print(row['peptide_id'], row['site'], dg, err[:100])
else:
    print('RUN_MMGBSA=False or no docking poses available.')
mmgbsa_scores = pd.DataFrame(mmgbsa_rows)
mmgbsa_scores.to_csv(REPORTS / '09_openmm_gbsa_scores.csv', index=False)
mmgbsa_scores.head(20)
"""
    ),
    md(
        r"""
## 8. Final ranking and export
"""
    ),
    code(
        r"""
#@title Build final ranking
if len(boltz_best):
    final = boltz_best.rename(columns={'boltz_filter_score': 'boltz_filter_score_raw'}).copy()
else:
    final = peptides.copy()
    final['boltz_filter_score_raw'] = np.nan
    final['boltz_filter_pass'] = False

if 'docking_poses' in globals() and len(docking_poses):
    dock_best = docking_poses.sort_values(['docking_score_norm','site_contact_score_norm'], ascending=[False, False]).groupby('peptide_id', as_index=False).head(1)
    final = final.merge(dock_best[['peptide_id','complex_pdb','pose_file','docking_score_norm','docking_site_dmin_A','docking_site_contacts_5A','site_contact_score_norm']], on='peptide_id', how='left')
else:
    final['docking_score_norm'] = np.nan
    final['site_contact_score_norm'] = np.nan

if 'mmgbsa_scores' in globals() and len(mmgbsa_scores):
    mm = mmgbsa_scores.copy()
    mm['mmgbsa_dg_norm'] = norm_0_100(mm['openmm_deltaG_gbsa_kcal_mol'], higher_is_better=False)
    mm_best = mm.sort_values('mmgbsa_dg_norm', ascending=False).groupby('peptide_id', as_index=False).head(1)
    final = final.merge(mm_best[['peptide_id','openmm_deltaG_gbsa_kcal_mol','mmgbsa_dg_norm','mmgbsa_error']], on='peptide_id', how='left')
else:
    final['openmm_deltaG_gbsa_kcal_mol'] = np.nan
    final['mmgbsa_dg_norm'] = np.nan

final['boltz_filter_score_norm'] = norm_0_100(final['boltz_filter_score_raw'], higher_is_better=True)
for col in ['boltz_filter_score_norm','docking_score_norm','mmgbsa_dg_norm','site_contact_score_norm']:
    final[f'{col}_missing'] = final[col].isna()
    final[col] = pd.to_numeric(final[col], errors='coerce').fillna(50.0)

weights = CONFIG['rank_weights']
final['final_screening_score'] = sum(float(weights[k]) * final[k] for k in weights)
final['screening_status'] = np.select(
    [final.get('boltz_filter_pass', False).fillna(False) & ~final['docking_score_norm_missing'] & ~final['mmgbsa_dg_norm_missing'], final.get('boltz_filter_pass', False).fillna(False)],
    ['complete_ranked', 'passed_boltz_incomplete_downstream'],
    default='failed_or_missing_boltz_filter'
)
final = final.sort_values('final_screening_score', ascending=False).reset_index(drop=True)
final.insert(0, 'rank', np.arange(1, len(final) + 1))
final.to_csv(REPORTS / '10_final_screening_ranking.csv', index=False)
final.head(50)
"""
    ),
    code(
        r"""
#@title Visualize a ranked complex
try:
    import py3Dmol
except Exception as e:
    py3Dmol = None
    print('py3Dmol unavailable:', e)

rank_to_view = 1 #@param {type:'integer'}
if py3Dmol and 'final' in globals() and len(final):
    row = final.iloc[max(0, rank_to_view - 1)]
    path = str(row.get('complex_pdb', '') or row.get('model_file', ''))
    if path and Path(path).exists():
        data = Path(path).read_text(errors='replace')
        fmt = 'pdb' if Path(path).suffix.lower() == '.pdb' else 'cif'
        view = py3Dmol.view(width=950, height=650)
        view.addModel(data, fmt)
        view.setStyle({'chain': CONFIG['target_chain']}, {'cartoon': {'color': 'lightgray'}})
        view.setStyle({'chain': 'B'}, {'cartoon': {'color': 'orange'}, 'stick': {}})
        for residues in CONFIG['active_sites_pdbnum'].values():
            view.addStyle({'chain': CONFIG['target_chain'], 'resi': residues}, {'stick': {'radius': 0.25}})
        view.zoomTo()
        view.show()
        print('Viewing:', path)
    else:
        print('No complex file available for this rank.')
"""
    ),
    code(
        r"""
%%bash
#@title Zip results
cd /content
zip -qr hache_peptide_screening_boltz2_results.zip hache_peptide_screening_boltz2 || true
ls -lh /content/hache_peptide_screening_boltz2_results.zip || true
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(notebook, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
print(f"Wrote {OUT}")
