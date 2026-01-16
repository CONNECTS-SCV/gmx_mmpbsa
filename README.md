# gmx_MMPBSA Workflow with YAML Configuration

CHARMM36m-compatible gmx_MMPBSA workflow with YAML configuration system and automatic result analysis.

## Features

- ✅ **YAML Configuration System**: Easy-to-use configuration files instead of complex input files
- ✅ **6 Analysis Types**: Binding energy, stability, alanine scanning, QM/MMGBSA, entropy, decomposition
- ✅ **CHARMM36m Compatible**: Full support for CHARMM force field
- ✅ **Automatic Analysis**: Auto-generates reports and visualizations after calculation
- ✅ **MPI Parallelization**: High-performance computing support

## Installation

### Prerequisites

- Conda/Mamba
- GROMACS 2025.4 (nompi version)
- AmberTools 23

### Quick Setup

```bash
# Create conda environment
conda create -n gmxmmpbsa python=3.9 -y
conda activate gmxmmpbsa

# Install dependencies
pip install -r requirements.txt

# Install GROMACS and AmberTools
conda install -c conda-forge ambertools=23 -y
conda install -c conda-forge "gromacs=2025.4=nompi_h26635d9_100" -y
```

## Usage

### Basic Workflow

```bash
# 1. Edit YAML configuration file
vim configs/1_binding_free_energy.yaml

# 2. Run analysis
./run_mmpbsa_full.sh -c configs/1_binding_free_energy.yaml

# 3. Results are automatically analyzed and saved to analysis_output/
```

### 6 Analysis Types

| Type | Config File | Purpose |
|------|-------------|---------|
| 1️⃣ **Binding Energy** | `1_binding_free_energy.yaml` | Calculate ΔG binding |
| 2️⃣ **Stability** | `2_stability.yaml` | Protein folding stability |
| 3️⃣ **Alanine Scanning** | `3_alanine_scanning.yaml` | Identify critical residues |
| 4️⃣ **QM/MMGBSA** | `4_qm_mmgbsa.yaml` | Quantum mechanics hybrid |
| 5️⃣ **Entropy** | `5_entropy_correction.yaml` | ΔG = ΔH - TΔS |
| 6️⃣ **Decomposition** | `6_decomposition.yaml` | Per-residue contributions |

See [ANALYSIS_TYPES.md](ANALYSIS_TYPES.md) for detailed descriptions.

## Automatic Result Analysis

After gmx_MMPBSA completes, results are automatically analyzed:

### Generated Files

```
example/
├── FINAL_RESULTS_MMPBSA.dat      # gmx_MMPBSA raw results
└── analysis_output/               # Auto-generated analysis
    ├── summary_report.txt         # Text summary
    ├── energy_components.png      # Energy bar chart
    ├── entropy_components.png     # Thermodynamic breakdown (Type 5)
    ├── top_residues_decomp.png    # Hot spot residues (Type 6)
    └── energy_breakdown.png       # Energy terms (Type 6)
```

### Analysis Features by Type

**1. Binding Energy**
- Energy components bar chart (ΔG gas, ΔG solv, ΔG total)
- Binding strength interpretation (Strong/Moderate/Weak)
- Top contributing residues (if decomposition enabled)

**2. Stability**
- Protein folding free energy
- Stability assessment notes

**3. Alanine Scanning**
- ΔΔG report with interpretation:
  - ⚠️ CRITICAL (ΔΔG > 2 kcal/mol)
  - ⚠️ Important (0.5 < ΔΔG < 2)
  - ℹ️ Neutral (-0.5 < ΔΔG < 0.5)
  - ✅ Favorable (ΔΔG < -0.5)

**4. QM/MMGBSA**
- QM-level binding energy
- QM vs MM comparison notes

**5. Entropy Correction**
- ΔH (enthalpy), -TΔS (entropy), ΔG (free energy)
- Thermodynamic components visualization
- Enthalpy-driven vs Entropy-driven determination

**6. Energy Decomposition**
- Top 20 contributing residues plot
- Energy term breakdown (vdW, Elec, Polar, Nonpolar)
- Hot spot residue list (|TDC| > 2 kcal/mol)

## Manual Analysis

You can also run analysis manually:

```bash
# Auto-detect analysis type
python scripts/analyze_results.py -d example/ -t auto

# Specify analysis type
python scripts/analyze_results.py -d example/ -t decomposition
```

Available types: `binding`, `stability`, `alanine`, `qmmm`, `entropy`, `decomposition`, `auto`

## Example Workflow

### 1. Binding Free Energy Calculation

```bash
# Edit config
vim configs/1_binding_free_energy.yaml

# Set your input files:
# - complex_structure: "step5_production.tpr"
# - complex_trajectory: "fit.xtc"
# - complex_topology: "topol.top"
# - receptor_group: 1
# - ligand_group: 13

# Run
./run_mmpbsa_full.sh -c configs/1_binding_free_energy.yaml

# Check results
cat example/analysis_output/summary_report.txt
```

### 2. Find Hot Spot Residues

```bash
# Run decomposition
./run_mmpbsa_full.sh -c configs/6_decomposition.yaml

# View results
cat example/analysis_output/decomposition_report.txt
open example/analysis_output/top_residues_decomp.png
```

### 3. Validate Critical Residue

```bash
# Edit alanine scanning config
vim configs/3_alanine_scanning.yaml

# Set residue to mutate (e.g., "A/50")
# Run
./run_mmpbsa_full.sh -c configs/3_alanine_scanning.yaml

# Check if residue is critical
cat example/analysis_output/alanine_scan_report.txt
```

## YAML Configuration Guide

### Required Fields

```yaml
calculation_type: binding_free_energy  # or stability, alanine_scanning

input_files:
  complex_structure: "path/to/structure.tpr"
  complex_trajectory: "path/to/trajectory.xtc"
  complex_topology: "path/to/topol.top"  # CHARMM36m required!
  receptor_group: 1
  ligand_group: 13

execution:
  mpi: true
  cores: 16

solvent_model:
  method: gb
  gb_model: 8

general:
  startframe: 1
  endframe: 100
  interval: 1
  forcefields: CHARMM36m
```

See individual YAML files in `configs/` for detailed examples with extensive comments.

## CHARMM36m Compatibility

### ✅ Supported
- GB (Generalized Born) - All models
- PB (Poisson-Boltzmann)
- Entropy: IE, QH, C2
- All 6 analysis types

### ❌ Not Supported
- NMODE (Normal Mode Analysis) - AMBER only
- Entropy in stability/alanine scanning modes (technical limitation)

## Output Files

### gmx_MMPBSA Native Outputs
- `FINAL_RESULTS_MMPBSA.dat` - Main results
- `FINAL_DECOMP_MMPBSA.csv` - Decomposition (Type 6)
- `_MMPBSA_*.mdout` - Detailed energy files
- `_MMPBSA_info` - Pickle file with metadata

### Auto-Analysis Outputs
- `analysis_output/summary_report.txt` - Text summary
- `analysis_output/*.png` - High-resolution plots (300 DPI)

All plots are publication-ready (PNG format, 300 DPI).

## Troubleshooting

### Common Errors

**1. NMODE with CHARMM**
```
ERROR: CHAMBER prmtops cannot be used with NMODE
```
**Solution**: Change entropy method to `ie`, `qh`, or `c2`

**2. KeyError: 'delta' in Stability**
```
KeyError: 'delta'
```
**Solution**: Remove entropy section from stability YAML

**3. Only ONE mutant residue allowed**
```
ERROR: Only ONE mutant residue is allowed!
```
**Solution**: Use format `"A/50"` (single residue with Chain ID)

**4. Cores > Frames (QM/MM)**
```
ERROR: Must have at least as many frames as processors!
```
**Solution**: Reduce cores or increase interval

## Performance Tips

- **Binding Energy**: 100 frames, interval=1, ~10 minutes
- **Stability**: 100 frames, interval=1, ~5 minutes
- **Alanine Scanning**: One residue at a time, ~20 minutes per residue
- **QM/MMGBSA**: 10-100 frames, interval=10, hours to days (very slow!)
- **Entropy (IE)**: 100+ frames, interval=1, ~30 minutes
- **Decomposition**: 100 frames, interval=1, ~20 minutes

Use more cores and MPI for faster computation.

## Citation

If you use this workflow, please cite:

- **gmx_MMPBSA**: Valdés-Tresanco et al. (2021) J. Chem. Theory Comput.
- **GROMACS**: Abraham et al. (2015) SoftwareX
- **AmberTools**: Case et al. (2023)

## License

MIT License

## Support

- GitHub Issues: https://github.com/your-repo/gmx_mmpbsa_workflow
- gmx_MMPBSA Documentation: https://valdes-tresanco-ms.github.io/gmx_MMPBSA/

## Version

- gmx_MMPBSA: 1.6.4
- GROMACS: 2025.4 (nompi)
- AmberTools: 23
- Force Field: CHARMM36m
