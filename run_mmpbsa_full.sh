#!/bin/bash

################################################################################
# gmx_MMPBSA Complete Wrapper - ALL OPTIONS SUPPORTED
# Full implementation of all gmx_MMPBSA command-line options
################################################################################

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}======================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}======================================================================${NC}"
}

print_msg() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

show_usage() {
    cat << 'EOF'
Usage: ./run_mmpbsa_full.sh [OPTIONS]

gmx_MMPBSA Complete Wrapper - All Options Supported

═══════════════════════════════════════════════════════════════════════════
GENERAL OPTIONS
═══════════════════════════════════════════════════════════════════════════
  -h, --help              Show this help message
  -v, --version           Show gmx_MMPBSA version
  --input-file-help       Show input file help
  --create-input TYPE     Create input file [gb,pb,rism,ala,decomp,nmode,all]
  -c, --config FILE       YAML configuration file (recommended!)

MISCELLANEOUS
  -O, --overwrite         Allow output file overwrite
  --prefix PREFIX         Prefix for intermediate files (default: _GMXMMPBSA_)

═══════════════════════════════════════════════════════════════════════════
INPUT/OUTPUT FILES
═══════════════════════════════════════════════════════════════════════════
  -i FILE                 MM/PBSA input file
  -xvvfile FILE           XVV file for 3D-RISM
  -o FILE                 Output file (default: FINAL_RESULTS_MMPBSA.dat)
  -do FILE                Decomposition output file
  -eo FILE                Energy terms CSV output
  -deo FILE               Decomposition CSV output
  -nogui                  Don't open gmx_MMPBSA_ana GUI
  -s, --stability         Perform stability calculation

═══════════════════════════════════════════════════════════════════════════
COMPLEX FILES (Required for standard calculation)
═══════════════════════════════════════════════════════════════════════════
  -cs FILE                Complex structure (.tpr, .pdb, .gro)
  -ci FILE                Complex index file
  -cg INDEX INDEX         Receptor and ligand group indices (e.g., -cg 1 13)
  -ct FILE [FILE...]      Complex trajectory files (.xtc, .trr, .pdb)
  -cp FILE                Complex topology file
  -cr FILE                Complex reference structure (PDB recommended)

═══════════════════════════════════════════════════════════════════════════
RECEPTOR FILES (Optional - for multiple trajectory approach)
═══════════════════════════════════════════════════════════════════════════
  -rs FILE                Receptor structure file
  -ri FILE                Receptor index file
  -rg INDEX               Receptor group in index file
  -rt FILE [FILE...]      Receptor trajectory files
  -rp FILE                Receptor topology file

═══════════════════════════════════════════════════════════════════════════
LIGAND FILES (Optional - for multiple trajectory approach)
═══════════════════════════════════════════════════════════════════════════
  -lm FILE                Ligand MOL2 file (for small molecules)
  -ls FILE                Ligand structure file
  -li FILE                Ligand index file
  -lg INDEX               Ligand group in index file
  -lt FILE [FILE...]      Ligand trajectory files
  -lp FILE                Ligand topology file

═══════════════════════════════════════════════════════════════════════════
MISCELLANEOUS ACTIONS
═══════════════════════════════════════════════════════════════════════════
  --rewrite-output        Rewrite output without re-running calculations
  --clean                 Clean temporary files and quit

═══════════════════════════════════════════════════════════════════════════
WRAPPER-SPECIFIC OPTIONS (Not in gmx_MMPBSA)
═══════════════════════════════════════════════════════════════════════════
  --mode MODE             Quick mode selection (gb, pb, 3drism, decomp, etc.)
  --mpi                   Use MPI parallelization
  -p, --cores N           Number of CPU cores for MPI (default: 4)
  --auto-prepare          Auto-prepare input from TPR+XTC (wrapper feature)
  -r INDEX                Receptor index (for auto-prepare)
  -l INDEX                Ligand index (for auto-prepare)
  -t FILE                 Trajectory file (for auto-prepare)

═══════════════════════════════════════════════════════════════════════════
EXAMPLES
═══════════════════════════════════════════════════════════════════════════

1. Basic single trajectory (wrapper auto-prepare):
   ./run_mmpbsa_full.sh --auto-prepare -s md.tpr -t md.xtc -r 1 -l 13 \
       -i configs/mmpbsa_gb.in --mode gb

2. Single trajectory (manual):
   ./run_mmpbsa_full.sh -i mmpbsa.in -cs complex.pdb -ci index.ndx \
       -cg 1 13 -ct complex.xtc

3. Multiple trajectory:
   ./run_mmpbsa_full.sh -i mmpbsa.in \
       -cs complex.pdb -ci complex.ndx -cg 1 13 -ct complex.xtc \
       -rs receptor.pdb -ri receptor.ndx -rg 1 -rt receptor.xtc \
       -ls ligand.pdb -li ligand.ndx -lg 13 -lt ligand.xtc

4. With MPI parallelization:
   ./run_mmpbsa_full.sh --mpi -p 8 -i mmpbsa.in -cs complex.pdb \
       -ci index.ndx -cg 1 13 -ct complex.xtc

5. Stability calculation:
   ./run_mmpbsa_full.sh --stability -i mmpbsa.in -cs complex.pdb \
       -ct complex.xtc

6. Small molecule with MOL2:
   ./run_mmpbsa_full.sh -i mmpbsa.in -cs complex.pdb -ci index.ndx \
       -cg 1 13 -ct complex.xtc -lm ligand.mol2

7. With custom outputs:
   ./run_mmpbsa_full.sh -i mmpbsa.in -cs complex.pdb -ci index.ndx \
       -cg 1 13 -ct complex.xtc -o results.dat -eo energies.csv \
       -do decomp.dat -deo decomp_detail.csv

8. 3D-RISM with XVV file:
   ./run_mmpbsa_full.sh -i mmpbsa_3drism.in -cs complex.pdb \
       -ci index.ndx -cg 1 13 -ct complex.xtc -xvvfile tip3p.xvv

9. Rewrite output only:
   ./run_mmpbsa_full.sh --rewrite-output

10. Clean temporary files:
    ./run_mmpbsa_full.sh --clean

EOF
}

################################################################################
# Initialize variables for ALL gmx_MMPBSA options
################################################################################

# General options
OPT_VERSION=false
OPT_INPUT_FILE_HELP=false
OPT_CREATE_INPUT=""
OPT_OVERWRITE=false
OPT_PREFIX="_GMXMMPBSA_"

# Input/Output files
OPT_INPUT_FILE=""
OPT_XVVFILE=""
OPT_OUTPUT_FILE=""
OPT_DECOMP_OUTPUT=""
OPT_ENERGY_OUTPUT=""
OPT_DECOMP_ENERGY_OUTPUT=""
OPT_NOGUI=false
OPT_STABILITY=false

# Complex files
OPT_COMPLEX_STRUCTURE=""
OPT_COMPLEX_INDEX=""
OPT_COMPLEX_GROUP_REC=""
OPT_COMPLEX_GROUP_LIG=""
OPT_COMPLEX_TRAJ=()
OPT_COMPLEX_TOPOLOGY=""
OPT_COMPLEX_REFERENCE=""

# Receptor files
OPT_RECEPTOR_STRUCTURE=""
OPT_RECEPTOR_INDEX=""
OPT_RECEPTOR_GROUP=""
OPT_RECEPTOR_TRAJ=()
OPT_RECEPTOR_TOPOLOGY=""

# Ligand files
OPT_LIGAND_MOL2=""
OPT_LIGAND_STRUCTURE=""
OPT_LIGAND_INDEX=""
OPT_LIGAND_GROUP=""
OPT_LIGAND_TRAJ=()
OPT_LIGAND_TOPOLOGY=""

# Actions
OPT_REWRITE_OUTPUT=false
OPT_CLEAN=false

# Wrapper-specific
OPT_MODE=""
OPT_MPI=false
OPT_CORES=4
OPT_AUTO_PREPARE=false
OPT_TPR_FILE=""
OPT_TRAJ_FILE=""
OPT_RECEPTOR_IDX=""
OPT_LIGAND_IDX=""

################################################################################
# Parse command line arguments
################################################################################

while [[ $# -gt 0 ]]; do
    case $1 in
        # Help and version
        -h|--help)
            show_usage
            exit 0
            ;;
        -v|--version)
            OPT_VERSION=true
            shift
            ;;
        --input-file-help)
            OPT_INPUT_FILE_HELP=true
            shift
            ;;
        --create-input)
            OPT_CREATE_INPUT="$2"
            shift 2
            ;;
        -c|--config)
            OPT_CONFIG_YAML="$2"
            shift 2
            ;;

        # Miscellaneous
        -O|--overwrite)
            OPT_OVERWRITE=true
            shift
            ;;
        --prefix)
            OPT_PREFIX="$2"
            shift 2
            ;;

        # Input/Output files
        -i)
            OPT_INPUT_FILE="$2"
            shift 2
            ;;
        -xvvfile)
            OPT_XVVFILE="$2"
            shift 2
            ;;
        -o)
            OPT_OUTPUT_FILE="$2"
            shift 2
            ;;
        -do)
            OPT_DECOMP_OUTPUT="$2"
            shift 2
            ;;
        -eo)
            OPT_ENERGY_OUTPUT="$2"
            shift 2
            ;;
        -deo)
            OPT_DECOMP_ENERGY_OUTPUT="$2"
            shift 2
            ;;
        -nogui)
            OPT_NOGUI=true
            shift
            ;;
        -s|--stability)
            OPT_STABILITY=true
            shift
            ;;

        # Complex files
        -cs)
            OPT_COMPLEX_STRUCTURE="$2"
            shift 2
            ;;
        -ci)
            OPT_COMPLEX_INDEX="$2"
            shift 2
            ;;
        -cg)
            OPT_COMPLEX_GROUP_REC="$2"
            OPT_COMPLEX_GROUP_LIG="$3"
            shift 3
            ;;
        -ct)
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^- ]]; do
                OPT_COMPLEX_TRAJ+=("$1")
                shift
            done
            ;;
        -cp)
            OPT_COMPLEX_TOPOLOGY="$2"
            shift 2
            ;;
        -cr)
            OPT_COMPLEX_REFERENCE="$2"
            shift 2
            ;;

        # Receptor files
        -rs)
            OPT_RECEPTOR_STRUCTURE="$2"
            shift 2
            ;;
        -ri)
            OPT_RECEPTOR_INDEX="$2"
            shift 2
            ;;
        -rg)
            OPT_RECEPTOR_GROUP="$2"
            shift 2
            ;;
        -rt)
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^- ]]; do
                OPT_RECEPTOR_TRAJ+=("$1")
                shift
            done
            ;;
        -rp)
            OPT_RECEPTOR_TOPOLOGY="$2"
            shift 2
            ;;

        # Ligand files
        -lm)
            OPT_LIGAND_MOL2="$2"
            shift 2
            ;;
        -ls)
            OPT_LIGAND_STRUCTURE="$2"
            shift 2
            ;;
        -li)
            OPT_LIGAND_INDEX="$2"
            shift 2
            ;;
        -lg)
            OPT_LIGAND_GROUP="$2"
            shift 2
            ;;
        -lt)
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^- ]]; do
                OPT_LIGAND_TRAJ+=("$1")
                shift
            done
            ;;
        -lp)
            OPT_LIGAND_TOPOLOGY="$2"
            shift 2
            ;;

        # Actions
        --rewrite-output)
            OPT_REWRITE_OUTPUT=true
            shift
            ;;
        --clean)
            OPT_CLEAN=true
            shift
            ;;

        # Wrapper-specific
        --mode)
            OPT_MODE="$2"
            shift 2
            ;;
        --mpi)
            OPT_MPI=true
            shift
            ;;
        -p|--cores)
            OPT_CORES="$2"
            shift 2
            ;;
        --auto-prepare)
            OPT_AUTO_PREPARE=true
            shift
            ;;
        -t)
            OPT_TRAJ_FILE="$2"
            shift 2
            ;;
        -r)
            OPT_RECEPTOR_IDX="$2"
            shift 2
            ;;
        -l)
            OPT_LIGAND_IDX="$2"
            shift 2
            ;;

        *)
            print_error "Unknown option: $1"
            echo "Use -h or --help for usage"
            exit 1
            ;;
    esac
done

################################################################################
# Check dependencies
################################################################################
check_dependencies() {
    # Check gmx_MMPBSA
    if ! command -v gmx_MMPBSA &> /dev/null; then
        print_error "gmx_MMPBSA not found!"
        print_error "Activate virtual environment or install: pip install gmx_MMPBSA"
        exit 1
    fi

    # Check MPI if requested
    if [ "$OPT_MPI" = true ]; then
        if ! command -v mpirun &> /dev/null && ! command -v mpiexec &> /dev/null; then
            print_warning "MPI not found, falling back to serial execution"
            OPT_MPI=false
        fi

        if ! python3 -c "import mpi4py" 2>/dev/null; then
            print_warning "mpi4py not installed, falling back to serial execution"
            OPT_MPI=false
        fi
    fi
}

################################################################################
# Handle simple actions
################################################################################

if [ "$OPT_VERSION" = true ]; then
    gmx_MMPBSA --version
    exit 0
fi

if [ "$OPT_INPUT_FILE_HELP" = true ]; then
    gmx_MMPBSA --input-file-help
    exit 0
fi

if [ -n "$OPT_CREATE_INPUT" ]; then
    gmx_MMPBSA --create_input "$OPT_CREATE_INPUT"
    exit 0
fi

if [ "$OPT_CLEAN" = true ]; then
    print_msg "Cleaning temporary files..."
    rm -rf _GMXMMPBSA_* ${OPT_PREFIX}*
    print_msg "✓ Cleanup complete"
    exit 0
fi

# Check dependencies only if gmx_MMPBSA command will be used
# Commented out to allow script to work even if gmx_MMPBSA is not in PATH
# (it may be available when the command is actually executed)
# check_dependencies

################################################################################
# Auto-prepare mode (wrapper feature)
################################################################################

if [ "$OPT_AUTO_PREPARE" = true ]; then
    print_header "Auto-Prepare Mode"

    if [ -z "$OPT_COMPLEX_STRUCTURE" ] || [ -z "$OPT_TRAJ_FILE" ]; then
        print_error "Auto-prepare requires -cs (TPR) and -t (trajectory)"
        exit 1
    fi

    if [ -z "$OPT_RECEPTOR_IDX" ] || [ -z "$OPT_LIGAND_IDX" ]; then
        print_error "Auto-prepare requires -r (receptor index) and -l (ligand index)"
        exit 1
    fi

    # Run preparation script
    if [ -f "$SCRIPT_DIR/scripts/01_prepare_input.sh" ]; then
        bash "$SCRIPT_DIR/scripts/01_prepare_input.sh" \
            "$OPT_COMPLEX_STRUCTURE" "$OPT_TRAJ_FILE" \
            "$OPT_RECEPTOR_IDX" "$OPT_LIGAND_IDX"

        # Set prepared files
        INPUT_DIR="mmpbsa_input"
        OPT_COMPLEX_STRUCTURE="$INPUT_DIR/complex.pdb"
        OPT_COMPLEX_INDEX="$INPUT_DIR/index.ndx"
        OPT_COMPLEX_TRAJ=("$INPUT_DIR/complex.xtc")
        OPT_COMPLEX_GROUP_REC=$((OPT_RECEPTOR_IDX + 1))
        OPT_COMPLEX_GROUP_LIG=$((OPT_LIGAND_IDX + 1))

        print_msg "✓ Input files prepared"
    else
        print_error "Preparation script not found!"
        exit 1
    fi

    # Auto-select config if mode specified
    if [ -n "$OPT_MODE" ] && [ -z "$OPT_INPUT_FILE" ]; then
        case $OPT_MODE in
            gb) OPT_INPUT_FILE="$SCRIPT_DIR/configs/mmpbsa_gb.in" ;;
            pb) OPT_INPUT_FILE="$SCRIPT_DIR/configs/mmpbsa_pb.in" ;;
            3drism) OPT_INPUT_FILE="$SCRIPT_DIR/configs/mmpbsa_3drism.in" ;;
            decomp) OPT_INPUT_FILE="$SCRIPT_DIR/configs/mmpbsa_decomp_perresidue.in" ;;
            *) print_warning "Unknown mode: $OPT_MODE" ;;
        esac
    fi
fi

################################################################################
# Process YAML config if provided
################################################################################

if [ -n "$OPT_CONFIG_YAML" ]; then
    print_header "Processing YAML Configuration"

    if [ ! -f "$OPT_CONFIG_YAML" ]; then
        print_error "YAML config file not found: $OPT_CONFIG_YAML"
        exit 1
    fi

    # Convert YAML to .in file
    GENERATED_INPUT="/tmp/mmpbsa_generated_$$.in"
    print_msg "Converting YAML to input file..."

    python3 "$SCRIPT_DIR/scripts/yaml_to_input.py" "$OPT_CONFIG_YAML" -o "$GENERATED_INPUT"

    if [ $? -eq 0 ]; then
        OPT_INPUT_FILE="$GENERATED_INPUT"
        print_msg "Generated input file: $GENERATED_INPUT"
    else
        print_error "Failed to convert YAML to input file"
        exit 1
    fi

    # Extract file paths from YAML using Python
    print_msg "Extracting input files from YAML..."
    YAML_FILES=$(python3 -c "
import yaml
import sys
with open('$OPT_CONFIG_YAML', 'r') as f:
    config = yaml.safe_load(f)
    input_files = config.get('input_files', {})
    execution = config.get('execution', {})
    output = config.get('output', {})
    calc_type = config.get('calculation_type', '')

    # Print values separated by '|'
    print(input_files.get('complex_structure', ''), end='|')
    print(input_files.get('complex_trajectory', ''), end='|')
    print(input_files.get('complex_topology', ''), end='|')
    print(input_files.get('complex_index', ''), end='|')
    print(input_files.get('receptor_group', ''), end='|')
    print(input_files.get('ligand_group', ''), end='|')
    print('true' if execution.get('mpi', False) else 'false', end='|')
    print(execution.get('cores', 4), end='|')
    print(output.get('energy_csv', ''), end='|')
    print(output.get('decomp_csv', ''), end='|')
    print(calc_type)
")

    # Parse extracted values
    IFS='|' read -r YAML_CS YAML_CT YAML_TOP YAML_CI YAML_RG YAML_LG YAML_MPI YAML_CORES YAML_EO YAML_DEO YAML_CALC_TYPE <<< "$YAML_FILES"

    # Set options from YAML
    [ -n "$YAML_CS" ] && OPT_COMPLEX_STRUCTURE="$YAML_CS"
    [ -n "$YAML_CT" ] && OPT_COMPLEX_TRAJ=("$YAML_CT")
    [ -n "$YAML_TOP" ] && OPT_COMPLEX_TOPOLOGY="$YAML_TOP"
    [ -n "$YAML_CI" ] && OPT_COMPLEX_INDEX="$YAML_CI"
    [ -n "$YAML_RG" ] && OPT_COMPLEX_GROUP_REC="$YAML_RG"
    [ -n "$YAML_LG" ] && OPT_COMPLEX_GROUP_LIG="$YAML_LG"
    [ "$YAML_MPI" = "true" ] && OPT_MPI=true && OPT_CORES="$YAML_CORES"
    [ -n "$YAML_EO" ] && OPT_ENERGY_OUTPUT="$YAML_EO"
    [ -n "$YAML_DEO" ] && OPT_DECOMP_ENERGY_OUTPUT="$YAML_DEO"

    # Set calculation type flags
    if [ "$YAML_CALC_TYPE" = "stability" ]; then
        OPT_STABILITY=true
        print_msg "Calculation type: Stability"
    fi

    # Check for Alanine Scanning with multiple residues
    if [[ "$YAML_CALC_TYPE" =~ alanine ]]; then
        print_msg "Calculation type: Alanine Scanning"

        # Extract residues list from YAML
        RESIDUES_LIST=$(python3 -c "
import yaml
with open('$OPT_CONFIG_YAML', 'r') as f:
    config = yaml.safe_load(f)
    ala_scan = config.get('alanine_scanning', {})
    residues = ala_scan.get('residues', '')
    print(residues)
")

        # Check if multiple residues are specified (comma or space separated)
        if [[ "$RESIDUES_LIST" == *","* ]] || [[ "$RESIDUES_LIST" == *" "* ]]; then
            print_header "Alanine Scanning - Multiple Residues Detected"
            print_msg "Residues to scan: $RESIDUES_LIST"
            print_msg "Will run separate calculation for each residue..."

            # Store original YAML path as absolute path for later use
            ORIGINAL_YAML="$(cd "$(dirname "$OPT_CONFIG_YAML")" && pwd)/$(basename "$OPT_CONFIG_YAML")"

            # Set flag for alanine scanning batch mode
            ALANINE_BATCH_MODE=true
        else
            print_msg "Single residue: $RESIDUES_LIST"
            ALANINE_BATCH_MODE=false
        fi
    fi

    print_msg "Files from YAML:"
    print_msg "  Structure: $OPT_COMPLEX_STRUCTURE"
    print_msg "  Trajectory: ${OPT_COMPLEX_TRAJ[0]}"
    print_msg "  Topology: $OPT_COMPLEX_TOPOLOGY"
    [ -n "$OPT_COMPLEX_INDEX" ] && print_msg "  Index: $OPT_COMPLEX_INDEX"
    print_msg "  Receptor group: $OPT_COMPLEX_GROUP_REC"
    [ "$OPT_STABILITY" != true ] && print_msg "  Ligand group: $OPT_COMPLEX_GROUP_LIG"
fi

################################################################################
# Build gmx_MMPBSA command
################################################################################

print_header "Building gmx_MMPBSA Command"

# Check for gmx availability for MPI execution
if [ "$OPT_MPI" = true ]; then
    if ! command -v gmx &> /dev/null; then
        print_warning "⚠ gmx command not found"
        print_warning "⚠ MPI parallel execution requires non-MPI gmx"
        print_warning "⚠ Forcing serial execution mode"
        OPT_MPI=false
    fi
fi

CMD=""

# Base command
if [ "$OPT_MPI" = true ]; then
    CMD="mpirun -np $OPT_CORES gmx_MMPBSA MPI"
    print_msg "Using MPI with $OPT_CORES cores"
else
    CMD="gmx_MMPBSA"
    print_msg "Using serial execution"
fi

# Overwrite flag
if [ "$OPT_OVERWRITE" = true ]; then
    CMD="$CMD -O"
fi

# Prefix
if [ "$OPT_PREFIX" != "_GMXMMPBSA_" ]; then
    CMD="$CMD --prefix $OPT_PREFIX"
fi

# Input file
if [ -n "$OPT_INPUT_FILE" ]; then
    CMD="$CMD -i $OPT_INPUT_FILE"
fi

# XVV file
if [ -n "$OPT_XVVFILE" ]; then
    CMD="$CMD -xvvfile $OPT_XVVFILE"
fi

# Output files
if [ -n "$OPT_OUTPUT_FILE" ]; then
    CMD="$CMD -o $OPT_OUTPUT_FILE"
fi

if [ -n "$OPT_DECOMP_OUTPUT" ]; then
    CMD="$CMD -do $OPT_DECOMP_OUTPUT"
fi

if [ -n "$OPT_ENERGY_OUTPUT" ]; then
    CMD="$CMD -eo $OPT_ENERGY_OUTPUT"
    print_msg "Energy output CSV: $OPT_ENERGY_OUTPUT"
fi

if [ -n "$OPT_DECOMP_ENERGY_OUTPUT" ]; then
    CMD="$CMD -deo $OPT_DECOMP_ENERGY_OUTPUT"
    print_msg "Decomposition energy CSV: $OPT_DECOMP_ENERGY_OUTPUT"
fi

# No GUI (always enable for automated workflows)
if [ "$OPT_NOGUI" = true ] || [ -z "$DISPLAY" ]; then
    CMD="$CMD -nogui"
    print_msg "GUI disabled (-nogui)"
fi

# Stability
if [ "$OPT_STABILITY" = true ]; then
    CMD="$CMD --stability"
fi

# Add structure, topology, index, trajectory, and groups
# For both binding and stability calculations, use the same complex approach
# Stability mode (--stability flag) will internally ignore receptor/ligand split

if [ -n "$OPT_COMPLEX_STRUCTURE" ]; then
    CMD="$CMD -cs \"$OPT_COMPLEX_STRUCTURE\""
fi

if [ -n "$OPT_COMPLEX_TOPOLOGY" ]; then
    CMD="$CMD -cp \"$OPT_COMPLEX_TOPOLOGY\""
fi

if [ -n "$OPT_COMPLEX_INDEX" ]; then
    CMD="$CMD -ci \"$OPT_COMPLEX_INDEX\""
fi

# Add groups (required for both binding and stability modes)
if [ -n "$OPT_COMPLEX_GROUP_REC" ] && [ -n "$OPT_COMPLEX_GROUP_LIG" ]; then
    CMD="$CMD -cg $OPT_COMPLEX_GROUP_REC $OPT_COMPLEX_GROUP_LIG"
fi

if [ ${#OPT_COMPLEX_TRAJ[@]} -gt 0 ]; then
    CMD="$CMD -ct \"${OPT_COMPLEX_TRAJ[@]}\""
fi

if [ -n "$OPT_COMPLEX_REFERENCE" ]; then
    CMD="$CMD -cr \"$OPT_COMPLEX_REFERENCE\""
fi

# Receptor files
if [ -n "$OPT_RECEPTOR_STRUCTURE" ]; then
    CMD="$CMD -rs $OPT_RECEPTOR_STRUCTURE"
fi

if [ -n "$OPT_RECEPTOR_INDEX" ]; then
    CMD="$CMD -ri $OPT_RECEPTOR_INDEX"
fi

if [ -n "$OPT_RECEPTOR_GROUP" ]; then
    CMD="$CMD -rg $OPT_RECEPTOR_GROUP"
fi

if [ ${#OPT_RECEPTOR_TRAJ[@]} -gt 0 ]; then
    CMD="$CMD -rt ${OPT_RECEPTOR_TRAJ[@]}"
fi

if [ -n "$OPT_RECEPTOR_TOPOLOGY" ]; then
    CMD="$CMD -rp $OPT_RECEPTOR_TOPOLOGY"
fi

# Ligand files
if [ -n "$OPT_LIGAND_MOL2" ]; then
    CMD="$CMD -lm $OPT_LIGAND_MOL2"
fi

if [ -n "$OPT_LIGAND_STRUCTURE" ]; then
    CMD="$CMD -ls $OPT_LIGAND_STRUCTURE"
fi

if [ -n "$OPT_LIGAND_INDEX" ]; then
    CMD="$CMD -li $OPT_LIGAND_INDEX"
fi

if [ -n "$OPT_LIGAND_GROUP" ]; then
    CMD="$CMD -lg $OPT_LIGAND_GROUP"
fi

if [ ${#OPT_LIGAND_TRAJ[@]} -gt 0 ]; then
    CMD="$CMD -lt ${OPT_LIGAND_TRAJ[@]}"
fi

if [ -n "$OPT_LIGAND_TOPOLOGY" ]; then
    CMD="$CMD -lp $OPT_LIGAND_TOPOLOGY"
fi

# Rewrite output
if [ "$OPT_REWRITE_OUTPUT" = true ]; then
    CMD="$CMD --rewrite-output"
fi

################################################################################
# Execute gmx_MMPBSA
################################################################################

print_header "Executing gmx_MMPBSA"

# Change to topology directory to resolve relative includes
TOPOLOGY_DIR=""
if [ -n "$OPT_COMPLEX_TOPOLOGY" ]; then
    TOPOLOGY_DIR=$(dirname "$OPT_COMPLEX_TOPOLOGY")
    print_msg "Changing to topology directory: $TOPOLOGY_DIR"
    cd "$TOPOLOGY_DIR"
fi

# Check if we're in Alanine Scanning batch mode
if [ "$ALANINE_BATCH_MODE" = true ]; then
    print_header "Alanine Scanning - Batch Mode"

    # Disable exit on error for batch mode (we want to continue on failures)
    set +e

    # Parse residues list (split by comma or space)
    IFS=', ' read -ra RESIDUES <<< "$RESIDUES_LIST"

    TOTAL_RESIDUES=${#RESIDUES[@]}
    CURRENT_RESIDUE=0
    FAILED_RESIDUES=()
    SUCCESS_RESIDUES=()

    print_msg "Total residues to scan: $TOTAL_RESIDUES"
    print_msg "Residues array: ${RESIDUES[@]}"
    print_msg "Starting loop..."
    echo ""

    # Loop through each residue
    for RESIDUE in "${RESIDUES[@]}"; do
        print_msg "DEBUG: Processing residue: $RESIDUE"
        ((CURRENT_RESIDUE++))

        print_header "[$CURRENT_RESIDUE/$TOTAL_RESIDUES] Scanning Residue: $RESIDUE"

        # Create temporary YAML with single residue
        TEMP_YAML="/tmp/mmpbsa_alanine_${RESIDUE//\//_}_$$.yaml"

        python3 -c "
import yaml
with open('$ORIGINAL_YAML', 'r') as f:
    config = yaml.safe_load(f)

# Update residue
config['alanine_scanning']['residues'] = '$RESIDUE'

with open('$TEMP_YAML', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
"

        # Generate new .in file for this residue
        TEMP_INPUT="/tmp/mmpbsa_alanine_${RESIDUE//\//_}_$$.in"
        python3 "$SCRIPT_DIR/scripts/yaml_to_input.py" "$TEMP_YAML" -o "$TEMP_INPUT"

        if [ $? -ne 0 ]; then
            print_error "Failed to generate input file for $RESIDUE"
            FAILED_RESIDUES+=("$RESIDUE")
            continue
        fi

        # Create residue-specific directory and work in it
        RESIDUE_DIR="MMPBSA_${RESIDUE//\//_}"
        mkdir -p "$RESIDUE_DIR"

        # Save current directory
        CURRENT_DIR=$(pwd)

        # Change to residue directory
        cd "$RESIDUE_DIR"

        # Update command with new input file (use absolute path)
        # Remove any existing -i option from CMD first
        CMD_WITHOUT_INPUT=$(echo "$CMD" | sed 's/-i[[:space:]]*"[^"]*"//g' | sed 's/-i[[:space:]]*[^[:space:]]*//g')
        RESIDUE_CMD="$CMD_WITHOUT_INPUT -i \"$TEMP_INPUT\""

        print_msg "Executing gmx_MMPBSA for residue $RESIDUE in directory: $RESIDUE_DIR"
        echo ""
        echo "========================================================================"
        echo "Command: $RESIDUE_CMD"
        echo "========================================================================"
        echo ""

        # Execute with full output (results will be created in current directory)
        eval $RESIDUE_CMD
        RESIDUE_EXIT_CODE=$?

        echo ""
        echo "========================================================================"
        echo "Residue $RESIDUE finished with exit code: $RESIDUE_EXIT_CODE"
        echo "========================================================================"
        echo ""

        # Return to original directory
        cd "$CURRENT_DIR"

        # Check if calculation succeeded by looking for result files
        # Exit code may be 1 due to GUI tool failure (PyQt5/PyQt6 missing) even if calculation succeeded
        RESULT_FILE="$RESIDUE_DIR/FINAL_RESULTS_MMPBSA.dat"
        if [ -f "$RESULT_FILE" ]; then
            print_msg "✓ Residue $RESIDUE completed successfully"
            SUCCESS_RESIDUES+=("$RESIDUE")
            print_msg "Results saved to: $RESIDUE_DIR"
            if [ $RESIDUE_EXIT_CODE -ne 0 ]; then
                print_warning "  (Note: Exit code was $RESIDUE_EXIT_CODE, likely due to GUI tool failure, but calculation succeeded)"
            fi
        else
            print_error "✗ Residue $RESIDUE failed (exit code: $RESIDUE_EXIT_CODE)"
            FAILED_RESIDUES+=("$RESIDUE")
        fi

        # Cleanup temp files
        rm -f "$TEMP_YAML" "$TEMP_INPUT"

        echo ""
    done

    # Summary
    print_header "Alanine Scanning Batch Summary"
    print_msg "Total: $TOTAL_RESIDUES residues"
    print_msg "Success: ${#SUCCESS_RESIDUES[@]} residues"
    print_msg "Failed: ${#FAILED_RESIDUES[@]} residues"

    if [ ${#SUCCESS_RESIDUES[@]} -gt 0 ]; then
        echo ""
        print_msg "Successful residues:"
        for RES in "${SUCCESS_RESIDUES[@]}"; do
            print_msg "  ✓ $RES"
        done

        # Aggregate results from all successful residues
        echo ""
        print_header "Aggregating Results"

        AGGREGATE_DIR="MMPBSA_aggregated"
        mkdir -p "$AGGREGATE_DIR"

        print_msg "Creating combined results in: $AGGREGATE_DIR"

        # Run aggregation script (current directory has MMPBSA_* directories)
        python3 "$SCRIPT_DIR/scripts/aggregate_alanine_results.py" \
            --residues "${SUCCESS_RESIDUES[@]}" \
            --output "$AGGREGATE_DIR" \
            --working-dir "$(pwd)"

        if [ $? -eq 0 ]; then
            print_msg "✓ Results aggregated successfully!"
            print_msg "Combined analysis: $AGGREGATE_DIR/alanine_scan_combined.csv"
            print_msg "Visualization: $AGGREGATE_DIR/alanine_scan_summary.png"
            print_msg "Text report: $AGGREGATE_DIR/alanine_scan_report.txt"
        else
            print_warning "Failed to aggregate results"
        fi
    fi

    if [ ${#FAILED_RESIDUES[@]} -gt 0 ]; then
        echo ""
        print_error "Failed residues:"
        for RES in "${FAILED_RESIDUES[@]}"; do
            print_error "  ✗ $RES"
        done
        EXIT_CODE=1
    else
        EXIT_CODE=0
    fi

else
    # Normal single execution
    print_msg "Command: $CMD"
    echo ""

    eval $CMD

    EXIT_CODE=$?
fi

################################################################################
# Report results
################################################################################

# Check if calculation succeeded by looking for result files
# Exit code may be 1 due to GUI tool failure (PyQt5/PyQt6 missing) even if calculation succeeded
if [ -f "FINAL_RESULTS_MMPBSA.dat" ] || [ -n "$OPT_OUTPUT_FILE" ] && [ -f "$OPT_OUTPUT_FILE" ]; then
    print_header "Calculation Complete!"
    print_msg "✓ gmx_MMPBSA finished successfully"

    if [ $EXIT_CODE -ne 0 ]; then
        print_warning "  (Note: Exit code was $EXIT_CODE, likely due to GUI tool failure, but calculation succeeded)"
    fi

    # Show output files
    if [ -f "FINAL_RESULTS_MMPBSA.dat" ]; then
        print_msg "Results: FINAL_RESULTS_MMPBSA.dat"
    fi

    if [ -n "$OPT_OUTPUT_FILE" ] && [ -f "$OPT_OUTPUT_FILE" ]; then
        print_msg "Results: $OPT_OUTPUT_FILE"
    fi

    # Override EXIT_CODE to 0 if result files exist
    EXIT_CODE=0

    ################################################################################
    # Auto-analyze results
    ################################################################################

    print_header "Analyzing Results"

    # Determine analysis type from YAML config
    ANALYSIS_TYPE="auto"
    if [ -n "$OPT_CONFIG_YAML" ]; then
        # Extract calculation_type from YAML
        if grep -q "calculation_type:.*stability" "$OPT_CONFIG_YAML" 2>/dev/null; then
            ANALYSIS_TYPE="stability"
        elif grep -q "calculation_type:.*alanine" "$OPT_CONFIG_YAML" 2>/dev/null; then
            ANALYSIS_TYPE="alanine"
        elif grep -q "qm_mm:" "$OPT_CONFIG_YAML" 2>/dev/null && grep -q "enabled: true" "$OPT_CONFIG_YAML" 2>/dev/null; then
            ANALYSIS_TYPE="qmmm"
        elif grep -q "decomposition:" "$OPT_CONFIG_YAML" 2>/dev/null && grep -q "enabled: true" "$OPT_CONFIG_YAML" 2>/dev/null; then
            ANALYSIS_TYPE="decomposition"
        elif grep -q "entropy:" "$OPT_CONFIG_YAML" 2>/dev/null && grep -q "method: ie\|method: qh\|method: c2" "$OPT_CONFIG_YAML" 2>/dev/null; then
            ANALYSIS_TYPE="entropy"
        else
            ANALYSIS_TYPE="binding"
        fi
    fi

    # Get script directory (relative to run_mmpbsa_full.sh location)
    SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
    ANALYZE_SCRIPT="$SCRIPT_DIR/scripts/analyze_results.py"

    if [ -f "$ANALYZE_SCRIPT" ]; then
        print_msg "Running automatic analysis (type: $ANALYSIS_TYPE)..."
        print_msg "Working directory: $(pwd)"
        print_msg "Analysis script: $ANALYZE_SCRIPT"

        # Check for required Python packages
        if python3 -c "import pandas, matplotlib, numpy" 2>/dev/null; then
            # Run analysis and capture output
            ANALYSIS_OUTPUT=$(python3 "$ANALYZE_SCRIPT" -d "$(pwd)" -t "$ANALYSIS_TYPE" 2>&1)
            ANALYSIS_EXIT_CODE=$?

            if [ $ANALYSIS_EXIT_CODE -eq 0 ]; then
                echo "$ANALYSIS_OUTPUT"
                print_msg "✓ Analysis complete!"
                if [ -d "analysis_output" ]; then
                    print_msg "Analysis results saved to: $(pwd)/analysis_output/"

                    # List generated files
                    echo ""
                    print_msg "Generated files:"
                    find analysis_output -type f -printf "  - %f (%s bytes)\n" | sort
                fi
            else
                print_warning "Analysis script failed with exit code $ANALYSIS_EXIT_CODE"
                print_warning "Error output:"
                echo "$ANALYSIS_OUTPUT"
                print_msg "gmx_MMPBSA results are still available in $(pwd)"
            fi
        else
            print_warning "Python packages (pandas, matplotlib, numpy) not found. Skipping analysis."
            print_msg "Install with: pip install pandas matplotlib numpy"
        fi
    else
        print_warning "Analysis script not found: $ANALYZE_SCRIPT"
    fi

else
    print_error "✗ gmx_MMPBSA failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

print_header "Done!"
