#!/bin/bash

################################################################################
# GROMACS to gmx_MMPBSA Input Preparation Script
# For CHARMM36m force field
################################################################################

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_msg() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if required arguments are provided
if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <tpr_file> <trajectory_file> <receptor_index> <ligand_index> [start_time] [end_time] [step]"
    echo ""
    echo "Arguments:"
    echo "  tpr_file        : GROMACS .tpr file"
    echo "  trajectory_file : GROMACS trajectory (.xtc or .trr)"
    echo "  receptor_index  : Index group for receptor (e.g., 1 for Protein)"
    echo "  ligand_index    : Index group for ligand (e.g., 13 for specific residue)"
    echo "  start_time      : Start time in ps (optional, default: 0)"
    echo "  end_time        : End time in ps (optional, default: last frame)"
    echo "  step            : Frame step (optional, default: 1)"
    echo ""
    echo "Example:"
    echo "  $0 md.tpr md_fit.xtc 1 13 50000 100000 10"
    exit 1
fi

# Input parameters
TPR_FILE="$1"
TRAJ_FILE="$2"
RECEPTOR_IDX="$3"
LIGAND_IDX="$4"
START_TIME="${5:-0}"
END_TIME="${6:-}"
STEP="${7:-1}"

# Output directory
OUTPUT_DIR="mmpbsa_input"
mkdir -p "$OUTPUT_DIR"

print_msg "================================================"
print_msg "GROMACS to gmx_MMPBSA Input Preparation"
print_msg "================================================"
print_msg "TPR file: $TPR_FILE"
print_msg "Trajectory: $TRAJ_FILE"
print_msg "Receptor index: $RECEPTOR_IDX"
print_msg "Ligand index: $LIGAND_IDX"
print_msg "Output directory: $OUTPUT_DIR"
print_msg "================================================"

# Check if files exist
if [ ! -f "$TPR_FILE" ]; then
    print_error "TPR file not found: $TPR_FILE"
    exit 1
fi

if [ ! -f "$TRAJ_FILE" ]; then
    print_error "Trajectory file not found: $TRAJ_FILE"
    exit 1
fi

# Extract complex structure
print_msg "Step 1: Extracting complex structure..."
echo -e "${RECEPTOR_IDX}\n${LIGAND_IDX}" | gmx trjconv -s "$TPR_FILE" -f "$TRAJ_FILE" \
    -o "$OUTPUT_DIR/complex.pdb" -dump 0 -pbc mol -ur compact &>/dev/null
print_msg "Complex structure saved: $OUTPUT_DIR/complex.pdb"

# Create index file for receptor and ligand
print_msg "Step 2: Creating index file..."
echo -e "${RECEPTOR_IDX}\n${LIGAND_IDX}\nq" | gmx make_ndx -f "$TPR_FILE" \
    -o "$OUTPUT_DIR/index.ndx" &>/dev/null
print_msg "Index file created: $OUTPUT_DIR/index.ndx"

# Extract receptor trajectory
print_msg "Step 3: Extracting receptor trajectory..."
CMD="gmx trjconv -s $TPR_FILE -f $TRAJ_FILE -o $OUTPUT_DIR/receptor.xtc -n $OUTPUT_DIR/index.ndx -pbc mol -ur compact"
[ -n "$START_TIME" ] && CMD="$CMD -b $START_TIME"
[ -n "$END_TIME" ] && CMD="$CMD -e $END_TIME"
[ -n "$STEP" ] && CMD="$CMD -skip $STEP"
echo "$RECEPTOR_IDX" | eval $CMD &>/dev/null
print_msg "Receptor trajectory saved: $OUTPUT_DIR/receptor.xtc"

# Extract ligand trajectory
print_msg "Step 4: Extracting ligand trajectory..."
CMD="gmx trjconv -s $TPR_FILE -f $TRAJ_FILE -o $OUTPUT_DIR/ligand.xtc -n $OUTPUT_DIR/index.ndx -pbc mol -ur compact"
[ -n "$START_TIME" ] && CMD="$CMD -b $START_TIME"
[ -n "$END_TIME" ] && CMD="$CMD -e $END_TIME"
[ -n "$STEP" ] && CMD="$CMD -skip $STEP"
echo "$LIGAND_IDX" | eval $CMD &>/dev/null
print_msg "Ligand trajectory saved: $OUTPUT_DIR/ligand.xtc"

# Extract complex trajectory (both receptor and ligand)
print_msg "Step 5: Extracting complex trajectory..."
CMD="gmx trjconv -s $TPR_FILE -f $TRAJ_FILE -o $OUTPUT_DIR/complex.xtc -n $OUTPUT_DIR/index.ndx -pbc mol -ur compact"
[ -n "$START_TIME" ] && CMD="$CMD -b $START_TIME"
[ -n "$END_TIME" ] && CMD="$CMD -e $END_TIME"
[ -n "$STEP" ] && CMD="$CMD -skip $STEP"
echo -e "${RECEPTOR_IDX}\n${LIGAND_IDX}" | eval $CMD &>/dev/null
print_msg "Complex trajectory saved: $OUTPUT_DIR/complex.xtc"

# Extract reference structure (first frame)
print_msg "Step 6: Extracting reference structure..."
echo "0" | gmx trjconv -s "$TPR_FILE" -f "$TRAJ_FILE" -o "$OUTPUT_DIR/reference.pdb" \
    -dump "$START_TIME" -pbc mol -ur compact &>/dev/null
print_msg "Reference structure saved: $OUTPUT_DIR/reference.pdb"

# Create topology files for gmx_MMPBSA
print_msg "Step 7: Preparing topology information..."

# Extract receptor structure for topology
echo "$RECEPTOR_IDX" | gmx trjconv -s "$TPR_FILE" -f "$TRAJ_FILE" \
    -o "$OUTPUT_DIR/receptor.pdb" -dump 0 -pbc mol -ur compact &>/dev/null
print_msg "Receptor structure saved: $OUTPUT_DIR/receptor.pdb"

# Extract ligand structure for topology
echo "$LIGAND_IDX" | gmx trjconv -s "$TPR_FILE" -f "$TRAJ_FILE" \
    -o "$OUTPUT_DIR/ligand.pdb" -dump 0 -pbc mol -ur compact &>/dev/null
print_msg "Ligand structure saved: $OUTPUT_DIR/ligand.pdb"

# Get trajectory statistics
print_msg "Step 8: Analyzing trajectory statistics..."
NFRAMES=$(gmx check -f "$OUTPUT_DIR/complex.xtc" 2>&1 | grep "Step" | wc -l)
print_msg "Number of frames: $NFRAMES"

# Create summary file
cat > "$OUTPUT_DIR/preparation_summary.txt" << EOF
================================
Input Preparation Summary
================================
Date: $(date)
TPR file: $TPR_FILE
Trajectory: $TRAJ_FILE
Receptor index: $RECEPTOR_IDX
Ligand index: $LIGAND_IDX
Start time: $START_TIME ps
End time: ${END_TIME:-"last frame"} ps
Step: $STEP
Number of frames: $NFRAMES

Output Files:
-------------
- complex.pdb: Initial complex structure
- complex.xtc: Complex trajectory
- receptor.pdb: Receptor structure
- receptor.xtc: Receptor trajectory
- ligand.pdb: Ligand structure
- ligand.xtc: Ligand trajectory
- reference.pdb: Reference structure
- index.ndx: Index file

These files are ready for gmx_MMPBSA calculations.
================================
EOF

print_msg "================================================"
print_msg "Preparation complete!"
print_msg "Summary saved: $OUTPUT_DIR/preparation_summary.txt"
print_msg "================================================"
print_msg "Next steps:"
print_msg "1. Review the structures in $OUTPUT_DIR"
print_msg "2. Run gmx_MMPBSA calculations using the config files"
print_msg "================================================"

cat "$OUTPUT_DIR/preparation_summary.txt"
