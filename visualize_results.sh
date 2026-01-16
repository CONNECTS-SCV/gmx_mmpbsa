#!/bin/bash

################################################################################
# gmx_MMPBSA Results Visualization Script
# Standalone visualization for completed gmx_MMPBSA analyses
################################################################################

set -e

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
    cat << 'USAGE'
Usage: ./visualize_results.sh -n <analysis_number> -d <directory> [OPTIONS]

Visualize gmx_MMPBSA results based on analysis type

═══════════════════════════════════════════════════════════════════════════
REQUIRED OPTIONS
═══════════════════════════════════════════════════════════════════════════
  -n, --number N          Analysis number (1-6)
                          1 = Binding Free Energy
                          2 = Stability
                          3 = Alanine Scanning
                          4 = QM/MMGBSA
                          5 = Entropy Correction
                          6 = Decomposition
  
  -d, --directory DIR     Directory containing FINAL_RESULTS_MMPBSA.dat

═══════════════════════════════════════════════════════════════════════════
OPTIONAL
═══════════════════════════════════════════════════════════════════════════
  -o, --output DIR        Output directory (default: analysis_output)
  -h, --help              Show this help message

═══════════════════════════════════════════════════════════════════════════
EXAMPLES
═══════════════════════════════════════════════════════════════════════════

1. Visualize binding free energy results:
   ./visualize_results.sh -n 1 -d /path/to/results

2. Visualize stability with custom output:
   ./visualize_results.sh -n 2 -d ./my_results -o ./my_plots

3. Visualize alanine scanning:
   ./visualize_results.sh -n 3 -d example/

═══════════════════════════════════════════════════════════════════════════
REQUIREMENTS
═══════════════════════════════════════════════════════════════════════════
- Python 3 with pandas, matplotlib, numpy
- FINAL_RESULTS_MMPBSA.dat in the specified directory
- Optional: FINAL_DECOMP_MMPBSA.dat for decomposition analysis

USAGE
}

# Initialize variables
ANALYSIS_NUMBER=""
RESULTS_DIR=""
OUTPUT_DIR=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--number)
            ANALYSIS_NUMBER="$2"
            shift 2
            ;;
        -d|--directory)
            RESULTS_DIR="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use -h or --help for usage"
            exit 1
            ;;
    esac
done

################################################################################
# Validate inputs
################################################################################

if [ -z "$ANALYSIS_NUMBER" ]; then
    print_error "Analysis number is required (-n)"
    echo "Use -h for help"
    exit 1
fi

if [ -z "$RESULTS_DIR" ]; then
    print_error "Results directory is required (-d)"
    echo "Use -h for help"
    exit 1
fi

if [ ! -d "$RESULTS_DIR" ]; then
    print_error "Directory not found: $RESULTS_DIR"
    exit 1
fi

# Validate analysis number
if ! [[ "$ANALYSIS_NUMBER" =~ ^[1-6]$ ]]; then
    print_error "Analysis number must be 1-6, got: $ANALYSIS_NUMBER"
    exit 1
fi

# Map analysis number to type
case $ANALYSIS_NUMBER in
    1) ANALYSIS_TYPE="binding" ;;
    2) ANALYSIS_TYPE="stability" ;;
    3) ANALYSIS_TYPE="alanine" ;;
    4) ANALYSIS_TYPE="qmmm" ;;
    5) ANALYSIS_TYPE="entropy" ;;
    6) ANALYSIS_TYPE="decomposition" ;;
esac

print_header "gmx_MMPBSA Results Visualization"
print_msg "Analysis Type: $ANALYSIS_NUMBER - $ANALYSIS_TYPE"
print_msg "Results Directory: $RESULTS_DIR"

################################################################################
# Check for required files
################################################################################

FINAL_RESULTS="$RESULTS_DIR/FINAL_RESULTS_MMPBSA.dat"
if [ ! -f "$FINAL_RESULTS" ]; then
    print_error "FINAL_RESULTS_MMPBSA.dat not found in $RESULTS_DIR"
    print_msg "Expected: $FINAL_RESULTS"
    exit 1
fi

print_msg "✓ Found FINAL_RESULTS_MMPBSA.dat"

# Check for optional decomposition file
DECOMP_FILE="$RESULTS_DIR/FINAL_DECOMP_MMPBSA.dat"
if [ -f "$DECOMP_FILE" ]; then
    print_msg "✓ Found FINAL_DECOMP_MMPBSA.dat"
fi

################################################################################
# Check dependencies
################################################################################

# Detect Python interpreter (prefer conda env if active)
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    PYTHON_CMD="python3"
    print_msg "Using conda environment: $CONDA_DEFAULT_ENV"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    print_error "python3 not found!"
    exit 1
fi

# Check for required packages
if ! $PYTHON_CMD -c "import pandas, matplotlib, numpy" 2>/dev/null; then
    print_error "Required Python packages not found!"
    print_msg "Current Python: $($PYTHON_CMD --version)"

    if [ -n "$CONDA_DEFAULT_ENV" ]; then
        print_msg "Install with: pip install pandas matplotlib numpy"
    else
        print_warning "Consider using conda environment 'gmxmmpbsa'"
        print_msg "Or install globally: pip install pandas matplotlib numpy"
    fi
    exit 1
fi

print_msg "✓ Python dependencies OK ($($PYTHON_CMD --version 2>&1 | head -1))"

################################################################################
# Run visualization
################################################################################

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ANALYZE_SCRIPT="$SCRIPT_DIR/scripts/analyze_results.py"

if [ ! -f "$ANALYZE_SCRIPT" ]; then
    print_error "Analysis script not found: $ANALYZE_SCRIPT"
    exit 1
fi

print_header "Running Visualization"

# Change to results directory
cd "$RESULTS_DIR"

print_msg "Working directory: $(pwd)"
print_msg "Analysis script: $ANALYZE_SCRIPT"

# Run analysis
OUTPUT=$($PYTHON_CMD "$ANALYZE_SCRIPT" -d "$(pwd)" -t "$ANALYSIS_TYPE" 2>&1)
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "$OUTPUT"
    print_msg "✓ Visualization complete!"
    
    # Show generated files
    if [ -d "analysis_output" ]; then
        echo ""
        print_header "Generated Files"
        find analysis_output -type f -printf "  %p (%s bytes)\n" | sort
        
        # Get analysis-specific folder
        case $ANALYSIS_NUMBER in
            1) ANALYSIS_FOLDER="1_binding_energy" ;;
            2) ANALYSIS_FOLDER="2_stability" ;;
            3) ANALYSIS_FOLDER="3_alanine_scanning" ;;
            4) ANALYSIS_FOLDER="4_qm_mmgbsa" ;;
            5) ANALYSIS_FOLDER="5_entropy_correction" ;;
            6) ANALYSIS_FOLDER="6_decomposition" ;;
        esac
        
        echo ""
        print_msg "Results saved to: $(pwd)/analysis_output/$ANALYSIS_FOLDER/"
        
        # Show summary if exists
        SUMMARY="analysis_output/$ANALYSIS_FOLDER/summary_report.txt"
        if [ -f "$SUMMARY" ]; then
            echo ""
            print_header "Summary Report"
            cat "$SUMMARY"
        fi
    fi
else
    print_error "Visualization failed with exit code $EXIT_CODE"
    echo ""
    print_error "Error output:"
    echo "$OUTPUT"
    exit $EXIT_CODE
fi

print_header "Done!"
