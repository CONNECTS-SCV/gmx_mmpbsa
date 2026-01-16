#!/bin/bash

################################################################################
# Cleanup and Fresh Install Script (Conda Version)
# Creates conda environment in /home/connects/miniforge3/envs/
################################################################################

set -e

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

################################################################################
# Configuration
################################################################################
CONDA_ENV_NAME="gmxmmpbsa"
CONDA_BASE="/home/connects/miniforge3"

# Disable user site-packages to avoid conflicts with ~/.local packages
export PYTHONNOUSERSITE=1

# Set OpenMPI library paths
export LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"
export PATH="/usr/local/bin:$PATH"

print_header "gmx_MMPBSA Conda Installation"
echo ""
print_msg "Environment name: $CONDA_ENV_NAME"
print_msg "Location: $CONDA_BASE/envs/$CONDA_ENV_NAME"
echo ""
print_warning "This will:"
print_warning "  1. Remove existing conda environment (if exists)"
print_warning "  2. Create new conda environment"
print_warning "  3. Install gmx_MMPBSA and dependencies"
echo ""
read -p "Continue? (y/N): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_msg "Cancelled"
    exit 0
fi

################################################################################
# Step 1: Check conda
################################################################################
print_header "Step 1: Checking Conda"

if ! command -v conda &> /dev/null; then
    print_error "Conda not found!"
    print_error "Please install miniforge or miniconda first"
    exit 1
fi

print_msg "Conda found: $(which conda)"
print_msg "Conda version: $(conda --version)"

################################################################################
# Step 2: Remove old environment
################################################################################
print_header "Step 2: Removing Old Environment"

if conda env list | grep -q "^$CONDA_ENV_NAME "; then
    print_warning "Found existing environment: $CONDA_ENV_NAME"
    print_msg "Removing..."
    conda env remove -n "$CONDA_ENV_NAME" -y
    print_msg "✓ Old environment removed"
else
    print_msg "No existing environment found"
fi

# Also remove old venv if exists
if [ -d "gmx_mmpbsa_env" ]; then
    print_msg "Removing old venv directory..."
    rm -rf gmx_mmpbsa_env
fi

################################################################################
# Step 3: Create conda environment
################################################################################
print_header "Step 3: Creating Conda Environment"

print_msg "Creating environment: $CONDA_ENV_NAME (Python 3.9)"
conda create -n "$CONDA_ENV_NAME" python=3.9 -y

if [ $? -ne 0 ]; then
    print_error "Failed to create conda environment"
    exit 1
fi

print_msg "✓ Conda environment created"

################################################################################
# Step 4: Activate environment
################################################################################
print_header "Step 4: Activating Environment"

# Source conda (try multiple possible locations)
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
elif [ -f "$CONDA_BASE/lib/python3.12/site-packages/conda/shell/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/lib/python3.12/site-packages/conda/shell/etc/profile.d/conda.sh"
elif [ -f "$CONDA_BASE/lib/python3.1/site-packages/conda/shell/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/lib/python3.1/site-packages/conda/shell/etc/profile.d/conda.sh"
else
    print_error "conda.sh not found!"
    print_error "Please source conda manually: source \$(conda info --base)/etc/profile.d/conda.sh"
    exit 1
fi

# Activate environment
conda activate "$CONDA_ENV_NAME"

if [ $? -eq 0 ]; then
    print_msg "✓ Environment activated"
    print_msg "Python: $(which python)"
else
    print_error "Failed to activate environment"
    exit 1
fi

################################################################################
# Step 5: Skip conda dependencies (will be installed by pip with gmx_MMPBSA)
################################################################################
print_header "Step 5: Preparing for gmx_MMPBSA Installation"

print_msg "Skipping conda package installation (pip will handle dependencies)"
print_msg "✓ Ready for gmx_MMPBSA installation"

################################################################################
# Step 6: Install gmx_MMPBSA
################################################################################
print_header "Step 6: Installing gmx_MMPBSA"

# First install numpy 1.23.5 to avoid binary incompatibility with pandas
print_msg "Installing compatible numpy version first..."
python -m pip install "numpy==1.23.5"

print_msg "Installing gmx_MMPBSA via pip..."
python -m pip install gmx_MMPBSA --no-deps

# Now install remaining dependencies
print_msg "Installing dependencies from requirements.txt..."
if [ -f "requirements.txt" ]; then
    python -m pip install -r requirements.txt
else
    # Fallback to hardcoded versions
    print_warning "requirements.txt not found, using fallback versions..."
    python -m pip install "pandas==1.2.2" "matplotlib==3.5.2" "seaborn==0.11.2" "scipy>=1.6.1" "tqdm" "parmed>=4.2.2" "mpi4py<=3.1.5"
fi

if [ $? -eq 0 ]; then
    print_msg "✓ gmx_MMPBSA and dependencies installed successfully"

    VERSION=$(gmx_MMPBSA --version 2>&1 | head -1 || echo "unknown")
    print_msg "Version: $VERSION"
else
    print_error "gmx_MMPBSA installation failed"
    exit 1
fi

################################################################################
# Step 7: Verify MPI Support
################################################################################
print_header "Step 7: Verifying MPI Support"

if command -v mpirun &> /dev/null || command -v mpiexec &> /dev/null; then
    print_msg "System MPI found: $(mpirun --version | head -1)"

    if python -c "import mpi4py" 2>/dev/null; then
        print_msg "✓ mpi4py installed - parallel processing available"
        MPI_VERSION=$(python -c "import mpi4py; print(mpi4py.__version__)" 2>/dev/null)
        print_msg "mpi4py version: $MPI_VERSION"
    else
        print_warning "mpi4py not found (should have been installed with gmx_MMPBSA)"
    fi
else
    print_warning "System MPI not found"
    print_warning "gmx_MMPBSA will work in serial mode only"
fi

################################################################################
# Step 8: Verify installation
################################################################################
print_header "Step 8: Verifying Installation"

print_msg "Checking installations..."
echo ""

# gmx_MMPBSA
if command -v gmx_MMPBSA &> /dev/null; then
    VERSION=$(gmx_MMPBSA --version 2>&1 | head -1)
    print_msg "✓ gmx_MMPBSA: $VERSION"
else
    print_error "✗ gmx_MMPBSA not found!"
    exit 1
fi

# Python packages
python << 'PYEOF'
import sys

packages = {
    'numpy': 'numpy',
    'pandas': 'pandas',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'scipy': 'scipy',
    'mpi4py': 'mpi4py (optional)'
}

for pkg, name in packages.items():
    try:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', 'unknown')
        print(f'\033[0;32m✓\033[0m {name}: {version}')
    except ImportError:
        if 'optional' in name:
            print(f'\033[1;33m⚠\033[0m {name}: not installed (optional)')
        else:
            print(f'\033[0;31m✗\033[0m {name}: not installed')
            sys.exit(1)
PYEOF

################################################################################
# Step 9: Create activation helper
################################################################################
print_header "Step 9: Creating Helper Scripts"

# Activation script
cat > activate_env.sh << 'EOF'
#!/bin/bash
# Activation script for gmx_MMPBSA conda environment

CONDA_BASE="/home/connects/miniforge3"

# Source conda (try multiple possible locations)
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
elif [ -f "$CONDA_BASE/lib/python3.12/site-packages/conda/shell/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/lib/python3.12/site-packages/conda/shell/etc/profile.d/conda.sh"
elif [ -f "$CONDA_BASE/lib/python3.1/site-packages/conda/shell/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/lib/python3.1/site-packages/conda/shell/etc/profile.d/conda.sh"
else
    echo "Error: conda.sh not found!"
    return 1
fi

# Activate environment
conda activate gmxmmpbsa

echo "================================================"
echo "✓ gmx_MMPBSA Environment Activated"
echo "================================================"
echo "Conda env: $CONDA_ENV_NAME"
echo "Python: \$(which python)"
echo "pip: \$(which pip)"
echo "gmx_MMPBSA: \$(which gmx_MMPBSA)"
echo ""
echo "Run: ./run_mmpbsa_full.sh --help"
echo "Deactivate: conda deactivate"
echo "================================================"
EOF

chmod +x activate_env.sh

print_msg "✓ Created activate_env.sh"

################################################################################
# Completion
################################################################################
print_header "Installation Complete!"

echo ""
print_msg "✅ Conda environment ready: $CONDA_ENV_NAME"
print_msg "✅ Location: $CONDA_BASE/envs/$CONDA_ENV_NAME"
print_msg "✅ All packages installed"
echo ""
print_msg "Quick Start:"
echo ""
echo "  # Activate environment"
echo "  source activate_env.sh"
echo ""
echo "  # Or manually"
echo "  conda activate $CONDA_ENV_NAME"
echo ""
echo "  # Run analysis"
echo "  ./run_mmpbsa_full.sh -c configs/1_binding_free_energy.yaml -cs md.tpr -ct md.xtc -cg 1 13"
echo ""

# Check GROMACS
if ! command -v gmx &> /dev/null && ! command -v gmx_mpi &> /dev/null; then
    print_warning "⚠ GROMACS not found!"
    print_warning "gmx_MMPBSA requires GROMACS to be installed"
fi

# Check MPI
if ! python -c "import mpi4py" 2>/dev/null; then
    print_warning "⚠ MPI not available"
    print_warning "Parallel processing disabled (will run slower)"
fi

echo ""
print_header "Ready to Use!"

# Deactivate
conda deactivate 2>/dev/null || true
