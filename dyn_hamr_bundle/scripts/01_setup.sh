#!/bin/bash
# =============================================================================
# Dyn-HaMR Setup Script for WSL2 + NVIDIA GPU
# =============================================================================
# Run this ONCE to set up the environment
# Usage: bash 01_setup.sh
# =============================================================================

set -e  # Exit on error

echo "=========================================="
echo "Dyn-HaMR Setup for WSL2"
echo "=========================================="

# Check CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Make sure CUDA is installed in WSL2."
    echo "See: https://docs.nvidia.com/cuda/wsl-user-guide/index.html"
    exit 1
fi

echo "CUDA detected:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo ""

# Create working directory
WORK_DIR="$HOME/dyn_hamr_workspace"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

echo "Working directory: $WORK_DIR"
echo ""

# =============================================================================
# Step 1: Clone Dyn-HaMR
# =============================================================================
echo "[1/5] Cloning Dyn-HaMR..."
if [ -d "Dyn-HaMR" ]; then
    echo "  Dyn-HaMR already exists, pulling latest..."
    cd Dyn-HaMR && git pull && cd ..
else
    git clone --recursive https://github.com/ZhengdiYu/Dyn-HaMR.git
fi
echo ""

# =============================================================================
# Step 2: Create conda environment
# =============================================================================
echo "[2/5] Setting up conda environment..."

# Check if conda exists
if ! command -v conda &> /dev/null; then
    echo "  Conda not found. Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    eval "$($HOME/miniconda/bin/conda shell.bash hook)"
    conda init
    echo "  Miniconda installed. Please restart terminal and re-run this script."
    exit 0
fi

# Create environment
cd Dyn-HaMR
if conda env list | grep -q "dynhamr"; then
    echo "  Environment 'dynhamr' already exists"
else
    echo "  Creating conda environment..."
    conda create -n dynhamr python=3.10 -y
fi

# Activate and install dependencies
eval "$(conda shell.bash hook)"
conda activate dynhamr

echo "  Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "  Installing other dependencies..."
pip install -r requirements.txt 2>/dev/null || echo "  (requirements.txt may need manual review)"

# Install pytorch3d (can be tricky)
echo "  Installing pytorch3d..."
pip install "git+https://github.com/facebookresearch/pytorch3d.git" || {
    echo "  pytorch3d pip install failed, trying conda..."
    conda install pytorch3d -c pytorch3d -y || echo "  WARNING: pytorch3d install may need manual attention"
}

echo ""

# =============================================================================
# Step 3: Download checkpoints and assets
# =============================================================================
echo "[3/5] Downloading checkpoints (this may take a while)..."
cd "$WORK_DIR/Dyn-HaMR"

if [ -f "prepare.sh" ]; then
    source prepare.sh || echo "  prepare.sh had some issues, continuing..."
else
    echo "  WARNING: prepare.sh not found. You may need to download checkpoints manually."
fi

echo ""

# =============================================================================
# Step 4: Set up MANO
# =============================================================================
echo "[4/5] Setting up MANO..."

MANO_DIR="$WORK_DIR/Dyn-HaMR/_DATA/data/mano"
mkdir -p "$MANO_DIR"

if [ -f "$MANO_DIR/MANO_RIGHT.pkl" ]; then
    echo "  MANO_RIGHT.pkl already exists"
else
    echo ""
    echo "  =================================================="
    echo "  ACTION REQUIRED: Download MANO model"
    echo "  =================================================="
    echo "  1. Go to: https://mano.is.tue.mpg.de/"
    echo "  2. Register/login and download MANO"
    echo "  3. Copy MANO_RIGHT.pkl to:"
    echo "     $MANO_DIR/MANO_RIGHT.pkl"
    echo "  =================================================="
    echo ""
fi

echo ""

# =============================================================================
# Step 5: Set up Hand-BMC constraints
# =============================================================================
echo "[5/5] Setting up biomechanical constraints..."

cd "$WORK_DIR"

if [ -d "Hand-BMC-pytorch" ]; then
    echo "  Hand-BMC-pytorch already exists"
else
    git clone https://github.com/MengHao666/Hand-BMC-pytorch.git
fi

BMC_OUTPUT_DIR="$WORK_DIR/Dyn-HaMR/dyn_hamr/optim/BMC"
mkdir -p "$BMC_OUTPUT_DIR"

# Check if BMC files already exist
if [ -f "$BMC_OUTPUT_DIR/bone_len_max.npy" ]; then
    echo "  BMC constraint files already exist"
else
    echo "  BMC files need to be generated. Run:"
    echo "    cd $WORK_DIR/Hand-BMC-pytorch"
    echo "    conda activate dynhamr"
    echo "    python calculate_bmc.py"
    echo "    python calculate_convex_hull.py"
    echo "  Then copy .npy files to: $BMC_OUTPUT_DIR/"
fi

echo ""

# =============================================================================
# Done
# =============================================================================
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Ensure MANO_RIGHT.pkl is in place (see above)"
echo "  2. Generate BMC constraints if needed (see above)"
echo "  3. Copy your input video to: $WORK_DIR/Dyn-HaMR/test/videos/"
echo "  4. Run: bash 02_process_video.sh <video_name>"
echo ""
echo "Workspace: $WORK_DIR"
echo "=========================================="
