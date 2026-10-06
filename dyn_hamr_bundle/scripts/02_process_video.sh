#!/bin/bash
# =============================================================================
# Dyn-HaMR Video Processing Script
# =============================================================================
# Processes a video through Dyn-HaMR pipeline
# Usage: bash 02_process_video.sh <video_filename>
# Example: bash 02_process_video.sh hand_test.mp4
# =============================================================================

set -e

if [ -z "$1" ]; then
    echo "Usage: bash 02_process_video.sh <video_filename>"
    echo "Example: bash 02_process_video.sh hand_test.mp4"
    exit 1
fi

VIDEO_NAME="$1"
VIDEO_BASE="${VIDEO_NAME%.*}"  # Remove extension

WORK_DIR="$HOME/dyn_hamr_workspace"
DYNHAMR_DIR="$WORK_DIR/Dyn-HaMR"
VIDEO_DIR="$DYNHAMR_DIR/test/videos"
OUTPUT_DIR="$WORK_DIR/outputs/$VIDEO_BASE"

echo "=========================================="
echo "Dyn-HaMR Video Processing"
echo "=========================================="
echo "Video: $VIDEO_NAME"
echo "Output: $OUTPUT_DIR"
echo ""

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate dynhamr

# Check video exists
if [ ! -f "$VIDEO_DIR/$VIDEO_NAME" ]; then
    echo "ERROR: Video not found at $VIDEO_DIR/$VIDEO_NAME"
    echo "Please copy your video to: $VIDEO_DIR/"
    exit 1
fi

# Check MANO exists
MANO_FILE="$DYNHAMR_DIR/_DATA/data/mano/MANO_RIGHT.pkl"
if [ ! -f "$MANO_FILE" ]; then
    echo "ERROR: MANO_RIGHT.pkl not found at $MANO_FILE"
    echo "Please download MANO and place it there."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

cd "$DYNHAMR_DIR"

echo "[1/3] Running Dyn-HaMR optimization..."
echo "  This may take 10-30 minutes depending on video length."
echo ""

# Run Dyn-HaMR
# Note: Adjust config paths based on actual repo structure
python -u run_opt.py \
    data=video_vipe \
    data.root="$VIDEO_DIR" \
    data.seq="$VIDEO_BASE" \
    run_opt=True \
    run_vis=True \
    is_static=False \
    output_dir="$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/dynhamr_log.txt"

echo ""
echo "[2/3] Locating output files..."

# Find the output param files (structure varies by Dyn-HaMR version)
PARAM_FILE=$(find "$OUTPUT_DIR" -name "*.npz" -o -name "*params*.pkl" | head -1)

if [ -z "$PARAM_FILE" ]; then
    echo "  Looking in default Dyn-HaMR output locations..."
    PARAM_FILE=$(find "$DYNHAMR_DIR/outputs" -name "*.npz" -newer "$OUTPUT_DIR/dynhamr_log.txt" | head -1)
fi

if [ -n "$PARAM_FILE" ]; then
    echo "  Found parameters: $PARAM_FILE"
    cp "$PARAM_FILE" "$OUTPUT_DIR/raw_params.npz" 2>/dev/null || \
    cp "$PARAM_FILE" "$OUTPUT_DIR/raw_params.pkl" 2>/dev/null || true
else
    echo "  WARNING: Could not locate parameter file automatically."
    echo "  Check $DYNHAMR_DIR/outputs/ for results."
fi

echo ""
echo "[3/3] Collecting renders..."

# Copy any rendered outputs
find "$DYNHAMR_DIR/outputs" -name "*.mp4" -newer "$OUTPUT_DIR/dynhamr_log.txt" -exec cp {} "$OUTPUT_DIR/" \; 2>/dev/null || true
find "$DYNHAMR_DIR/outputs" -name "*.png" -newer "$OUTPUT_DIR/dynhamr_log.txt" -exec cp {} "$OUTPUT_DIR/" \; 2>/dev/null || true

echo ""
echo "=========================================="
echo "Dyn-HaMR processing complete!"
echo "=========================================="
echo ""
echo "Output directory: $OUTPUT_DIR"
echo "Contents:"
ls -la "$OUTPUT_DIR"
echo ""
echo "Next: Run post-smoothing with:"
echo "  python 03_smooth_params.py $OUTPUT_DIR/raw_params.npz"
echo "=========================================="
