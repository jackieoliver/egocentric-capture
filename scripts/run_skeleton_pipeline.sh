#!/bin/bash
# Haptica Skeleton Pipeline
# Full pipeline: GoPro video → Downsample (1080p/30fps) → Undistort → HaMeR 3-pass → Skeleton Overlay
#
# HaMeR 3-pass architecture:
#   Pass 1: YOLO hand detection (extract raw bboxes)
#   Pass 2: Bbox cleaning (remove hallucinations/overlaps)
#   Pass 3: HaMeR inference (MANO mesh prediction)
#
# Usage:
#   ./run_skeleton_pipeline.sh <input_video> <output_dir> [--skip-undistort]
#
# Example:
#   ./run_skeleton_pipeline.sh /path/to/GX010116.MP4 ./output/
#   ./run_skeleton_pipeline.sh /path/to/already_undistorted.mp4 ./output/ --skip-undistort

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check arguments
if [ $# -lt 2 ]; then
    echo -e "${RED}Usage: $0 <input_video> <output_dir> [--skip-undistort]${NC}"
    exit 1
fi

INPUT_VIDEO="$1"
OUTPUT_DIR="$2"
SKIP_UNDISTORT=false

if [ "$3" == "--skip-undistort" ]; then
    SKIP_UNDISTORT=true
fi

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
DYN_HAMR_ROOT="$HOME/dyn_hamr_workspace/Dyn-HaMR"
HAMER_3PASS_SCRIPT="$REPO_ROOT/scripts/dyn-hamr/run.py"
SKELETON_RENDER_SCRIPT="$REPO_ROOT/scripts/skeleton/render_skeleton_2d.py"
YOLO_MODEL="$DYN_HAMR_ROOT/third-party/hamer/pretrained_models/detector.pt"

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/images"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Haptica Skeleton Pipeline${NC}"
echo -e "${GREEN}========================================${NC}"
echo "Input: $INPUT_VIDEO"
echo "Output: $OUTPUT_DIR"
echo ""

# Activate conda environment
source ~/miniconda/etc/profile.d/conda.sh
conda activate dynhamr

# Step 1: Downsample to 1080p 30fps
echo -e "${GREEN}[1/6] Downsampling to 1080p 30fps...${NC}"
DOWNSAMPLED_VIDEO="$OUTPUT_DIR/downsampled_1080p30.mp4"
ffmpeg -y -i "$INPUT_VIDEO" \
    -vf "scale=1920:1080,fps=30" \
    -c:v libx264 -crf 18 -preset veryfast -pix_fmt yuv420p \
    "$DOWNSAMPLED_VIDEO" 2>/dev/null

# Step 2: Undistort GoPro video
if [ "$SKIP_UNDISTORT" = true ]; then
    echo -e "${YELLOW}[2/5] Skipping undistort (--skip-undistort)${NC}"
    UNDISTORTED_VIDEO="$DOWNSAMPLED_VIDEO"
else
    echo -e "${GREEN}[2/5] Undistorting GoPro video...${NC}"
    UNDISTORTED_VIDEO="$OUTPUT_DIR/undistorted.mp4"
    python3 "$REPO_ROOT/scripts/gopro_undistort.py" \
        --input "$DOWNSAMPLED_VIDEO" \
        --output "$UNDISTORTED_VIDEO"
fi

# Step 3: Extract frames
echo -e "${GREEN}[3/5] Extracting frames...${NC}"
ffmpeg -y -i "$UNDISTORTED_VIDEO" -qscale:v 2 "$OUTPUT_DIR/images/%06d.jpg" 2>/dev/null
FRAME_COUNT=$(ls -1 "$OUTPUT_DIR/images"/*.jpg 2>/dev/null | wc -l)
echo "  Extracted $FRAME_COUNT frames"

# Step 4: Run HaMeR 3-pass (YOLO detection + bbox cleaning + inference)
echo -e "${GREEN}[4/5] Running HaMeR 3-pass (YOLO + bbox clean + inference)...${NC}"
cd "$REPO_ROOT"
export PYTHONPATH="$DYN_HAMR_ROOT/third-party/hamer:$PYTHONPATH"
python "$HAMER_3PASS_SCRIPT" \
    --img_folder "$OUTPUT_DIR/images" \
    --out_folder "$OUTPUT_DIR/hamer_out" \
    --res_folder "$OUTPUT_DIR/hamer_out/results.pkl" \
    --checkpoint "$DYN_HAMR_ROOT" \
    --yolo_model "$YOLO_MODEL" \
    --batch_size 16 \
    --rescale_factor 1.3

# Step 5: Render skeleton overlay
echo -e "${GREEN}[5/5] Rendering skeleton overlay...${NC}"
python "$SKELETON_RENDER_SCRIPT" \
    --hamer "$OUTPUT_DIR/hamer_out/results.pkl" \
    --video "$UNDISTORTED_VIDEO" \
    --output "$OUTPUT_DIR/skeleton_overlay_raw.mp4"

# Convert to H.264 for better compatibility
echo "Converting to H.264..."
ffmpeg -y -i "$OUTPUT_DIR/skeleton_overlay_raw.mp4" \
    -c:v libx264 -crf 18 -preset fast -pix_fmt yuv420p \
    "$OUTPUT_DIR/skeleton_overlay.mp4" 2>/dev/null
rm "$OUTPUT_DIR/skeleton_overlay_raw.mp4"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Pipeline complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo "Output video: $OUTPUT_DIR/skeleton_overlay.mp4"
echo ""
echo "Intermediate files:"
echo "  - Downsampled: $DOWNSAMPLED_VIDEO"
echo "  - Undistorted: $UNDISTORTED_VIDEO"
echo "  - Frames: $OUTPUT_DIR/images/"
echo "  - HaMeR: $OUTPUT_DIR/hamer_out/"
