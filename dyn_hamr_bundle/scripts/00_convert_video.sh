#!/bin/bash
# Convert webm to mp4 for Dyn-HaMR
# Run this first if your video is .webm format

if [ -z "$1" ]; then
    echo "Usage: bash 00_convert_video.sh input.webm"
    exit 1
fi

INPUT="$1"
OUTPUT="${INPUT%.webm}.mp4"

echo "Converting $INPUT to $OUTPUT..."
ffmpeg -i "$INPUT" -c:v libx264 -preset fast -crf 23 "$OUTPUT"
echo "Done: $OUTPUT"
