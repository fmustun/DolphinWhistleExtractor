#!/bin/bash
set -e

MODEL=models/run3/model_vgg_best.h5
SCRIPT=Predict_and_extract/main.py

# ── Run 1: 2022 recordings ─────────────────────────────────────────────────
TMPFILE=$(mktemp)
cat > "$TMPFILE" <<EOF
Exp_30_Oct_2022_1145_channel_0.wav
Exp_29_Apr_2022_1545_channel_1.wav
EOF

python "$SCRIPT" \
    --model_path "$MODEL" \
    --recordings /media/DOLPHIN2/Recordings/2022 \
    --saving_folder test/new_model/ \
    --specific_files "$TMPFILE" \
    --save_p \
    --batch_size 16 \
    --max_workers 2 \
    --cpu

rm -f "$TMPFILE"

# ── Run 2: 2024 recordings (selected folders) ──────────────────────────────
TMPFILE=$(mktemp)
cat > "$TMPFILE" <<EOF
Exp_01_Aug_2024_0800_channel_1.wav
Exp_01_Aug_2024_1200_channel_1.wav
EOF

python "$SCRIPT" \
    --model_path "$MODEL" \
    --recordings /media/DOLPHIN1/2024 \
    --saving_folder test/new_model/ \
    --specific_files "$TMPFILE" \
    --save_p \
    --batch_size 16 \
    --max_workers 2 \
    --cpu

rm -f "$TMPFILE"
