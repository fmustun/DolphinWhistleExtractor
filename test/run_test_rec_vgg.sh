#!/bin/bash
set -e

MODEL=models/run3/model_vgg_best.pt
SCRIPT=Predict_and_extract/main.py
OUTDIR=test/new_model_torch

rm -rf "$OUTDIR/Exp_29_Apr_2022_1545_channel_1"
rm -rf "$OUTDIR/Exp_30_Oct_2022_1145_channel_0"
rm -rf "$OUTDIR/Exp_01_Aug_2024_0800_channel_1"
rm -rf "$OUTDIR/Exp_01_Aug_2024_1200_channel_1"

run_subset() {
    local recordings="$1"
    local tmpfile
    tmpfile=$(mktemp)
    cat > "$tmpfile"

    python "$SCRIPT" \
        --model_path "$MODEL" \
        --recordings "$recordings" \
        --saving_folder "$OUTDIR"/ \
        --specific_files "$tmpfile" \
        --save_p \
        --batch_size 16 \
        --max_workers 2 \
        --cpu

    rm -f "$tmpfile"
}

# ── Run 1: 2022 recordings ─────────────────────────────────────────────────
run_subset /media/DOLPHIN2_robin/Recordings/2022 <<EOF
Exp_30_Oct_2022_1145_channel_0.wav
Exp_29_Apr_2022_1545_channel_1.wav
EOF

# ── Run 2: 2024 recordings (selected folders) ──────────────────────────────
run_subset /media/DOLPHIN1_robin/2024 <<EOF
Exp_01_Aug_2024_0800_channel_1.wav
Exp_01_Aug_2024_1200_channel_1.wav
EOF
