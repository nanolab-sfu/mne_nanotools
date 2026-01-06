#!/bin/zsh

ROOT_DIR="/Users/isaant/Documents/PosDoc/Projects/project/data"
TSSS_DIR="/Users/isaant/Documents/PosDoc/Projects/tsss_params/2023"
SUBJECT_ID="sub_PROJXXX"

SESSIONS=("251016" "251017")
RESTS=("rest1" "rest2")

for SESSION in "${SESSIONS[@]}"; do
    for REST in "${RESTS[@]}"; do
        echo ">>> Processing session $SESSION, resting state $REST"

        python generic_taskfree_MEGIN.py \
            --root_dir "$ROOT_DIR" \
            --subject_id "$SUBJECT_ID" \
            --session "$SESSION" \
            --resting "$REST" \
            --tsss_dir "$TSSS_DIR" \
            --l_freq 0.5 \
            --h_freq 200 \
            --line_freqs 60 120 180 \
            --downsample 500 \
            --st_duration 10.0 \
            --inv_method beamformer
        break
    done
    break
done