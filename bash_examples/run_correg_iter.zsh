ROOT="/path/to/project/data"
SUBJECTS_DIR="MRI/freesurfer"

while read -r subj ses; do
    echo "Processing $subj session $ses"

    python corregistration.py \
        --root_dir "$ROOT" \
        --subject_id "$subj" \
        --session "$ses" \
        --subjects_dir "$SUBJECTS_DIR"

done < /path/to/subjects_sessions.txt