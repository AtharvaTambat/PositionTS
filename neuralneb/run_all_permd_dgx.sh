#!/bin/bash
echo "hello"

PRODUCT_DIR="../permuted_data/isomer_splits/xyz_data/product"
REACTANT_DIR="../permuted_data/isomer_splits/xyz_data/reactant"
OUTPUT_DIR="./generated_transition_states/isomer_permd"

LOG_FILE="progress_isomer_permd.log"
MAX_DGX_JOBS=4

mkdir -p $OUTPUT_DIR
touch $LOG_FILE

TOTAL=$(ls $REACTANT_DIR/entry_*.xyz | wc -l)
DONE=0
START_TIME=$(date +%s)

get_dgx_jobs() {
    squeue -u $USER | grep " dgx " | grep -E "R|PD" | wc -l
}

for file in $REACTANT_DIR/entry_*.xyz; do
    fname=$(basename $file)

    product_file="$PRODUCT_DIR/$fname"
    reactant_file="$REACTANT_DIR/$fname"
    output_file="$OUTPUT_DIR/$fname"

    # ✅ skip completed
    if [ -f "$output_file" ]; then
        ((DONE++))
        continue
    fi

    # 🔥 wait for slot
    while true; do
        current_dgx=$(get_dgx_jobs)

        if [ "$current_dgx" -lt "$MAX_DGX_JOBS" ]; then
            break
        fi

        sleep 20
    done

    echo "Submitting $fname" >> $LOG_FILE

    # ❌ suppress SLURM .out files
    sbatch \
    --output=/dev/null \
    --error=/dev/null \
    --open-mode=append \
    script_dgx.slurm \
    oa2 scripts/neb_new_2.py \
    $product_file \
    $reactant_file \
    --output_dir $OUTPUT_DIR

    ((DONE++))

    # -------------------------
    # ETA CALCULATION
    # -------------------------
    NOW=$(date +%s)
    ELAPSED=$((NOW - START_TIME))

    if [ $DONE -gt 0 ]; then
        AVG=$((ELAPSED / DONE))
        REMAINING=$((TOTAL - DONE))
        ETA=$((AVG * REMAINING))
    else
        ETA=0
    fi

    printf "[%d/%d] Submitted %s | ETA: %02dh:%02dm:%02ds\n" \
        $DONE $TOTAL $fname \
        $((ETA/3600)) $((ETA%3600/60)) $((ETA%60)) >> $LOG_FILE

    sleep 2
done

echo "All jobs submitted!" >> $LOG_FILE