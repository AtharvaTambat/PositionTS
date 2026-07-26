#!/bin/bash
# Submit one NEB job per reaction for a dataset, throttling concurrent jobs.
#
#   ./run_all.sh <dataset> [--permuted]
#
# <dataset> is one of transition1x, isomer, rgd1, rgdx, rgdy. With --permuted the
# atom-order-permuted copies of the test split are used instead.
#
# Environment:
#   DATA_ROOT      root holding <dataset>/xyz_files/test_split/     (default ../ts_gen/data)
#   PERMUTED_ROOT  root holding <dataset>_splits/xyz_data/          (default ../../permuted_data)
#   OUTPUT_ROOT    where generated transition states are written    (default ./generated_transition_states)
#   CONDA_ENV      conda environment passed to run.slurm            (default oa2)
#   PARTITION      queue to submit to and to count when throttling  (default: cluster default queue)
#   MAX_JOBS       maximum concurrent jobs                          (default 4)
set -u

DATASET="${1:-}"
PERMUTED="${2:-}"
usage() { echo "Usage: $0 <transition1x|isomer|rgd1|rgdx|rgdy> [--permuted]"; exit 1; }
case "$DATASET" in
    transition1x|isomer|rgd1|rgdx|rgdy) ;;
    *) usage ;;
esac
case "$PERMUTED" in
    ""|--permuted) ;;
    *) echo "ERROR: unrecognised second argument '$PERMUTED'"; usage ;;
esac

DATA_ROOT="${DATA_ROOT:-../ts_gen/data}"
PERMUTED_ROOT="${PERMUTED_ROOT:-../../permuted_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./generated_transition_states}"
CONDA_ENV="${CONDA_ENV:-oa2}"
USER="${USER:-$(id -un)}"
MAX_JOBS="${MAX_JOBS:-4}"
PARTITION="${PARTITION:-}"
WALLTIME="${WALLTIME:-48:00:00}"   # the dgx submitters used 120:00:00
# The originals submitted the non-logging wrapper with output discarded; keep that
# default so a 1000-reaction sweep does not leave 1000 log files behind.
export LOG_TO_FILE="${LOG_TO_FILE:-0}"

if [ "$PERMUTED" = "--permuted" ]; then
    PRODUCT_DIR="$PERMUTED_ROOT/${DATASET}_splits/xyz_data/product"
    REACTANT_DIR="$PERMUTED_ROOT/${DATASET}_splits/xyz_data/reactant"
    OUTPUT_DIR="$OUTPUT_ROOT/${DATASET}_permd"
    SUFFIX="_permd"
else
    PRODUCT_DIR="$DATA_ROOT/$DATASET/xyz_files/test_split/products"
    REACTANT_DIR="$DATA_ROOT/$DATASET/xyz_files/test_split/reactants"
    OUTPUT_DIR="$OUTPUT_ROOT/$DATASET"
    SUFFIX=""
fi

LOG_FILE="progress_${DATASET}${SUFFIX}.log"
mkdir -p "$OUTPUT_DIR"
touch "$LOG_FILE"

TOTAL=$(ls "$REACTANT_DIR"/entry_*.xyz | wc -l)
DONE=0
START_TIME=$(date +%s)

count_queued_jobs() {
    if [ -n "$PARTITION" ]; then
        squeue -u "$USER" -h -p "$PARTITION" -t R,PD | wc -l
    else
        squeue -u "$USER" -h -t R,PD | wc -l
    fi
}

for file in "$REACTANT_DIR"/entry_*.xyz; do
    fname=$(basename "$file")
    output_file="$OUTPUT_DIR/$fname"

    # skip reactions already generated
    if [ -f "$output_file" ]; then
        ((DONE++))
        continue
    fi

    # wait for a free slot
    while [ "$(count_queued_jobs)" -ge "$MAX_JOBS" ]; do
        sleep 20
    done

    echo "Submitting $fname" >> "$LOG_FILE"

    # SLURM's own .out/.err files are suppressed; run.slurm writes its own log
    sbatch \
        --output=/dev/null \
        --error=/dev/null \
        --open-mode=append \
        -t "$WALLTIME" \
        ${PARTITION:+--partition="$PARTITION" --qos="$PARTITION"} \
        ./run.slurm \
        "$CONDA_ENV" scripts/neb_new_2.py \
        "$PRODUCT_DIR/$fname" \
        "$REACTANT_DIR/$fname" \
        --output_dir "$OUTPUT_DIR"

    ((DONE++))

    NOW=$(date +%s)
    ELAPSED=$((NOW - START_TIME))
    if [ $DONE -gt 0 ]; then
        AVG=$((ELAPSED / DONE))
        ETA=$((AVG * (TOTAL - DONE)))
    else
        ETA=0
    fi
    printf "[%d/%d] Submitted %s | ETA: %02dh:%02dm:%02ds\n" \
        $DONE $TOTAL "$fname" \
        $((ETA/3600)) $((ETA%3600/60)) $((ETA%60)) >> "$LOG_FILE"

    sleep 2
done

echo "All jobs submitted!" >> "$LOG_FILE"
