#!/bin/bash
# Script to run excc_v1.cu with different batch sizes to find optimal batch size per graph
# Usage: ./run_cc_batch_sweep.sh

OUTPUT_FILE="cc_batch_sweep_results.txt"
GRAPH_DIR="EGRs" # Give the graph directory path
SOURCE="excc_v1.cu"
EXEC="./excc"

BATCH_SIZES_MB=(16 32 64 128 256 512)

GRAPH_NAMES=(
    "kron_g500-logn21"
    "arabic-2005"
    "uk-2002"
    "kmer_A2a"
    "com-Friendster"
    "Moliere_2026"
    "Agatha-2015"
)

echo "Starting ExCC batch size sweep at $(date)" > "$OUTPUT_FILE"
echo "==========================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

if [ ! -f "$SOURCE" ]; then
    echo "ERROR: Source file $SOURCE not found!" | tee -a "$OUTPUT_FILE"
    exit 1
fi

if [ ! -d "$GRAPH_DIR" ]; then
    echo "ERROR: Graph directory $GRAPH_DIR not found!" | tee -a "$OUTPUT_FILE"
    exit 1
fi

echo "Compiling phem_cc_v1.cu..." | tee -a "$OUTPUT_FILE"
nvcc -O3 -std=c++11 -arch=sm_86 "$SOURCE" -o "$EXEC" 2>&1 | tee -a "$OUTPUT_FILE"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "ERROR: Compilation failed!" | tee -a "$OUTPUT_FILE"
    exit 1
fi
echo "Compilation successful!" | tee -a "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

for graph_name in "${GRAPH_NAMES[@]}"; do
    graph="$GRAPH_DIR/${graph_name}.egr"

    if [ ! -f "$graph" ]; then
        echo "WARNING: Graph file $graph not found, skipping..." | tee -a "$OUTPUT_FILE"
        continue
    fi

    echo "==========================================" | tee -a "$OUTPUT_FILE"
    echo "Graph: ${graph_name}.egr" | tee -a "$OUTPUT_FILE"
    echo "==========================================" | tee -a "$OUTPUT_FILE"

    for batch_mb in "${BATCH_SIZES_MB[@]}"; do
        echo "" | tee -a "$OUTPUT_FILE"
        echo ">>> Batch size: ${batch_mb} MB" | tee -a "$OUTPUT_FILE"
        echo "---" | tee -a "$OUTPUT_FILE"
        $EXEC "$graph" "$batch_mb" 2>&1 | tee -a "$OUTPUT_FILE"
        EXIT_CODE=${PIPESTATUS[0]}
        if [ $EXIT_CODE -ne 0 ]; then
            echo "WARNING: Failed with exit code $EXIT_CODE" | tee -a "$OUTPUT_FILE"
        fi
        echo "---" | tee -a "$OUTPUT_FILE"
    done
    echo "" >> "$OUTPUT_FILE"
done

echo "==========================================" | tee -a "$OUTPUT_FILE"
echo "Completed at $(date)" | tee -a "$OUTPUT_FILE"
echo "==========================================" | tee -a "$OUTPUT_FILE"

rm -f "$EXEC"
echo "Cleanup completed. Output saved to: $OUTPUT_FILE"
