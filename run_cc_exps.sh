#!/bin/bash
# Usage: ./run_cc_exps.sh  (run from project root)
# Output is saved to results_cc.txt

OUTPUT_FILE="results_cc.txt"
GRAPH_DIR="EGRs" # Give the graph directory path

# Source files (paths relative to project root)
SOURCE="excc_v1.cu"

# Compilation settings
ARCH="sm_86"
CXX_STD="c++17"
COMMON_COMPILE_FLAGS="-O3 -std=$CXX_STD -arch=$ARCH"
ECL_COMPILE_FLAGS="-O3 -arch=$ARCH"

# Executable names
EXEC="./excc"

GRAPH_NAMES=(
    "kron_g500-logn21"
    "arabic-2005"
    "uk-2002"
    "kmer_A2a"
    "com-Friendster"
    "Moliere_2026"
    "Agatha-2015"
)

# Clear output file and start logging
echo "Starting CC test run for ExCC at $(date)" > "$OUTPUT_FILE"
echo "==========================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Check if graph directory exists
if [ ! -d "$GRAPH_DIR" ]; then
    echo "ERROR: Graph directory $GRAPH_DIR not found!" | tee -a "$OUTPUT_FILE"
    exit 1
fi

# Compile
echo "Compiling..." | tee -a "$OUTPUT_FILE"
echo "==========================================" | tee -a "$OUTPUT_FILE"

echo ">>> Compiling ExCC..." | tee -a "$OUTPUT_FILE"
nvcc $COMMON_COMPILE_FLAGS "$SOURCE" -o "$EXEC" 2>&1 | tee -a "$OUTPUT_FILE"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "ERROR: Compilation of ExCC failed!" | tee -a "$OUTPUT_FILE"
    exit 1
fi
echo "ExCC compilation successful!" | tee -a "$OUTPUT_FILE"

# Process each graph
COUNT=0
for graph_name in "${GRAPH_NAMES[@]}"; do
    graph="$GRAPH_DIR/${graph_name}.egr"

    if [ ! -f "$graph" ]; then
        echo "WARNING: Graph file $graph not found, skipping..." | tee -a "$OUTPUT_FILE"
        continue
    fi

    COUNT=$((COUNT + 1))
    echo "==========================================" | tee -a "$OUTPUT_FILE"
    echo "[$COUNT/${#GRAPH_NAMES[@]}] Processing: ${graph_name}.egr" | tee -a "$OUTPUT_FILE"
    echo "==========================================" | tee -a "$OUTPUT_FILE"
    echo "" | tee -a "$OUTPUT_FILE"

    # Run ExCC
    echo ">>> Running ExCC: $EXEC $graph" | tee -a "$OUTPUT_FILE"
    echo "---" | tee -a "$OUTPUT_FILE"
    $EXEC "$graph" 2>&1 | tee -a "$OUTPUT_FILE"
    EXIT_CODE_ExCC=${PIPESTATUS[0]}
    echo "---" | tee -a "$OUTPUT_FILE"
    if [ $EXIT_CODE_ExCC -ne 0 ]; then
        echo "WARNING: ExCC failed with exit code $EXIT_CODE_ExCC" | tee -a "$OUTPUT_FILE"
    fi
    echo "" | tee -a "$OUTPUT_FILE"
done

echo "==========================================" | tee -a "$OUTPUT_FILE"
echo "Completed processing $COUNT graph(s)" | tee -a "$OUTPUT_FILE"
echo "Test run finished at $(date)" | tee -a "$OUTPUT_FILE"
echo "==========================================" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

# Cleanup compiled executables
echo "Cleaning up compiled executables..." | tee -a "$OUTPUT_FILE"
rm -f "$EXEC"
rmdir ecl 2>/dev/null
echo "Cleanup completed." | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"
echo "All output saved to: $OUTPUT_FILE"
