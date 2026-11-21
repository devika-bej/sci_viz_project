#!/usr/bin/env bash
# Full benchmark: correctness + memory, extremum-graph sizes, thread scaling.
# Usage:  scripts/benchmark.sh [data files...]        (default: all of data/*.raw)
# Env:    REPEATS=3   THREADS="1 2 4 8 12 24"   CLEAN=1 (0 keeps old results/)
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=./build/release/extreem
REPEATS=${REPEATS:-3}
MAXT=$(nproc)
if [ -z "${THREADS:-}" ]; then              # default: 1 2 4 8 ... up to nproc
  THREADS=""; t=1
  while [ "$t" -lt "$MAXT" ]; do THREADS="$THREADS $t"; t=$((t * 2)); done
  THREADS="$THREADS $MAXT"
fi
if [ $# -gt 0 ]; then DATA=("$@"); else DATA=(data/*.raw); fi

export OMP_PROC_BIND=close OMP_PLACES=cores  # stable thread placement for scaling curves

if [ "${CLEAN:-1}" = 1 ]; then rm -rf results; fi
mkdir -p results

echo "== build =="
cmake --build build/release

echo "== metadata =="
{
  echo "date: $(date)"
  echo "commit: $(git rev-parse HEAD 2>/dev/null || echo n/a)"
  echo "compiler: $(g++ --version | head -1)"
  echo "threads: $THREADS   repeats: $REPEATS"
  echo "datasets: ${DATA[*]}"
  echo; lscpu; echo; free -g
} > results/meta.txt

# ---------------------------------------------------------------------------
echo "== [1/3] correctness + peak memory =="
MEM=results/memory.csv
echo "dataset,algo,threads,max_rss_kb" > "$MEM"

run_mem() {   # stem algo threads file
  local stem=$1 algo=$2 t=$3 file=$4 log
  log=$(mktemp)
  OMP_NUM_THREADS=$t /usr/bin/time -v "$BIN" "$file" --algo "$algo" > /dev/null 2> "$log"
  echo "$stem,$algo,$t,$(grep 'Maximum resident set size' "$log" | awk '{print $NF}')" >> "$MEM"
  rm -f "$log"
}

fail=0
for f in "${DATA[@]}"; do
  stem=$(basename "$f" .raw); d="results/$stem"
  echo "-- $stem"
  run_mem "$stem" baseline 1 "$f"
  run_mem "$stem" egp "$MAXT" "$f"
  if diff -q "$d/pairs_baseline.csv" "$d/pairs_egp.csv" > /dev/null; then
    echo "   SAME  ExTreeM pairs == baseline pairs"
  else
    echo "   DIFF  ExTreeM pairs != baseline pairs"; fail=1
  fi
done
[ "$fail" = 0 ] || { echo "correctness failed, stopping"; exit 1; }
mv results/timings.csv results/timings_correctness.csv   # keep scaling timings clean

# ---------------------------------------------------------------------------
echo "== [2/3] extremum graph sizes =="
for f in "${DATA[@]}"; do
  OMP_NUM_THREADS=$MAXT "$BIN" "$f" --algo graph > /dev/null
done
mv results/timings.csv results/timings_graph.csv          # contains graph_uf (serial sweep on G)

# ---------------------------------------------------------------------------
echo "== [3/3] thread scaling (repeats=$REPEATS, threads:$THREADS) =="
for f in "${DATA[@]}"; do
  stem=$(basename "$f" .raw)
  for r in $(seq 1 "$REPEATS"); do
    echo "-- $stem  repeat $r/$REPEATS"
    OMP_NUM_THREADS=1 "$BIN" "$f" --algo baseline > /dev/null
    for t in $THREADS; do
      OMP_NUM_THREADS=$t "$BIN" "$f" --algo egp > /dev/null
    done
  done
done

echo
echo "done. outputs:"
echo "  results/timings.csv              scaling runs (dataset,algo,threads,step,seconds)"
echo "  results/timings_correctness.csv  timings from the correctness runs"
echo "  results/timings_graph.csv        includes graph_uf (serial sweep on G)"
echo "  results/graph_stats.csv          |V(G)|, |E(G)|, ratio"
echo "  results/memory.csv               peak memory per dataset/algorithm"
echo "  results/meta.txt                 machine + settings"
echo "  results/<dataset>/pairs_*.csv    persistence pairs"