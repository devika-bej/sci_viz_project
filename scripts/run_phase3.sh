#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

echo "== unit tests (debug + sanitizers) =="
cmake --build build/debug
./build/debug/run_tests

echo "== release build =="
cmake --build build/release

rm -f results/graph_stats.csv
fail=0
for f in data/*.raw; do
  stem=$(basename "$f" .raw)
  d="results/$stem"
  echo "== $stem =="
  [ -f "$d/pairs_baseline.csv" ] || ./build/release/extreem "$f" --algo baseline > /dev/null
  ./build/release/extreem "$f" --algo graph
  if diff -q "$d/pairs_baseline.csv" "$d/pairs_graph.csv" > /dev/null; then
    echo "  SAME  pairs_graph == pairs_baseline"
  else
    echo "  DIFF  pairs_graph != pairs_baseline"; fail=1
  fi
done

echo "== extremum graph sizes =="
column -s, -t results/graph_stats.csv
exit $fail