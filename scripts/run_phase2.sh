#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

echo "== unit tests (debug + sanitizers) =="
cmake --build build/debug
./build/debug/run_tests

echo "== release build =="
cmake --build build/release

for f in data/*.raw; do
  echo "== $f =="
  ./build/release/extreem "$f" --algo baseline > /dev/null
  ./build/release/extreem "$f" --algo cp
done

echo "== compare against baseline =="
source .venv/bin/activate
for d in results/*/; do
  python python/compare_saddles.py "$d"
done