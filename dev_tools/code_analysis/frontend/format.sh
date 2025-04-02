#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
cd "$SCRIPT_DIR" || exit 1
cd ../../../frontend || exit 1

# Format Dart files
format_files() {
  dart format .
}

# Check Dart format without modifying files
check_format() {
  echo "Checking Dart format..."
  
  # Capture the output of dart format
  FORMAT_OUTPUT=$(dart format --output=none .)
  
  # Check if the output contains the text "(0 changed)", which indicates no changes
  if echo "$FORMAT_OUTPUT" | grep -q "(0 changed)"; then
    echo "✅ Formatting is correct."
  else
    echo "❌ Formatting issues found in the following files:"
    echo "$FORMAT_OUTPUT"
    exit 1
  fi
}

# Run Dart Analyze
run_analyze() {
  echo "Running dart analyze..."
  dart analyze
}

# Check for analyze issues
check_analyze() {
  ANALYZE_OUTPUT=$(dart analyze .)
  
  # Capture the last line of the output
  LAST_LINE=$(echo "$ANALYZE_OUTPUT" | tail -n 1)
  
  # Check if the last line contains "issue found"
  if echo "$LAST_LINE" | grep -q "issue found"; then
    echo "❌ Analysis issues found:"
    echo "$ANALYZE_OUTPUT"
    exit 1
  else
    echo "✅ No issues found in analysis."
  fi
}

# Reformat all Dart files
reformat() {
  echo "Reformatting Dart files..."
  format_files
}

# Main script logic
if [ "${1:-}" == "--check" ]; then
  check_format
  check_analyze
else
  reformat
  run_analyze
fi
