#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
cd "$SCRIPT_DIR" || exit 1
cd ../../../backend || exit 1

run_mypy_check() {
	echo "Running mypy check..."
	git ls-files "*.py" | xargs mypy
}

run_pylint_check() {
	echo "Running pylint check..."
	git ls-files "*.py" | xargs pylint
}

# Run all checks and capture their exit codes
mypy_failed=0
pylint_failed=0

run_mypy_check || mypy_failed=1
run_pylint_check || pylint_failed=1

# Exit with failure if any check failed
[ $mypy_failed -eq 0 ] && [ $pylint_failed -eq 0 ] || exit 1
