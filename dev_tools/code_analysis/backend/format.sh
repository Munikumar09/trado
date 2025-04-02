#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
cd "$SCRIPT_DIR" || exit 1
cd ../../../backend || exit 1


format_files() {
	git ls-files "*.py" | xargs black "$@"
	git ls-files "*.py" | xargs isort --profile black "$@"
}
format_bash_files() {
	git ls-files "*.sh" | xargs shfmt -w "$@"
}

check_format() {
	format_files --check
	format_bash_files -d

	cd $SCRIPT_DIR
	./yaml_format --check
}

reformat() {
	format_files
	format_bash_files

	cd $SCRIPT_DIR
	./yaml_format --all
}

if [ "${1:-}" == "--check" ]; then
	check_format
else
	reformat
fi
