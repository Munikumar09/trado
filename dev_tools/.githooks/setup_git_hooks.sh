#!/bin/bash

set -euo pipefail  # Exit on error, undefined variable, or pipeline failure

# Get the directory of the script
SCRIPT_DIR=$(dirname "$(realpath "$0")")
GIT_HOOKS_DIR="$SCRIPT_DIR/../../.git/hooks"

echo "🔍 Determining Git hooks directory..."
echo "📂 Git Hooks Directory: $GIT_HOOKS_DIR"
echo "📂 Script Directory: $SCRIPT_DIR"

# Ensure the Git hooks directory exists
if [ ! -d "$GIT_HOOKS_DIR" ]; then
    echo "❌ Error: Git hooks directory does not exist at $GIT_HOOKS_DIR"
    exit 1
fi

# Get list of files in hooks directory before copying
PRE_EXISTING_FILES=$(ls "$GIT_HOOKS_DIR")

# Copy all hook scripts to the Git hooks directory
echo "🚀 Copying hook scripts..."
if ! cp -r "$SCRIPT_DIR"/* "$GIT_HOOKS_DIR"; then
    echo "❌ Error: Failed to copy hook scripts."
    exit 1
fi

# Identify newly copied files (files that were NOT present before)
declare -a NEW_FILES=()
for file in "$GIT_HOOKS_DIR"/*; do
    filename=$(basename "$file")
    if [[ ! " $PRE_EXISTING_FILES " =~ " $filename " ]]; then
        NEW_FILES+=("$file")
    fi
done

# Make only newly copied files executable
if [[ ${#NEW_FILES[@]} -gt 0 ]]; then
    echo "🔑 Setting executable permissions for new hook scripts..."
    chmod +x "${NEW_FILES[@]}"
else
    echo "✅ No new files required permission changes."
fi

echo "✅ Git hooks have been set up successfully!"
exit 0
