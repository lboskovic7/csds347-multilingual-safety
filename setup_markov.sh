#!/bin/bash
# One-time setup on Markov. Run this from a login shell (not inside a job).
# Usage: bash setup_markov.sh
set -e

COURSE=csds447
TEAM=team5
PARTNER=jxy1203   # teammate who gets full access to the shared folder
BASE=/mnt/vstor/courses/$COURSE/$TEAM

# Shared team folder in course space (home is only 25 GB; each 7-9B model is ~15-18 GB)
mkdir -p -m 700 "$BASE"

# Give the teammate access to everything here now and to anything created later
# Same ACL as the course guide (section 6), applied recursively so existing files are shared too
if setfacl -R -m u:$PARTNER:rwX,d:u:$PARTNER:rwx,d:u:$USER:rwx "$BASE"; then
    echo "Shared $BASE with $PARTNER"
else
    echo "WARNING: could not share folder with $PARTNER (setfacl failed); continuing"
fi

# Check Python version (needs 3.9+)
python3 -c 'import sys; assert sys.version_info >= (3, 9), sys.version' \
    || { echo "python3 is too old. Run: module spider Python, then module load <version>, and rerun."; exit 1; }

# Python virtual environment in course space
python3 -m venv "$BASE/venv"
source "$BASE/venv/bin/activate"
pip install --upgrade pip
pip install torch transformers accelerate datasets pandas

# Send all Hugging Face downloads to course space, not your home directory
mkdir -p "$BASE/hf_cache" "$BASE/outputs"
grep -q "HF_HOME=$BASE/hf_cache" ~/.bashrc || echo "export HF_HOME=$BASE/hf_cache" >> ~/.bashrc

echo
echo "Done"
echo "  source $BASE/venv/bin/activate && export HF_HOME=$BASE/hf_cache && hf auth login"

