#!/bin/bash

# SBATCH -A $USER
# SBATCH -n 9
# SBATCH --gres=gpu:1
# SBATCH --mem-per-cpu=2G
# SBATCH --time=4-00:00:00
# SBATCH --output=out.txt
# SBATCH --mail-user=vansh.garg@research.iiit.ac.in
# SBATCH --mail-type=ALL

# Usage: ./batch.sh /path/to/directory
pwd

file_path="$1"
# Check if the path is valid
if [ -f "$file_path" ]; then
    echo "Path is valid: $file_path"
else
    echo "Invalid path: $file_path"
    exit 1
fi

python wandb-4.3.py --path "$file_path"