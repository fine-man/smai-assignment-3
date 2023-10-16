#!/bin/bash
# Usage: ./batch.sh /path/to/directory

file_path="$1"
# Check if the path is valid
if [ -f "$file_path" ]; then
    echo "Path is valid: $file_path"
else
    echo "Invalid path: $file_path"
    exit 1
fi

# SBATCH -A $USER
# SBATCH -n 10
# SBATCH --mem-per-cpu=4G
# SBATCH --time=4-00:00:00
# SBATCH --output=darknet_file.txt
# SBATCH --mail-user=vansh.garg@research.iiit.ac.in
# SBATCH --mail-type=ALL

conda activate lip-reading

# python wandb-2.2.py --path $file_path