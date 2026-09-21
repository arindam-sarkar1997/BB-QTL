#!/bin/bash
set -euo pipefail

analysis_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$analysis_dir"
exec bash 0_run_analyses_slurm.sh "$@"
