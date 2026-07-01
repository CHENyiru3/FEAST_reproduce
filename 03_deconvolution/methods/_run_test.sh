#!/bin/bash
source /maiziezhou_lab2/yiru/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
export PATH="/maiziezhou_lab2/yiru/envs/cell2loc_env/bin:$PATH"
exec /maiziezhou_lab2/yiru/envs/cell2loc_env/bin/python /maiziezhou_lab2/yiru/FEAST_experiments/05_deconvolution/methods/_test_regression.py 2>&1
echo "EXIT=$?"
