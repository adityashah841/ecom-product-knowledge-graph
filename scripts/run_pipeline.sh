#!/bin/bash
set -e
echo "Starting E-Commerce Product Knowledge Graph Pipeline"
python -m src.pipeline --config config/config.yaml "$@"
