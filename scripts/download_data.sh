#!/bin/bash
set -e
echo "Downloading and preprocessing data..."
python -m src.data.downloader --config config/config.yaml
echo "Data download and preprocessing complete."
