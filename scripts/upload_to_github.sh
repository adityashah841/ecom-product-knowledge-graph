#!/bin/bash
set -e
jupyter nbconvert --clear-output notebooks/*.ipynb
git add -A
git commit -m "Final: complete KG pipeline with results"
git push origin main
echo "Successfully pushed to GitHub"
