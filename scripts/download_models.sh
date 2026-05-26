#!/bin/bash
set -e

echo "Downloading models from W&B..."
python - <<'EOF'
import wandb
import os
import shutil

os.environ["WANDB_SILENT"] = "true"

api = wandb.Api()
entity = "nathan2massicot-berner-fachhochschule"
project = "brain-tumor-classification"

os.makedirs("/app/models", exist_ok=True)

models = [
    "model-resnet50_transfer",
    "model-unet_classifier",
    "model-simple_cnn",
    "model-baseline",
]

for model_name in models:
    try:
        artifact = api.artifact(f"{entity}/{project}/{model_name}:latest")
        download_dir = artifact.download()
        # Dateien nach /app/models/ verschieben
        for f in os.listdir(download_dir):
            if f.endswith(".pt"):
                shutil.copy(os.path.join(download_dir, f), f"/app/models/{f}")
                print(f"✓ {f}")
    except Exception as e:
        print(f"✗ {model_name}: {e}")

print("Download complete!")
EOF

exec uvicorn brain_tumor_mlops.api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1