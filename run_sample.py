from cyto_dl.api import CytoDLModel

# Trying to override the progress bar
import os
# Force PyTorch Lightning to drop the 'rich' animated UI completely
os.environ["RICH_TRACKING_PROGRESS"] = "0"
os.environ["PROGRESS_BAR"] = "0"
os.environ["HYDRA_FULL_ERROR"] = "1"

# 1. Initialize the model wrapper
model = CytoDLModel()

# 2. Load the default image-to-image segmentation configuration
# We add 'trainer=cpu' to ensure it runs cleanly on your Intel i5 processor
model.load_default_experiment(
    "segmentation",
    output_dir="./output",
    overrides=["trainer=cpu"
               ]
)

# 3. Start the training loop on the sample dataset!
print("Starting sample training loop...")
model.train()
