from cyto_dl.api import CytoDLModel

model = CytoDLModel()

# 1. Use the broad category name expected by the API
model.load_default_experiment(
    "representation_learning",
    output_dir="./vae_output",
    overrides=[
        # 2. Force the specific SDF template here inside the overrides!
        "experiment=representation_learning/pc_sdf",
        "trainer=cpu",
        "data.data_path=./neutrophil_sdf_manifest.csv",
        "data.point_cloud_column=point_cloud_path",
        "data.sdf_column=sdf_path",
        "model.network.encoder.num_points=2048"
    ]
)

print("Starting Autoencoder Training...")
model.train()
