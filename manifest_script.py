import os
import glob
import pandas as pd

# 1. Point this to the folder you just saved everything into
data_dir = "./processed_data"

# Find all the point cloud files
ptcloud_files = sorted(glob.glob(os.path.join(data_dir, "*_ptcloud.npy")))

# 2. Generate the matching SDF file paths
# We safely replace the suffix so they match up perfectly row-by-row
sdf_files = [f.replace("_ptcloud.npy", "_sdf.npy") for f in ptcloud_files]

# Optional: Quick check to make sure no files are missing
for sdf_path in sdf_files:
    if not os.path.exists(sdf_path):
        print(f"Warning: Missing SDF file for {sdf_path}")

# 3. Create the Manifest DataFrame
df = pd.DataFrame({
    "point_cloud_path": ptcloud_files,
    "sdf_path": sdf_files
})

# 4. Save it to CSV
manifest_path = "./neutrophil_sdf_manifest.csv"
df.to_csv(manifest_path, index=False)

print(f"Manifest created successfully with {len(df)} cells paired!")
print(df.head())
