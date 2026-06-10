import os
from pathlib import Path
import napari
import numpy as np
from skimage import morphology, io, measure
import warnings
import point_cloud_utils as pcu

# Old hole filling function


def fill(x):
    newfile = np.zeros_like(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        for i in range(len(x)):
            roi = x[i, :, :]
            new = morphology.remove_small_holes(roi, area_threshold=1000)
            newfile[i, :, :] = new
        newfile = morphology.remove_small_holes(newfile, area_threshold=100000)
        newfile = morphology.remove_small_objects(newfile, min_size=5000)
    return newfile


viewer = napari.Viewer()

directory_path = Path(
    '/Users/coding/Downloads/_nuclear_morpho_data/fresh_data_june')
# Filter out non-TIFF system files like .DS_Store
all_files = sorted([str(f.path) for f in os.scandir(
    directory_path) if f.is_file() and f.name.endswith(('.tif', '.tiff'))])

NUM_POINTS = 2048  # Fixed target token size required by point cloud autoencoders
point_clouds = {}
sdfs = {}


for file_path in all_files:
    ogfile = io.imread(file_path)
    ogfile = (ogfile > 0).astype(np.uint8)  # Ensure binary layout

    # Extract the two-digit file identifier (e.g., '033')
    file_stem = Path(file_path).stem

    print(f"Processing {file_stem}")

    """ if file_stem != "051":
        print(f"skipped {file_stem}")
        continue """

    # 2. Clear internal gaps/holes
    mask_filled = fill(ogfile)

    mask_filled = np.pad(mask_filled, pad_width=1,
                         mode='constant', constant_values=0)

    # viewer.add_image(mask_filled, name=f"{file_stem}", visible=False)

    # Blue noise sampling with fixed number of points with Lloyd relaxation
    # algorithm from marching cubes dervied mesh

    verts, faces, normals, values = measure.marching_cubes(
        mask_filled,
        # spacing=[6, 1, 1],
        step_size=1

    )
    surface_data = (verts, faces)
    viewer.add_surface(surface_data, name=f"mesh {file_stem}", visible=False)

    faces = faces.astype(np.int32)

    points_lloyd = pcu.sample_mesh_lloyd(verts, faces, NUM_POINTS)

    RESOLUTION = 10_000
    verts_w, faces_w = pcu.make_mesh_watertight(verts, faces, RESOLUTION)

    # 2.5. Compute SDF to mesh

    # 1. Find the exact coordinates of the solid nucleus
    z_idx, y_idx, x_idx = np.where(mask_filled > 0)

    # 2. Define the bounding box with a 5-pixel padding buffer
    padding = 5
    x_min, x_max = x_idx.min() - padding, x_idx.max() + padding
    y_min, y_max = y_idx.min() - padding, y_idx.max() + padding
    z_min, z_max = z_idx.min() - padding, z_idx.max() + padding

    # 3. Generate exactly 4000 random points within this tight bounding box
    num_points = 4000

    # np.random.uniform cleanly handles the min/max scaling for all 3 axes at once
    points = np.random.uniform(
        low=[x_min, y_min, z_min],
        high=[x_max, y_max, z_max],
        size=(num_points, 3)
    )

    points = points.astype(np.float32)
    verts_w = verts_w.astype(np.float32)
    faces_w = faces_w.astype(np.int32)

    sdf, fid, bc = pcu.signed_distance_to_mesh(points, verts_w, faces_w)

    sdf_array = np.column_stack((points, sdf))

    sdfs[file_stem] = sdf_array

    # 3. Extract 3D boundary layer surface voxels
    edges = np.zeros_like(mask_filled)

    for axis in range(3):
        for s in [-1, 1]:
            diff = mask_filled != np.roll(mask_filled, shift=s, axis=axis)
            edges |= diff

    edges &= mask_filled

    z_indices, y_indices, x_indices = np.where(edges > 0)

    if len(z_indices) == 0:
        print(f"Skipping {file_stem} - No structural surfaces found.")
        continue

    raw_points_zyx = np.column_stack((z_indices, y_indices, x_indices))

    # print(len(raw_points_zyx))

    # 4. Uniform sampling using native NumPy (Guarantees exactly 2048 points)
    num_surface_points = len(raw_points_zyx)

    if num_surface_points >= NUM_POINTS:
        # Randomly select 2048 unique indices without duplication
        selected_indices = np.random.choice(
            num_surface_points, size=NUM_POINTS, replace=False)
    else:
        # If the cell is tiny and has fewer than 2048 points, allow duplication (upsampling)
        selected_indices = np.random.choice(
            num_surface_points, size=NUM_POINTS, replace=True)

    resampled_points_zyx = raw_points_zyx[selected_indices]

    # 5. Flip to standard (X, Y, Z) geometry
    autoencoder_ready_xyz = np.column_stack(
        (resampled_points_zyx[:, 2], resampled_points_zyx[:, 1], resampled_points_zyx[:, 0]))

    # 6. Store array in active local memory using the cell ID string as the key
    point_clouds[file_stem] = autoencoder_ready_xyz

    # 7. Feed the geometry to Napari for visual inspection
    # viewer.add_labels(mask_filled, name=f"Mask_{file_stem}", visible=False)
    viewer.add_points(
        # Uses (Z, Y, X) order to map to image space properly
        resampled_points_zyx,
        size=1.2,
        face_color='magenta',
        name=f"points_{file_stem}",
        visible=False
    )
    viewer.add_points(
        points_lloyd,
        size=1.2,
        face_color='cyan',
        name=f"lloyd_{file_stem}",
        visible=False
    )

    # Still need to flip Lloyd points to xyz

# Force the canvas interface into active 3D mode automatically
viewer.dims.ndisplay = 3

print(f"\nProcessing Complete!")
print(f"Total processed geometries loaded in memory: {len(point_clouds)}")
print(
    f"Array dimensions check for cell '033': {point_clouds.get('033', np.array([])).shape}")

for layer in viewer.layers:
    layer.scale = [6, 1, 1]

napari.run()

# 1. Define the folder where you want your permanent files to live
# Changed name to reflect it holds both types
output_directory = "./processed_data"
os.makedirs(output_directory, exist_ok=True)

print("Writing point clouds and SDF arrays to disk...")

# 2. Loop through your cell IDs
for cell_id in point_clouds.keys():

    # Grab arrays from memory
    pt_array = point_clouds[cell_id]
    sdf_array = sdfs[cell_id]

    # 3. Save the Point Cloud NumPy array
    pt_path = os.path.join(output_directory, f"{cell_id}_ptcloud.npy")
    np.save(pt_path, pt_array)

    # 4. Save the SDF NumPy array
    sdf_path = os.path.join(output_directory, f"{cell_id}_sdf.npy")
    np.save(sdf_path, sdf_array)

print(
    f"Success! Saved {len(point_clouds)} paired files permanently in '{output_directory}/'.")
