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


for file_path in all_files:
    ogfile = io.imread(file_path)
    ogfile = (ogfile > 0).astype(np.uint8)  # Ensure binary layout

    # Extract the two-digit file identifier (e.g., '033')
    file_stem = Path(file_path).stem

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

# napari.run()
