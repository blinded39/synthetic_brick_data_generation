# 🧱 Synthetic Photorealistic Brick Data Generator

This project generates high-quality, photorealistic, and annotated images of synthetic bricks using 
- [BlenderProc](https://github.com/DLR-RM/BlenderProc),
- [bpy](https://pypi.org/project/bpy/),
- [geometry-script](https://github.com/carson-katri/geometry-script).

The resulting images and segmentation masks are used to train deep learning models for object detection and pose estimation in existing brickwork structures. The pipeline includes:
- Randomized scene generation with procedural geometry using Geometry Nodes
- Physically-based rendering with BlenderProc
- Automatic generation of segmentation masks and COCO annotations

---

## Directory Structure

```

synthetic-data-generator/
├── code/                     # Python scripts (generation, loading textures, etc.)
├── data/
│   └── generated\_data/       # Output images and annotations
├── docs/                     # Images of some samples.
├── resources/                # External assets like CC Textures
├── README.md

````

---

## Sample Outputs

Here are examples of generated data:

| RGB Image | Segmentation Mask |
|----------|-------------------|
| ![Sample RGB](docs/1.jpg) | ![Sample Mask](docs/1_mask.PNG) |
| ![Sample RGB](docs/2.jpg) | ![Sample Mask](docs/2_mask.PNG) |
| ![Sample RGB](docs/3.jpg) | ![Sample Mask](docs/3_mask.PNG) |

---

## Cloning the repo (without large files)

Under **`data/generated_data/`** you can find the data we generated to train our networks. Be careful, this is a large file! Here's how to clone the repository **without downloading LFS-tracked files** (like `data/generated_data`):

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/blinded39/synthetic_brick_data_generation.git
```

If you later want to download specific LFS files, you can run:

```bash
git lfs pull --include="data/generated_data/occluded_data"
```
---

## Installation

1. Install **Blender** version with **Python 3.10** (e.g., Blender 3.6.x)
2. Install **geometry_script** according to [instructions](https://carson-katri.github.io/geometry-script/setup/installation.html).
3. Install **BlenderProc** in version **2.5.0** according to [instructions](https://github.com/DLR-RM/BlenderProc).

---

## Usage

### 1. Load CC Textures into Resources

Download and organize your [CC Textures](https://cc0textures.com/) into the `resources/` folder. You can rename the textures as needed. These will be used randomly for the environment.

```bash
blenderproc run load_cctextures.py resources --custom-blender-path ~/blender/blender-3.6.x-linux-x64
```

### 2. Generate a Single Scene

This is the most interesting code, to generate randomized mortar-occluded brick wall geometries with randomized textures and render them with randomized settings.

```bash
blenderproc run generate.py resources --custom-blender-path ~/blender/blender-3.6.x-linux-x64
```

### 3. Render Multiple Scenes in Batch

This is to run the previous code multiple times, to generate a lot of data!

```bash
python3 run_multi.py run generate.py resources --custom-blender-path ~/blender/blender-3.6.x-linux-x64
```

### 4. Visualize Dataset

See the masked version of the images via:

```bash
blenderproc vis coco \
  -i 0 \
  -c coco_annotations.json \
  -b /home/your-name/synthetic_brick_data_generation/data/test_output/train_pbr/000000
```

Replace `/home/your-name/...` with the correct path to your dataset folder, 0 with the image number and `.../train_pbr/000000` with the folder you like.

---

## Notes

* Requires `cv2`, `skimage`, and other Python libraries.
* Designed for training models on occluded, photorealistic, and segmented brick instances.
* You can extend `generate.py` to control your output directory, number of views, lighting, camera positioning, or class variations.
