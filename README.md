# 🧱 Synthetic Photorealistic Brick Data Generator

This project generates high-quality, photorealistic, and annotated images of synthetic bricks using [BlenderProc](https://github.com/DLR-RM/BlenderProc), [bpy](https://pypi.org/project/bpy/), and [geometry-script](https://github.com/carson-katri/geometry-script). The resulting images and segmentation masks are used to train deep learning models for object detection and pose estimation in existing brickwork structures.

The pipeline includes:
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
├── resources/                # External assets like CC Textures
├── README.md

````

---

## Sample Outputs

Here are examples of generated data:

| RGB Image | Segmentation Mask |
|----------|-------------------|
| ![Sample RGB](data/generated_data/sample_rgb.png) | ![Sample Mask](data/generated_data/sample_mask.png) |

> Under **`data/generated_data/`** you can find the data we generated to train our networks. Be careful, this is a huge file!

---

## Installation

1. **Install Blender**  
   Use a Blender version with **Python 3.11** (e.g., Blender 3.6.x)

2. **Set up Conda Environment**

```bash
conda create -n gen_env python=3.11
conda activate gen_env
pip install -r blenderproc/requirements.txt
````

---

## Usage

### 1. Load CC Textures into Resources

Download and organize your [CC Textures](https://cc0textures.com/) into the `resources/` folder. You can rename the textures as needed.

```bash
blenderproc run load_cctextures.py resources --custom-blender-path ~/blender/blender-3.6.x-linux-x64
```

### 2. Generate a Single Scene

```bash
blenderproc run generate.py resources --custom-blender-path ~/blender/blender-3.6.x-linux-x64
```

### 3. Render Multiple Scenes in Batch

```bash
python run_multi.py run generate.py resources --custom-blender-path ~/blender/blender-3.6.x-linux-x64
```

### 4. Visualize Dataset

```bash
blenderproc vis coco \
  -i 0 \
  -c coco_annotations.json \
  -b /home/your-name/synthetic_brick_data_generation/data/test_output/train_pbr/000000
```

Replace `/home/your-name/...` with the correct path to your dataset folder.

---

## Notes

* Requires `cv2`, `skimage`, and other Python libraries listed in your `requirements.txt`.
* Designed for training models on occluded, photorealistic, and segmented brick instances.
* You can extend `generate.py` to control your output directory, number of views, lighting, camera positioning, or class variations.
