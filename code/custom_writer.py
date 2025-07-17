"""Allows rendering the content of the scene in the bop file format."""

import json
import os
import glob
from typing import List, Optional
import shutil

import math
import numpy as np
import cv2
import bpy
import png
from mathutils import Matrix
from typing import Optional, Dict, Union, Tuple, List
import csv
import warnings

from blenderproc.python.utility.Utility import Utility
from blenderproc.python.utility.LabelIdMapping import LabelIdMapping

from blenderproc.python.types.MeshObjectUtility import MeshObject, get_all_mesh_objects
from blenderproc.python.utility.Utility import Utility, resolve_path
from blenderproc.python.postprocessing.PostProcessingUtility import dist2depth
from blenderproc.python.writer.WriterUtility import _WriterUtility
from blenderproc.python.types.LinkUtility import Link
from blenderproc.python.writer.BopWriterUtility import _BopWriterUtility
from blenderproc.python.writer.CocoWriterUtility import _CocoWriterUtility


class _MyWriterUtility():
    """ Saves the synthesized dataset in the BOP format. The dataset is split
        into chunks which are saved as individual "scenes". For more details
        about the BOP format, visit the BOP toolkit docs:
        https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md

    """

    @staticmethod
    def load_json(path, keys_to_int=False):
        """Loads content of a JSON file.
        From the BOP toolkit (https://github.com/thodan/bop_toolkit).

        :param path: Path to the JSON file.
        :param keys_to_int: Convert digit dict keys to integers. Default: False
        :return: Content of the loaded JSON file.
        """

        # Keys to integers.
        def convert_keys_to_int(x):
            return {int(k) if k.lstrip('-').isdigit() else k: v for k, v in x.items()}

        with open(path, 'r', encoding="utf-8") as f:
            if keys_to_int:
                content = json.load(f, object_hook=convert_keys_to_int)
            else:
                content = json.load(f)

        return content

    @staticmethod
    def save_json(path, content):
        """ Saves the content to a JSON file in a human-friendly format.
        From the BOP toolkit (https://github.com/thodan/bop_toolkit).

        :param path: Path to the output JSON file.
        :param content: Dictionary/list to save.
        """
        text = ""
        with open(path, 'w', encoding="utf-8") as file:
            if isinstance(content, dict):
                text += '{\n'
                content_sorted = sorted(content.items(), key=lambda x: x[0])
                for elem_id, (k, v) in enumerate(content_sorted):
                    text += f'  "{k}": {json.dumps(v, sort_keys=True)}'
                    if elem_id != len(content) - 1:
                        text += ','
                    text += '\n'
                text += '}'
                file.write(text)
            elif isinstance(content, list):
                text += '[\n'
                for elem_id, elem in enumerate(content):
                    text += f'  {json.dumps(elem, sort_keys=True)}'
                    if elem_id != len(content) - 1:
                        text += ','
                    text += '\n'
                text += ']'
                file.write(text)
            else:
                json.dump(content, file, sort_keys=True)

    @staticmethod
    def save_depth(path: str, im: np.ndarray):
        """Saves a depth image (16-bit) to a PNG file.
        From the BOP toolkit (https://github.com/thodan/bop_toolkit).

        :param path: Path to the output depth image file.
        :param im: ndarray with the depth image to save.
        """
        if not path.endswith(".png"):
            raise ValueError('Only PNG format is currently supported.')

        im[im > 65535] = 65535
        im_uint16 = np.round(im).astype(np.uint16)

        # PyPNG library can save 16-bit PNG and is faster than imageio.imwrite().
        w_depth = png.Writer(im.shape[1], im.shape[0], greyscale=True, bitdepth=16)
        with open(path, 'wb') as f:
            w_depth.write(f, np.reshape(im_uint16, (-1, im.shape[1])))

    @staticmethod
    def write_camera(camera_path: str, depth_scale: float = 1.0):
        """ Writes camera.json into dataset_dir.
        :param camera_path: Path to camera.json
        :param depth_scale: Multiply the uint16 output depth image with this factor to get depth in mm.
        """
        # Use second frame for reading intrinsics (due to backwards compatibility)
        bpy.context.scene.frame_set(1)
        cam_K = _WriterUtility.get_cam_attribute(bpy.context.scene.camera, 'cam_K')
        camera = {'cx': cam_K[0][2],
                  'cy': cam_K[1][2],
                  'depth_scale': depth_scale,
                  'fx': cam_K[0][0],
                  'fy': cam_K[1][1],
                  'height': bpy.context.scene.render.resolution_y,
                  'width': bpy.context.scene.render.resolution_x}

        _MyWriterUtility.save_json(camera_path, camera)

    @staticmethod
    def get_frame_gt(dataset_objects: List[bpy.types.Mesh], unit_scaling: float, ignore_dist_thres: float,
                     destination_frame: Optional[List[str]] = None):
        """ Returns GT pose annotations between active camera and objects.
        :param dataset_objects: Save annotations for these objects.
        :param unit_scaling: 1000. for outputting poses in mm
        :param ignore_dist_thres: Distance between camera and object after which object is ignored.
                                  Mostly due to failed physics.
        :param destination_frame: Transform poses from Blender internal coordinates to OpenCV coordinates
        :return: A list of GT camera-object pose annotations for scene_gt.json
        """
        if destination_frame is None:
            destination_frame = ["X", "-Y", "-Z"]

        H_c2w_opencv = Matrix(_WriterUtility.get_cam_attribute(bpy.context.scene.camera, 'cam2world_matrix',
                                                               local_frame_change=destination_frame))

        frame_gt = []
        for obj in dataset_objects:
            if isinstance(obj, Link):
                if not obj.visuals:
                    continue
                if len(obj.visuals) > 1:
                    warnings.warn('BOP Writer only supports saving poses of one visual mesh per Link')
                H_m2w = Matrix(obj.get_visual_local2world_mats()[0])
            else:
                H_m2w = Matrix(obj.get_local2world_mat())
                assert obj.has_cp("category_id"), f"{obj.get_name()} object has no custom property 'category_id'"
            
            H_rot_x = Matrix.Rotation(math.radians(90.0), 4, 'X')
            cam_H_m2c = H_c2w_opencv.inverted() @ H_m2w @ H_rot_x
            cam_R_m2c = cam_H_m2c.to_quaternion().to_matrix()
            cam_t_m2c = cam_H_m2c.to_translation()
            scale_nocs2m = (obj.get_scale() * 1000).reshape(3).tolist()
            
            # ignore examples that fell through the plane
            if not np.linalg.norm(list(cam_t_m2c)) > ignore_dist_thres:
                cam_t_m2c = list(cam_t_m2c * unit_scaling)
                frame_gt.append({
                    'cam_R_m2c': list(cam_R_m2c[0]) + list(cam_R_m2c[1]) + list(cam_R_m2c[2]),
                    'cam_t_m2c': cam_t_m2c,
                    'scale_nocs2m': scale_nocs2m , 
                    'NOCS_category_id': obj.get_cp("category_id") if not isinstance(obj, Link) else obj.visuals[0].get_cp(
                        'category_id'),
                    'shapenet_category_id': obj.get_cp("used_synset_id"),
                    'shapenet_instance_id': obj.get_cp("used_source_id"),

                })
            else:
                print('ignored obj, ', obj["category_id"], 'because either ')
                print('(1) it is further away than parameter "ignore_dist_thres: ",', ignore_dist_thres)
                print('(e.g. because it fell through a plane during physics sim)')
                print('or')
                print('(2) the object pose has not been given in meters')

        return frame_gt

    @staticmethod
    def get_frame_camera(save_world2cam: bool, depth_scale: float = 1.0, unit_scaling: float = 1000.,
                         destination_frame: Optional[List[str]] = None):
        """ Returns camera parameters for the active camera.
        :param save_world2cam: If true, camera to world transformations "cam_R_w2c", "cam_t_w2c" are saved
                               in scene_camera.json
        :param depth_scale: Multiply the uint16 output depth image with this factor to get depth in mm.
        :param unit_scaling: 1000. for outputting poses in mm
        :param destination_frame: Transform poses from Blender internal coordinates to OpenCV coordinates
        :return: dict containing info for scene_camera.json
        """
        if destination_frame is None:
            destination_frame = ["X", "-Y", "-Z"]

        cam_K = _WriterUtility.get_cam_attribute(bpy.context.scene.camera, 'cam_K')

        frame_camera_dict = {
            'cam_K': cam_K[0] + cam_K[1] + cam_K[2],
            'depth_scale': depth_scale
        }

        if save_world2cam:
            H_c2w_opencv = Matrix(_WriterUtility.get_cam_attribute(bpy.context.scene.camera, 'cam2world_matrix',
                                                                   local_frame_change=destination_frame))

            H_w2c_opencv = H_c2w_opencv.inverted()
            R_w2c_opencv = H_w2c_opencv.to_quaternion().to_matrix()
            t_w2c_opencv = H_w2c_opencv.to_translation() * unit_scaling

            frame_camera_dict['cam_R_w2c'] = list(R_w2c_opencv[0]) + list(R_w2c_opencv[1]) + list(R_w2c_opencv[2])
            frame_camera_dict['cam_t_w2c'] = list(t_w2c_opencv)

        return frame_camera_dict

    @staticmethod
    def write_frames(chunks_dir: str, dataset_objects: list, 
                     depths: Optional[List[np.ndarray]] = None,
                     colors: Optional[List[np.ndarray]] = None, 
                     depths_noise: Optional[List[np.ndarray]] = None,
                     instance_masks: Optional[List[np.ndarray]] = None, 
                     category_masks: Optional[List[np.ndarray]] = None, 
                     color_file_format: str = "PNG",
                     depth_scale: float = 1.0, frames_per_chunk: int = 1000, m2mm: bool = True,
                     ignore_dist_thres: float = 100., save_world2cam: bool = True, jpg_quality: int = 95, is_shapenet=True):
        """Write each frame's ground truth into chunk directory in BOP format

        :param chunks_dir: Path to the output directory of the current chunk.
        :param dataset_objects: Save annotations for these objects.
        :param depths: List of depth images in m to save
        :param colors: List of color images to save
        :param color_file_format: File type to save color images. Available: "PNG", "JPEG"
        :param jpg_quality: If color_file_format is "JPEG", save with the given quality.
        :param depth_scale: Multiply the uint16 output depth image with this factor to get depth in mm. Used to
                            trade-off between depth accuracy and maximum depth value. Default corresponds to
                            65.54m maximum depth and 1mm accuracy.
        :param ignore_dist_thres: Distance between camera and object after which object is ignored.
                                  Mostly due to failed physics.
        :param m2mm: Original bop annotations and models are in mm. If true, we convert the gt annotations
                     to mm here. This is needed if BopLoader option mm2m is used.
        :param frames_per_chunk: Number of frames saved in each chunk (called scene in BOP)
        """

        # Format of the depth images.
        depth_ext = '.png'
        mask_ext = '.png'

        rgb_tpath = os.path.join(chunks_dir, '{chunk_id:06d}', 'rgb', '{im_id:06d}' + '{im_type}')
        depth_tpath = os.path.join(chunks_dir, '{chunk_id:06d}', 'depth', '{im_id:06d}' + depth_ext)
        depth_noise_tpath = os.path.join(chunks_dir, '{chunk_id:06d}', 'depth_noise', '{im_id:06d}' + depth_ext)
        instance_tpath = os.path.join(chunks_dir, '{chunk_id:06d}', 'instance', '{im_id:06d}' + mask_ext)
        category_tpath = os.path.join(chunks_dir, '{chunk_id:06d}', 'category', '{im_id:06d}' + mask_ext)
        chunk_camera_tpath = os.path.join(chunks_dir, '{chunk_id:06d}', 'scene_camera.json')
        chunk_gt_tpath = os.path.join(chunks_dir, '{chunk_id:06d}', 'scene_gt.json')

        # Paths to the already existing chunk folders (such folders may exist
        # when appending to an existing dataset).
        chunk_dirs = sorted(glob.glob(os.path.join(chunks_dir, '*')))
        chunk_dirs = [d for d in chunk_dirs if os.path.isdir(d)]

        # Get ID's of the last already existing chunk and frame.
        curr_chunk_id = 0
        curr_frame_id = 0
        if len(chunk_dirs):
            last_chunk_dir = sorted(chunk_dirs)[-1]
            last_chunk_gt_fpath = os.path.join(last_chunk_dir, 'scene_gt.json')
            chunk_gt = _MyWriterUtility.load_json(last_chunk_gt_fpath, keys_to_int=True)

            # Last chunk and frame ID's.
            last_chunk_id = int(os.path.basename(last_chunk_dir))
            last_frame_id = int(sorted(chunk_gt.keys())[-1])

            # Current chunk and frame ID's.
            curr_chunk_id = last_chunk_id
            curr_frame_id = last_frame_id + 1
            if curr_frame_id % frames_per_chunk == 0:
                curr_chunk_id += 1
                curr_frame_id = 0

        # Initialize structures for the GT annotations and camera info.
        chunk_gt = {}
        chunk_camera = {}
        if curr_frame_id != 0:
            # Load GT and camera info of the chunk we are appending to.
            chunk_gt = _MyWriterUtility.load_json(
                chunk_gt_tpath.format(chunk_id=curr_chunk_id), keys_to_int=True)
            chunk_camera = _MyWriterUtility.load_json(
                chunk_camera_tpath.format(chunk_id=curr_chunk_id), keys_to_int=True)

        # Go through all frames.
        num_new_frames = bpy.context.scene.frame_end - bpy.context.scene.frame_start

        if len(depths) != len(depths_noise) != len(colors) != len(instance_masks) != len(category_masks) != num_new_frames:
            raise Exception("The amount of images stored in the depths/colors does not correspond to the amount"
                            "of images specified by frame_start to frame_end.")

        for frame_id in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end):
            # Activate frame.
            bpy.context.scene.frame_set(frame_id)

            # Reset data structures and prepare folders for a new chunk.
            if curr_frame_id == 0:
                chunk_gt = {}
                chunk_camera = {}
                os.makedirs(os.path.dirname(
                    rgb_tpath.format(chunk_id=curr_chunk_id, im_id=0, im_type='PNG')))
                os.makedirs(os.path.dirname(
                    depth_tpath.format(chunk_id=curr_chunk_id, im_id=0)))
                os.makedirs(os.path.dirname(
                    depth_noise_tpath.format(chunk_id=curr_chunk_id, im_id=0)))

            # Output translation gt in m or mm
            unit_scaling = 1000. if m2mm else 1.

            if is_shapenet:
                chunk_gt[curr_frame_id] = _MyWriterUtility.get_frame_gt(dataset_objects, unit_scaling, ignore_dist_thres)
                chunk_camera[curr_frame_id] = _MyWriterUtility.get_frame_camera(save_world2cam, depth_scale, unit_scaling)
            else:
                chunk_gt[curr_frame_id] = _BopWriterUtility.get_frame_gt(dataset_objects, unit_scaling, ignore_dist_thres)
                chunk_camera[curr_frame_id] = _BopWriterUtility.get_frame_camera(save_world2cam, depth_scale, unit_scaling)

            if colors:
                color_rgb = colors[frame_id]
                color_bgr = color_rgb.copy()
                color_bgr[..., :3] = color_bgr[..., :3][..., ::-1]
                if color_file_format == 'PNG':
                    rgb_fpath = rgb_tpath.format(chunk_id=curr_chunk_id, im_id=curr_frame_id, im_type='.png')
                    cv2.imwrite(rgb_fpath, color_bgr)
                elif color_file_format == 'JPEG':
                    rgb_fpath = rgb_tpath.format(chunk_id=curr_chunk_id, im_id=curr_frame_id, im_type='.jpg')
                    cv2.imwrite(rgb_fpath, color_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
            else:
                rgb_output = Utility.find_registered_output_by_key("colors")
                if rgb_output is None:
                    raise Exception("RGB image has not been rendered.")
                color_ext = '.png' if rgb_output['path'].endswith('png') else '.jpg'
                # Copy the resulting RGB image.
                rgb_fpath = rgb_tpath.format(chunk_id=curr_chunk_id, im_id=curr_frame_id, im_type=color_ext)
                shutil.copyfile(rgb_output['path'] % frame_id, rgb_fpath)

            if depths:
                depth = depths[frame_id]
            else:
                # Load the resulting dist image.
                dist_output = Utility.find_registered_output_by_key("distance")
                if dist_output is None:
                    raise Exception("Distance image has not been rendered.")
                distance = _WriterUtility.load_output_file(resolve_path(dist_output['path'] % frame_id), remove=False)
                depth = dist2depth(distance)

            if depths_noise:
                depth_noise = depths_noise[frame_id]

            if instance_masks:
                instance_mask = instance_masks[frame_id]
            else:
                # Find path pattern of segmentation images
                instance_mask_key = 'segmap'
                instance_mask = Utility.find_registered_output_by_key(instance_mask_key)
                if instance_mask is None:
                    raise RuntimeError(f"There is no output registered with key {instance_mask_key}. Are you sure you "
                                    f"ran the SegMapRenderer module before?")

            if category_masks:
                category_mask = category_masks[frame_id]
            else:
                # Find path pattern of segmentation images
                category_mask_key = 'segmap'
                category_mask = Utility.find_registered_output_by_key(category_mask_key)
                if category_mask is None:
                    raise RuntimeError(f"There is no output registered with key {category_mask_key}. Are you sure you "
                                    f"ran the SegMapRenderer module before?")

            # Scale the depth to retain a higher precision (the depth is saved
            # as a 16-bit PNG image with range 0-65535).
            depth_mm = 1000.0 * depth  # [m] -> [mm]
            depth_mm_scaled = depth_mm / float(depth_scale)
            
            # Save the scaled depth image.
            depth_fpath = depth_tpath.format(chunk_id=curr_chunk_id, im_id=curr_frame_id)
            _MyWriterUtility.save_depth(depth_fpath, depth_mm_scaled)
    
            depth_noise_mm = 1000.0 * depth_noise  # [m] -> [mm]
            depth_noise_mm_scaled = depth_noise_mm / float(depth_scale)

            # Save the scaled noisy depth image.
            depth_noise_fpath = depth_noise_tpath.format(chunk_id=curr_chunk_id, im_id=curr_frame_id)
            _MyWriterUtility.save_depth(depth_noise_fpath, depth_noise_mm_scaled)


            # Save the chunk info if we are at the end of a chunk or at the last new frame.
            if ((curr_frame_id + 1) % frames_per_chunk == 0) or \
                    (frame_id == num_new_frames - 1):

                # Save GT annotations.
                _MyWriterUtility.save_json(chunk_gt_tpath.format(chunk_id=curr_chunk_id), chunk_gt)

                # Save camera info.
                _MyWriterUtility.save_json(chunk_camera_tpath.format(chunk_id=curr_chunk_id), chunk_camera)

                # Update ID's.
                curr_chunk_id += 1
                curr_frame_id = 0
            else:
                curr_frame_id += 1
        return curr_chunk_id-1


def write_my(output_dir: str, target_objects: Optional[List[MeshObject]] = None,
              depths: Optional[List[np.ndarray]] = None, 
              depths_noise: Optional[List[np.ndarray]] = None, 
              colors: Optional[List[np.ndarray]] = None,
              instance_masks: Optional[List[np.ndarray]] = None, category_masks: Optional[List[np.ndarray]] = None,
              instance_attribute_maps: Optional[List[np.ndarray]] = None,
              color_file_format: str = "PNG", dataset: str = "", append_to_existing_output: bool = True,
              depth_scale: float = 1.0, jpg_quality: int = 95, save_world2cam: bool = True,
              ignore_dist_thres: float = 100., m2mm: bool = True, frames_per_chunk: int = 1000, 
              is_shapenet=True, chunk_name='train_pbr'):
    """Write the BOP data

    :param output_dir: Path to the output directory.
    :param target_objects: Objects for which to save ground truth poses in BOP format. Default: Save all objects or
                           from specified dataset
    :param depths: List of depth images in m to save
    :param colors: List of color images to save
    :param color_file_format: File type to save color images. Available: "PNG", "JPEG"
    :param jpg_quality: If color_file_format is "JPEG", save with the given quality.
    :param dataset: Only save annotations for objects of the specified bop dataset. Saves all object poses if undefined.
    :param append_to_existing_output: If true, the new frames will be appended to the existing ones.
    :param depth_scale: Multiply the uint16 output depth image with this factor to get depth in mm. Used to trade-off
                        between depth accuracy and maximum depth value. Default corresponds to 65.54m maximum depth
                        and 1mm accuracy.
    :param save_world2cam: If true, camera to world transformations "cam_R_w2c", "cam_t_w2c" are saved
                           in scene_camera.json
    :param ignore_dist_thres: Distance between camera and object after which object is ignored. Mostly due to
                              failed physics.
    :param m2mm: Original bop annotations and models are in mm. If true, we convert the gt annotations to mm here. This
                 is needed if BopLoader option mm2m is used.
    :param frames_per_chunk: Number of frames saved in each chunk (called scene in BOP)
    """
    if depths is None:
        depths = []
    if depths_noise is None:
        depths_noise = []
    if colors is None:
        colors = []
    if instance_masks is None:
        instance_masks = []
    if category_masks is None:
        category_masks = []

    # Output paths.
    dataset_dir = os.path.join(output_dir, dataset)
    chunks_dir = os.path.join(dataset_dir, chunk_name)
    camera_path = os.path.join(dataset_dir, 'camera_{}.json'.format(chunk_name))
    coco_annotations_tpath =  os.path.join(chunks_dir, '{chunk_id:06d}')

    # Create the output directory structure.
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        os.makedirs(chunks_dir)
    elif not append_to_existing_output:
        raise FileExistsError(f"The output folder already exists: {dataset_dir}")

    # Select target objects or objects from the specified dataset or all objects
    if target_objects is not None:
        dataset_objects = target_objects
    elif dataset:
        dataset_objects = []
        for obj in get_all_mesh_objects():
            if "bop_dataset_name" in obj.blender_obj and not obj.blender_obj.hide_render:
                if obj.blender_obj["bop_dataset_name"] == dataset:
                    dataset_objects.append(obj)
    else:
        dataset_objects = get_all_mesh_objects()

    # Check if there is any object from the specified dataset.
    if not dataset_objects:
        raise RuntimeError(f"The scene does not contain any object from the specified dataset: {dataset}. "
                           f"Either remove the dataset parameter or assign custom property 'bop_dataset_name'"
                           f" to selected objects")

    # Save the data.
    _MyWriterUtility.write_camera(camera_path, depth_scale=depth_scale)
    curr_chunk_id = _MyWriterUtility.write_frames(chunks_dir, dataset_objects=dataset_objects, depths=depths, 
                                                depths_noise=depths_noise, colors=colors, 
                                                instance_masks=instance_masks, category_masks=category_masks,
                                                color_file_format=color_file_format, frames_per_chunk=frames_per_chunk,
                                                m2mm=m2mm, ignore_dist_thres=ignore_dist_thres, save_world2cam=save_world2cam,
                                                depth_scale=depth_scale, jpg_quality=jpg_quality, is_shapenet=is_shapenet)
    write_my_coco_annotations(coco_annotations_tpath.format(chunk_id=curr_chunk_id), 
    instance_masks, instance_attribute_maps, colors, color_file_format)


def write_my_coco_annotations(output_dir: str, instance_segmaps: Optional[List[np.ndarray]] = None,
                           instance_attribute_maps: Optional[List[dict]] = None,
                           colors: Optional[List[np.ndarray]] = None, color_file_format: str = "PNG",
                           mask_encoding_format: str = "rle", supercategory: str = "coco_annotations",
                           append_to_existing_output: bool = True, segmap_output_key: str = "segmap",
                           segcolormap_output_key: str = "segcolormap", rgb_output_key: str = "colors",
                           jpg_quality: int = 95, label_mapping: Optional[LabelIdMapping] = None,
                           file_prefix: str = "", indent: Optional[Union[int, str]] = None):
    """ Writes coco annotations in the following steps:
    1. Locate the seg images
    2. Locate the rgb maps
    3. Locate the seg mappings
    4. Read color mappings
    5. For each frame write the coco annotation

    :param output_dir: Output directory to write the coco annotations
    :param instance_segmaps: List of instance segmentation maps
    :param instance_attribute_maps: per-frame mappings with idx, class and optionally supercategory/bop_dataset_name
    :param colors: List of color images. Does not support stereo images, enter left and right inputs subsequently.
    :param color_file_format: Format to save color images in
    :param mask_encoding_format: Encoding format of the binary masks. Default: 'rle'. Available: 'rle', 'polygon'.
    :param supercategory: name of the dataset/supercategory to filter for, e.g. a specific BOP dataset set
                          by 'bop_dataset_name' or any loaded object with specified 'cp_supercategory'
    :param append_to_existing_output: If true and if there is already a coco_annotations.json file in the output
                                      directory, the new coco annotations will be appended to the existing file.
                                      Also, the rgb images will be named such that there are no collisions.
    :param segmap_output_key: The output key with which the segmentation images were registered. Should be the same
                              as the output_key of the SegMapRenderer module. Default: segmap.
    :param segcolormap_output_key: The output key with which the csv file for object name/class correspondences
                                   was registered. Should be the same as the colormap_output_key of the
                                   SegMapRenderer module. Default: segcolormap.
    :param rgb_output_key: The output key with which the rgb images were registered. Should be the same as
                           the output_key of the RgbRenderer module. Default: colors.
    :param jpg_quality: The desired quality level of the jpg encoding
    :param label_mapping: The label mapping which should be used to label the categories based on their ids.
                          If None, is given then the `name` field in the csv files is used or - if not existing -
                          the category id itself is used.
    :param file_prefix: Optional prefix for image file names
    :param indent: If indent is a non-negative integer or string, then the annotation output
                   will be pretty-printed with that indent level. An indent level of 0, negative, or "" will
                   only insert newlines. None (the default) selects the most compact representation.
                   Using a positive integer indent indents that many spaces per level.
                   If indent is a string (such as "\t"), that string is used to indent each level.
    """
    instance_segmaps = [] if instance_segmaps is None else list(instance_segmaps)
    colors = [] if colors is None else list(colors)
    if instance_attribute_maps is None:
        instance_attribute_maps = []

    if len(colors) > 0 and len(colors[0].shape) == 4:
        raise ValueError("BlenderProc currently does not support writing coco annotations for stereo images. "
                         "However, you can enter left and right images / segmaps separately.")

    # Create output directory
    os.makedirs(os.path.join(output_dir), exist_ok=True)

    if not instance_segmaps:
        # Find path pattern of segmentation images
        segmentation_map_output = Utility.find_registered_output_by_key(segmap_output_key)
        if segmentation_map_output is None:
            raise RuntimeError(f"There is no output registered with key {segmap_output_key}. Are you sure you "
                               f"ran the SegMapRenderer module before?")

    if not colors:
        # Find path pattern of rgb images
        rgb_output = Utility.find_registered_output_by_key(rgb_output_key)
        if rgb_output is None:
            raise RuntimeError(f"There is no output registered with key {rgb_output_key}. Are you sure you "
                               f"ran the RgbRenderer module before?")

    if not instance_attribute_maps:
        # Find path of name class mapping csv file
        segcolormap_output = Utility.find_registered_output_by_key(segcolormap_output_key)
        if segcolormap_output is None:
            raise RuntimeError(f"There is no output registered with key {segcolormap_output_key}. Are you sure you "
                               f"ran the SegMapRenderer module with 'map_by' set to 'instance' before?")

    coco_annotations_path = os.path.join(output_dir, "coco_annotations.json")
    # Calculate image numbering offset, if append_to_existing_output is activated and coco data exists
    if append_to_existing_output and os.path.exists(coco_annotations_path):
        with open(coco_annotations_path, 'r', encoding="utf-8") as fp:
            existing_coco_annotations = json.load(fp)
        image_offset = max(image["id"] for image in existing_coco_annotations["images"]) + 1
    else:
        image_offset = 0
        existing_coco_annotations = None

    # collect all RGB paths
    new_coco_image_paths = []
    # collect all mappings from csv (backwards compat)
    segcolormaps = []
    # collect all instance segmaps (backwards compat)
    inst_segmaps = []

    # for each rendered frame
    for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end):

        if not instance_attribute_maps:
            # read colormappings, which include object name/class to integer mapping
            segcolormap = []
            with open(segcolormap_output["path"] % frame, 'r', encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for mapping in reader:
                    segcolormap.append(mapping)
            segcolormaps.append(segcolormap)

        if not instance_segmaps:
            # Load segmaps (backwards compat)
            segmap = np.load(segmentation_map_output["path"] % frame)
            inst_channel = int(segcolormap[0]['channel_instance'])
            inst_segmaps.append(segmap[:, :, inst_channel])

        if colors:
            color_rgb = colors[frame - bpy.context.scene.frame_start]
            if color_file_format == 'PNG':
                target_base_path = f'rgb/{file_prefix}{frame + image_offset:06d}.png'
                target_path = os.path.join(output_dir, target_base_path)
            elif color_file_format == 'JPEG':
                target_base_path = f'rgb/{file_prefix}{frame + image_offset:06d}.jpg'
                target_path = os.path.join(output_dir, target_base_path)
            else:
                raise RuntimeError(f'Unknown color_file_format={color_file_format}. Try "PNG" or "JPEG"')

            # Reverse channel order for opencv
            color_bgr = color_rgb.copy()
            color_bgr[..., :3] = color_bgr[..., :3][..., ::-1]

        else:
            source_path = rgb_output["path"] % frame
            target_base_path = os.path.join('rgb',
                                            file_prefix + os.path.basename(rgb_output["path"] % (frame + image_offset)))
            target_path = os.path.join(output_dir, target_base_path)
            shutil.copyfile(source_path, target_path)

        new_coco_image_paths.append(target_base_path)

    instance_attibute_maps = segcolormaps if segcolormaps else instance_attribute_maps
    instance_segmaps = inst_segmaps if inst_segmaps else instance_segmaps

    coco_output = _CocoWriterUtility.generate_coco_annotations(instance_segmaps,
                                                               instance_attibute_maps,
                                                               new_coco_image_paths,
                                                               supercategory,
                                                               mask_encoding_format,
                                                               existing_coco_annotations,
                                                               label_mapping)

    print("Writing coco annotations to " + coco_annotations_path)
    with open(coco_annotations_path, 'w', encoding="utf-8") as fp:
        json.dump(coco_output, fp, indent=indent)

