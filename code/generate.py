import blenderproc as bproc

import os
import sys
import numpy as np
import bpy
from geometry_script import *
import time
import random

# Setup path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from custom_writer import write_my

################################################################
# Blenderproc parameters
################################################################

out_foldername = 'test_output' # Folder name to save the output.
num_views = 15 # Number of views from a single geometry.

intr = np.array([[607.156, 0 , 324.652],  
                 [0, 607.194 , 236.489], 
                 [0, 0, 1]]).astype(np.float32) # Camera intrinsics.
intr = intr.reshape(-1).tolist()

catId2name = {0: 'Unknown', 1: 'Brick'}

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
cfg = dict(H=480, W=640, K=intr, 
        out_dir=os.path.join(base_dir, 'data', out_foldername), 
        cc_textures_path=os.path.join(base_dir, 'resources'))
K = np.array(cfg['K']).reshape(3, 3)
out_dir = cfg['out_dir']

mat_dir = {'Brick': os.path.join(base_dir, 'resources', 'brick_material.blend'),
           'Mortar':os.path.join(base_dir, 'resources', 'mortar_material.blend')}

################################################################
# Bpy functions
################################################################

def purge_orphans():
    """
    Remove all orphan data blocks
    """
    if bpy.app.version >= (3, 0, 0):
        # run this only for Blender versions 3.0 and higher
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    else:
        # run this only for Blender versions lower than 3.0
        # call purge_orphans() recursively until there are no more orphan data blocks to purge
        result = bpy.ops.outliner.orphans_purge()
        if result.pop() != "CANCELLED":
            purge_orphans()


def clean_scene():
    """
    Removing all of the objects, collection, materials, particles,
    textures, images, curves, meshes, actions, nodes, and worlds from the scene
    """
    # make sure the active object is not in Edit Mode
    if bpy.context.active_object and bpy.context.active_object.mode == "EDIT":
        bpy.ops.object.editmode_toggle()

    # make sure non of the objects are hidden from the viewport, selection, or disabled
    for obj in bpy.data.objects:
        obj.hide_set(False)
        obj.hide_select = False
        obj.hide_viewport = False

    # select all the object and delete them (just like pressing A + X + D in the viewport)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # find all the collections and remove them
    collection_names = [col.name for col in bpy.data.collections]
    for name in collection_names:
        bpy.data.collections.remove(bpy.data.collections[name])

    # in the case when you modify the world shader
    # delete and recreate the world object
    world_names = [world.name for world in bpy.data.worlds]
    for name in world_names:
        bpy.data.worlds.remove(bpy.data.worlds[name])
    # create a new world data block
    bpy.ops.world.new()
    bpy.context.scene.world = bpy.data.worlds["World"]

    purge_orphans()


def active_object():
    """
    returns the currently active object
    """
    return bpy.context.active_object


def time_seed():
    """
    Sets the random seed based on the time
    and copies the seed into the clipboard
    """
    seed = time.time()
    print(f"seed: {seed}")
    random.seed(seed)

    # add the seed value to your clipboard
    bpy.context.window_manager.clipboard = str(seed)

    return seed


def set_scene_props(fps, frame_count):
    """
    Set scene properties
    """
    scene = bpy.context.scene
    scene.frame_end = frame_count

    # set the world background to black
    world = bpy.data.worlds["World"]
    if "Background" in world.node_tree.nodes:
        world.node_tree.nodes["Background"].inputs[0].default_value = (0, 0, 0, 1)

    scene.render.fps = fps

    scene.frame_current = 1
    scene.frame_start = 1


def scene_setup():
    fps = 30
    loop_seconds = 12
    frame_count = fps * loop_seconds

    seed = 0
    if seed:
        random.seed(seed)
    else:
        time_seed()

    clean_scene()

    set_scene_props(fps, frame_count)


################################################################
# Set the scene
################################################################
    
scene_setup()

bproc.init()
bproc.camera.set_intrinsics_from_K_matrix(K,  cfg['W'],  cfg['H'])

with bpy.data.libraries.load(mat_dir['Mortar']) as (data_from, data_to):
    data_to.materials = ['Mortar']

with bpy.data.libraries.load(mat_dir['Brick']) as (data_from, data_to):
    data_to.materials = ['Brick']

# Important tree is created after the cleanup
@tree("Brickwall")
def brickwall(geometry: Geometry, brick: Object, top_mortar: Object, brick_layers: Int):

    # Brick dimensions
    brick_length = 0.237 + random.uniform(0.01, 0.03) # brick length + space in between
    brick_height = 0.10 + random.uniform(0.03, 0.06) # brick height + space in between
    
    print("Brick height is {}, length is {}.".format(brick_height, brick_length))

    brick_cube = object_info(object=brick).geometry
    top_mortar_cube = object_info(object=top_mortar).geometry
    mortar_material = bpy.data.materials.get("Mortar")

    # Create the wall's base curve
    base_curve = bezier_segment(mode=BezierSegment.Mode.POSITION, resolution=16, 
                                start=(random.uniform(-1, -0.2), random.uniform(-0.5, 0.5), 0), 
                                end=(random.uniform(1, 0.2), random.uniform(-0.5, 0.5), 0), 
                                start_handle=(random.uniform(-1, 0), random.uniform(-1, 1), 0), 
                                end_handle=(random.uniform(0, 1), random.uniform(-1, 1), 0))

    # Resample curve with brick length
    resampled_curve = resample_curve(mode=ResampleCurve.Mode.LENGTH, curve=base_curve, length=brick_length)
    named_curve = store_named_attribute(data_type=StoreNamedAttribute.DataType.FLOAT_VECTOR, domain=StoreNamedAttribute.Domain.POINT, geometry=resampled_curve, name='tan', value=curve_tangent())

    meshed_curve = curve_to_mesh(curve=named_curve)
    curve_attribute = capture_attribute(geometry=meshed_curve, data_type=CaptureAttribute.DataType.BOOLEAN, domain=CaptureAttribute.Domain.FACE, value=is_shade_smooth())
    
    # Extrude the wall 
    wall_mesh = extrude_mesh(mesh=curve_attribute.geometry, mode=ExtrudeMesh.Mode.EDGES, offset=(0,0,1), offset_scale=brick_height).mesh
    smoooth_wall_mesh = set_shade_smooth(geometry=wall_mesh, shade_smooth=curve_attribute.attribute)
    duplicated_meshes_result = duplicate_elements(domain=DuplicateElements.Domain.FACE, geometry=smoooth_wall_mesh, amount=brick_layers)

    top_layer_height = duplicated_meshes_result.duplicate_index*brick_height
    duplicated_meshes_positioned = set_position(geometry=duplicated_meshes_result.geometry, offset=combine_xyz(x=0.0, y=0.0, z=top_layer_height))
    full_wall_mesh = merge_by_distance(mode=MergeByDistance.Mode.ALL, geometry=duplicated_meshes_positioned, distance=0.001)
    
    # Middle layer brick points 
    middle_brick_points = mesh_to_points(mode=MeshToPoints.Mode.FACES, mesh=full_wall_mesh)

    # Main layer brick points
    top_layer_selection = compare(operation=Compare.Operation.GREATER_THAN, a=separate_xyz(vector=position()).z, b=top_layer_height)
    main_brick_points = separate_geometry(domain=SeparateGeometry.Domain.POINT, geometry=full_wall_mesh, selection=top_layer_selection)

    # Delete some bricks
    delete_bricks = False
    if delete_bricks:
        all = False
        if all:
            print("Delete all top bricks.")
            delete_possibility = random_value(data_type=RandomValue.DataType.BOOLEAN, probability=0.9)
        else:
            print("Delete some bricks.")
            delete_possibility = random_value(data_type=RandomValue.DataType.BOOLEAN, probability=0.6)
        rest_top_layer_points = delete_geometry(mode=DeleteGeometry.Mode.ALL, domain=DeleteGeometry.Domain.POINT, geometry=main_brick_points.selection, selection=delete_possibility)
    else:
        rest_top_layer_points = main_brick_points.selection

    # Bring together all the points
    full_wall_geometry = join_geometry(geometry=[middle_brick_points, main_brick_points.inverted, rest_top_layer_points])

    # Place the instances of the brick on the wall
    brick_rotation = align_euler_to_vector(vector=named_attribute(data_type=NamedAttribute.DataType.FLOAT_VECTOR, name='tan')["attribute"])

    brick_instances = instance_on_points(points=full_wall_geometry, instance=brick_cube, rotation=brick_rotation)

    # # Create mortar on some top layer
    # top_mortar_possibility = random_value(data_type=RandomValue.DataType.BOOLEAN, probability=0.3)
    # top_mortar_points = delete_geometry(mode=DeleteGeometry.Mode.ALL, domain=DeleteGeometry.Domain.POINT, geometry=rest_top_layer_points, selection=top_mortar_possibility)
    # top_mortar_instances = instance_on_points(points=top_mortar_points, instance=top_mortar_cube, rotation=brick_rotation)
    # top_mortar_wm = set_material(geometry=top_mortar_instances, material=mortar_material)

    # Create the mortar layer
    mortar_random_seed = random.uniform(1, 200)
    mortar_style = True

    if mortar_style:
        print("Thin mortar style.")
        # Create the double layer thin mortar
        mortar_density = random.uniform(2500.000, 4000.000)
        mortar_volume_radius = random.uniform(0.025, 0.035)
        mortar_placement = random.uniform(0.012, 0.02)

        mortar_mesh = duplicate_elements(domain=DuplicateElements.Domain.FACE, geometry=full_wall_mesh, amount=2)
        duplicated_mortar_mesh = set_position(geometry=mortar_mesh.geometry, offset=vector_math(operation=VectorMath.Operation.SCALE, vector=normal(), scale = mortar_placement * (1 - 2 * mortar_mesh.duplicate_index)))
        mortar_points = distribute_points_on_faces(distribute_method=DistributePointsOnFaces.DistributeMethod.RANDOM, mesh=duplicated_mortar_mesh, density=mortar_density, seed=mortar_random_seed).points
        mortar_volume = points_to_volume(resolution_mode=PointsToVolume.ResolutionMode.VOXEL_AMOUNT, points=mortar_points, density=500.000, voxel_amount=900.000, radius=mortar_volume_radius)
        mortar_final_mesh = volume_to_mesh(resolution_mode=VolumeToMesh.ResolutionMode.GRID, volume=mortar_volume, threshold=9.900, adaptivity=0.239)
        mortar = set_shade_smooth(geometry=mortar_final_mesh)
        # More distortion
        noise = noise_texture(noise_dimensions=NoiseTexture.NoiseDimensions._3D, scale=0.5, detail=20.000, roughness=1.000, distortion=80.000).color
        distorted_mortar = set_position(geometry=mortar, offset=vector_math(operation=VectorMath.Operation.MULTIPLY, vector=(normal(), math(operation=Math.Operation.POWER, value=(noise, 9.000)))))
    else:
        print("Thick mortar style.")
        # Create the single layer thick mortar
        mortar_density = random.uniform(800.000, 1000.000)
        mortar_volume_radius = random.uniform(0.05, 0.06)

        mortar_points = distribute_points_on_faces(distribute_method=DistributePointsOnFaces.DistributeMethod.RANDOM, mesh=full_wall_mesh, density=mortar_density, seed=mortar_random_seed).points
        mortar_volume = points_to_volume(resolution_mode=PointsToVolume.ResolutionMode.VOXEL_AMOUNT, points=mortar_points, density=10.000, voxel_amount=300.000, radius=mortar_volume_radius)
        mortar_final_mesh = volume_to_mesh(resolution_mode=VolumeToMesh.ResolutionMode.GRID, volume=mortar_volume, threshold=9.900, adaptivity=0.239)
        mortar = set_shade_smooth(geometry=mortar_final_mesh)
        # More distortion
        noise = noise_texture(noise_dimensions=NoiseTexture.NoiseDimensions._3D, scale=0.5, detail=20.000, roughness=1.000, distortion=80.000).color
        distorted_mortar = set_position(geometry=mortar, offset=vector_math(operation=VectorMath.Operation.MULTIPLY, vector=(normal(), math(operation=Math.Operation.POWER, value=(noise, 8.000)))))

    # Set mortar material here
    mortar_wm = set_material(geometry=distorted_mortar, material=mortar_material)

    full_model = join_geometry(geometry=[brick_instances, mortar_wm])
    return full_model

@tree("ModifyBrick")
def modify_brick(geometry: Geometry, dist_seed: Int):
    subdivided_geometry = subdivision_surface(uv_smooth=SubdivisionSurface.UvSmooth.PRESERVE_BOUNDARIES, boundary_smooth=SubdivisionSurface.BoundarySmooth.ALL, mesh=geometry, level=4, edge_crease=0.5, vertex_crease=0.4)
    mortar_noise = noise_texture(noise_dimensions=NoiseTexture.NoiseDimensions._4D, w=random_value(data_type=RandomValue.DataType.FLOAT, min=0.0, max=100.000, seed=dist_seed), scale=9.000, detail=6.000, roughness=5.000, distortion=2.000)
    powered_mortar_noise = math(operation=Math.Operation.POWER, value=(mortar_noise.color, 8.000))
    offset_vector = vector_math(operation=VectorMath.Operation.MULTIPLY, vector=(normal(), powered_mortar_noise))
    positioned_geometry = set_position(geometry=subdivided_geometry, offset=offset_vector)
    full_brick_geometry = mesh_boolean(operation=MeshBoolean.Operation.INTERSECT, mesh_1=geometry, mesh_2=[geometry, positioned_geometry], self_intersection=True).mesh
    brick_material = bpy.data.materials.get("Brick")
    brick_wm = set_material(geometry=full_brick_geometry, material=brick_material)
    return brick_wm

@tree("ModifyTopMortar")
def modify_brick(geometry: Geometry):
    subdivided_mesh = subdivide_mesh(mesh=geometry, level=6)
    top_mortar_noise = noise_texture(noise_dimensions=NoiseTexture.NoiseDimensions._3D, scale=0.4, detail=0.1, roughness=0.5, distortion=0.3)
    top_mortar_points = distribute_points_on_faces(distribute_method=DistributePointsOnFaces.DistributeMethod.POISSON, mesh=subdivided_mesh, density_max=15000, density_factor=top_mortar_noise.fac, seed=38)
    top_mortar_volume = points_to_volume(resolution_mode=PointsToVolume.ResolutionMode.VOXEL_AMOUNT, points=top_mortar_points.points, voxel_amount=64.000, radius=0.015)
    top_mortar_shaded = set_shade_smooth(geometry=volume_to_mesh(resolution_mode=VolumeToMesh.ResolutionMode.VOXEL_AMOUNT, volume=top_mortar_volume, voxel_amount=64.000, threshold=0.100))
    top_mortar_scaled = scale_elements(domain=ScaleElements.Domain.FACE, scale_mode=ScaleElements.ScaleMode.SINGLE_AXIS, geometry=top_mortar_shaded, scale=0.7, axis=(0, 0, 1))

    # More distortion
    m_noise = noise_texture(noise_dimensions=NoiseTexture.NoiseDimensions._3D, scale=0.5, detail=20.000, roughness=1.000, distortion=80.000).color
    distorted_top_mortar = set_position(geometry=top_mortar_scaled, offset=vector_math(operation=VectorMath.Operation.MULTIPLY, vector=(normal(), math(operation=Math.Operation.POWER, value=(m_noise, 10.000)))))

    return distorted_top_mortar

################################################################
# Blender creating objects and applying modifiers
################################################################

begin = time.time()
brick_layers = random.randint(1, 4)

# Create the brick cube
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.025))
cube_obj = bpy.context.active_object
cube_obj.name = "Brick"
bx, by, bz = 0.237, 0.115, 0.05
cube_obj.scale = (bx, by, bz)
bpy.ops.object.transform_apply(location=True, rotation=False, scale=True)
resample_modifier = cube_obj.modifiers.new(name="Remesh", type='REMESH')
resample_modifier.octree_depth = 4 
resample_modifier.mode = 'SHARP'

# Create the top mortar cube
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.05))
mortar_obj = bpy.context.active_object
mortar_obj.name = "TopMortar"
mortar_obj.scale = (0.2, 0.1, 0.02)
bpy.ops.object.transform_apply(location=True, rotation=False, scale=True)
bevel_modifier = mortar_obj.modifiers.new(name="Bevel", type='BEVEL')
bevel_modifier.width = 0.2
bevel_modifier.segments = 5
bevel_modifier.angle_limit = 1.0472
geo_nodes_modifier = mortar_obj.modifiers.new(name="ModifyTopMortar", type='NODES')
replacement = bpy.data.node_groups["ModifyTopMortar"]
geo_nodes_modifier.node_group = replacement

# Create a plane
bpy.ops.mesh.primitive_plane_add()
plane_obj = bpy.context.active_object

# Add geometry tree to plane
geo_nodes_modifier = plane_obj.modifiers.new(name="Brickwall", type='NODES')
replacement = bpy.data.node_groups["Brickwall"]
geo_nodes_modifier.node_group = replacement
geo_nodes_modifier["Input_1"] = cube_obj
geo_nodes_modifier["Input_3"] = brick_layers

# Realize the instances from the plane and move them to a collection
bpy.ops.object.duplicates_make_real()
bpy.ops.object.move_to_collection(collection_index=0, is_new=True, new_collection_name="all_instances")

# Remove the geo node modifier from all the instanced plane objects
collection = bpy.data.collections.get("all_instances")
bpy.ops.object.select_all(action='DESELECT')
for p in collection.objects:
    p.select_set(True)
ref_object = collection.objects[0]
bpy.ops.object.modifier_remove(modifier="Brickwall")
bpy.ops.object.make_links_data(type='MODIFIERS')

# Link the data from the brick object to instanced plane objects
source_object = bpy.data.objects["Brick"]
source_object.select_set(True)
bpy.context.view_layer.objects.active = source_object
bpy.ops.object.make_links_data(type='OBDATA')

# First hide and then add the brick object to bproc
b_proc_obj = bproc.object.convert_to_meshes(blender_objects=[source_object])[0]
b_proc_obj.hide()
category_id = 0
b_proc_obj.set_cp('category_id', category_id)
b_proc_obj.set_cp("category_name", catId2name[category_id])

# Add geometry tree to the main plane to make mortar
geo_nodes_modifier = plane_obj.modifiers.new(name="Brickwall", type='NODES')
replacement = bpy.data.node_groups["Brickwall"]
geo_nodes_modifier.node_group = replacement
geo_nodes_modifier["Input_2"] = mortar_obj
geo_nodes_modifier["Input_3"] = brick_layers
plane_proc_obj = bproc.object.convert_to_meshes(blender_objects=[plane_obj])[0]
### can remove mortar from segmentation?
category_id = 0
plane_proc_obj.set_cp('category_id', category_id)
plane_proc_obj.set_cp("category_name", catId2name[category_id])

# Add geometry tree to instanced planes to modify them
target_objects = []
top_objects = []
highest_z = float('-inf')
for count, p in enumerate(collection.objects):
    dist_seed = random.randint(1, 100)
    geo_nodes_modifier = p.modifiers.new(name="ModifyBrick", type='NODES')
    replacement = bpy.data.node_groups["ModifyBrick"]
    geo_nodes_modifier.node_group = replacement
    geo_nodes_modifier["Input_8"] = dist_seed
    obj = bproc.object.convert_to_meshes(blender_objects=[p])[0]
    category_id = 1
    obj.set_cp("category_id", category_id) 
    obj.set_cp("category_name", catId2name[category_id])
    target_objects.append(obj)
    #Add the highest bricks to a list
    z_coordinate = p.matrix_world.translation.z
    if z_coordinate > highest_z:
        highest_z = z_coordinate
        top_objects = []
        top_objects.append(obj)
    elif z_coordinate == highest_z:
        top_objects.append(obj)

# First hide and then add the top mortar object to bproc
tm_proc_obj = bproc.object.convert_to_meshes(blender_objects=[bpy.data.objects["TopMortar"]])[0]
tm_proc_obj.hide()
category_id = 0
tm_proc_obj.set_cp('category_id', category_id)
tm_proc_obj.set_cp("category_name", catId2name[category_id])

################################################################
# Rendering with blenderproc
################################################################

# set shading and physics properties and randomize PBR materials
for j, obj in enumerate(target_objects):
    mass, fiction_coeff = (0.4, 0.5)
    obj.enable_rigidbody(True, mass=mass, friction=mass * fiction_coeff, 
    linear_damping = 1.99, angular_damping = 0, collision_margin=0.0001)
    obj.set_shading_mode('auto')

# create room
room = [bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),
        bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.570796, 0, 0]),
        bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 2, 2], rotation=[1.570796, 0, 0]),
        bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -1.570796, 0]),
        bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.570796, 0])]

# Create the ambient light
light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
light_plane.set_name('light_plane')
light_plane_material = bproc.material.create('light_material')
light_plane_material.make_emissive(emission_strength=np.random.uniform(3,8), 
                                emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]))    
light_plane.replace_materials(light_plane_material)

print(cfg['cc_textures_path'])
# sample CC Texture and assign to room planes
if cfg['cc_textures_path'] is not None:
    cc_textures = bproc.loader.load_ccmaterials(cfg['cc_textures_path'])
    print("Texture list length is {}".format(len(cc_textures)))
    for plane in room:
        random_cc_texture = np.random.choice(cc_textures)
        plane.replace_materials(random_cc_texture)

# set attributes
room.append(light_plane)
for plane in room:
    plane.enable_rigidbody(False, collision_shape='BOX', friction = 100, linear_damping = 0.0, angular_damping = 0.0)
    plane.set_cp('category_id', 0)
    plane.set_cp("category_name", 'Unknown')

# For spotlight sample point light on shell
light_point = bproc.types.Light()
light_point.set_energy(np.random.uniform(0, 100))
# lights = [[0,0,1], [1,0,0], [0,1,0]] debug only
light_point.set_color(np.random.uniform([0.5,0.5,0.5],[1,1,1]))
location = bproc.sampler.shell(center = [0, 0, 0], radius_min = 1, radius_max = 1.5,
                        elevation_min = 5, elevation_max = 89, uniform_volume = False)
light_point.set_location(location)

print("=================== Spent {:.3f} seconds for creating the geometries... =================== ".format(time.time() - begin))

# print("Amount of top objects are {}, target are {}".format(len(top_objects), len(target_objects)))
# Spawn object one by one with collision check
begin = time.time()

# BVH tree used for camera obstacle checks
bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(target_objects)

# setup camera
i = 0
radius_min, radius_max = (0.4, 0.8)


#!!!!!!!!!!!!!!! Change the N_row depending on if there are bricks on top or not


N_row = brick_layers * 2 + 1 # +1 if there are some bricks on top
print("{} layers of brick.".format(N_row))

while i < num_views:

    # Sample location
    inplane_rot = np.random.uniform(-0.35, 0.35) 
    radius = np.random.uniform(low=radius_min, high=radius_max) 
    dist_above_center = 0.0
    location = bproc.sampler.part_sphere([0, 0, N_row * bz], radius=radius, dist_above_center=dist_above_center, mode="SURFACE")
    
    poi = bproc.object.compute_poi(np.random.choice(top_objects, size=4))

    # Determine point of interest in scene as the object closest to the mean of a subset of objects
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi-location, inplane_rot=inplane_rot)

    # Add homog cam pose based on location an rotation
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)

    # Check that obstacles are at least 0.3 meter away from the camera and make sure the view interesting enough
    if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.3}, bop_bvh_tree):
        # If camera location is not at top.
        # print(location[2], (N_row * bz + radius - 0.1))
        if location[2] < (N_row * bz + radius - 0.1):
            # Persist camera pose
            bproc.camera.add_camera_pose(cam2world_matrix)
            i += 1
            # print("Center at {}, Radius is {}, location of camera is {}".format(N_row * bz, radius, location))
        else:
            print("One cam pose avoided.")

# render the whole pipeline
bproc.renderer.enable_depth_output(activate_antialiasing=False)
# bproc.renderer.enable_normals_output()
bproc.renderer.set_max_amount_of_samples(50)
bproc.renderer.enable_segmentation_output(map_by=["instance", "category_id",])

data = bproc.renderer.render()

#################!!!!!!!!!!!!!! This part for no mortar masks!!!!!!!!!!!

# data = bproc.renderer.render()
# bproc.object.delete_multiple(entities=[plane_proc_obj])

# bproc.renderer.enable_segmentation_output(map_by=["instance", "category_id",])
# data_seg = bproc.renderer.render()
###################


# postprocess depth using the kinect azure noise model
data["depth_kinect"] = bproc.postprocessing.add_kinect_azure_noise(data["depth"], data["colors"], 10)

print("=================== Spent {:.3f} seconds for rendering... =================== ".format(time.time() - begin))

write_my(out_dir,   
        dataset='',
        target_objects=target_objects,
        depths = data["depth"],
        depths_noise = data["depth_kinect"],
        colors = data["colors"], 
        instance_masks=data['instance_segmaps'],
        category_masks=data['category_id_segmaps'],
        instance_attribute_maps=data["instance_attribute_maps"],
        color_file_format = "JPEG",
        ignore_dist_thres = 10,
        frames_per_chunk=1000,
        is_shapenet=False)