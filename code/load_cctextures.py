import blenderproc as bproc
import os

# Initialize BlenderProc
bproc.init()

# Create a simple plane
plane = bproc.object.create_primitive('PLANE', scale=[2, 2, 1])
plane.set_location([0, 0, 0])

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Replace 'ground_asphalt_03' with any available asset name from ambientcg.com
texture_dir = bproc.loader.load_cc_textures(
    cc_textures=[
        {
            "name": "ground_asphalt_03",  # <- change to any texture you want
            "type": "surface"             # usually 'surface' or 'imperfection'
        }
    ],
    data_dir=os.path.join(base_dir, 'resources') # local cache dir; will be created if it doesn't exist
)

# Get the material from the loaded texture
material = bproc.material.create_material_from_cc_texture(os.path.join(texture_dir, "ground_asphalt_03"))

# Assign the material to the plane
plane.set_material(material)

# Set up the camera
bproc.camera.add_camera_pose(bproc.math.build_transformation_mat([3, -3, 3], [1.1, 0, 0.8]))
bproc.camera.set_resolution(512, 512)

# Set up lighting
bproc.light.add_point_light(location=[2, -2, 4], energy=100)

# Render
bproc.renderer.set_output_dir("./output")
data = bproc.renderer.render()
bproc.writer.write_hdf5("./output/scene.hdf5", data)
