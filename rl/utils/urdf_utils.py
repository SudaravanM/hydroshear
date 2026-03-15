import os
from yourdfpy import URDF

def extract_obj_path_from_urdf(urdf_path):
    urdf = URDF.load(urdf_path)
    # Iterate through all links
    for link in urdf.link_map.values():
        if hasattr(link, "visuals"):
            for visual in link.visuals:
                geometry = visual.geometry
                if geometry.mesh is not None:
                    return geometry.mesh.filename

def resolve_obj_path(urdf_path, obj_path):
    obj_path = os.path.abspath(os.path.join(os.path.dirname(urdf_path), obj_path))
    return obj_path