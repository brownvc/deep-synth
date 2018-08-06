import pickle
import os
import numpy as np
from data import ObjectData
import utils

"""
Taking care of wavefront obj files
Convert to pickle for faster loading
Currently just geometric information.
Call this file once to create a pickled version of the objects
For faster loading in the future
"""

class Obj():
    """
    Standard vertex-face representation, triangulated
    Order: x, z, y
    """
    def __init__(self, modelId, houseId=None, from_source=False, is_room=False, mirror=False):
        """
        Parameters
        ----------
        modelId (string): name of the object to be loaded
        houseId (string, optional): If loading a room, specify which house does the room belong to
        from_source (bool, optional): If false, loads the pickled version of the object
            need to call object.py once to create the pickled version.
            does not apply for rooms
        mirror (bool, optional): If true, loads the mirroed version
        """
        if is_room: from_source = True  #Don't want to save rooms...
        data_dir = utils.get_data_root_dir()
        self.vertices = []
        self.faces = []
        if from_source:
            if is_room:
                path = f"{data_dir}/suncg_data/room/{houseId}/{modelId}.obj"
            else:
                path = f"{data_dir}/suncg_data/object/{modelId}/{modelId}.obj"
            with open(path,"r") as f:
                for line in f:
                    data = line.split()
                    if len(data) > 0:   
                        if data[0] == "v":
                            v = np.asarray([float(i) for i in data[1:4]]+[1])
                            self.vertices.append(v)
                        if data[0] == "f":
                            face = [int(i.split("/")[0])-1 for i in data[1:]]
                            if len(face) == 4:
                                self.faces.append([face[0],face[1],face[2]])
                                self.faces.append([face[0],face[2],face[3]])
                            elif len(face) == 3:
                                self.faces.append([face[0],face[1],face[2]])
                            else:
                                print(f"Found a face with {len(face)} edges!!!")

            self.vertices = np.asarray(self.vertices)
            data = ObjectData()
            if not is_room and data.get_alignment_matrix(modelId) is not None:
                self.transform(data.get_alignment_matrix(modelId))
        else:
            with open(f"{data_dir}/object/{modelId}/vertices.pkl", "rb") as f:
                self.vertices = pickle.load(f)
            with open(f"{data_dir}/object/{modelId}/faces.pkl", "rb") as f:
                self.faces = pickle.load(f)
        

        if mirror:
            t = np.asarray([[-1, 0, 0, 0], \
                            [0, 1, 0, 0], \
                            [0, 0, 1, 0], \
                            [0, 0, 0, 1]])
            self.transform(t)
            self.modelId = modelId+"_mirror"
        else:
            self.modelId = modelId
                
    def save(self):
        data_dir = utils.get_data_root_dir()
        dest_dir = f"{data_dir}/object/{self.modelId}"
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        with open(f"{dest_dir}/vertices.pkl", "wb") as f:
            pickle.dump(self.vertices, f, pickle.HIGHEST_PROTOCOL)
        with open(f"{dest_dir}/faces.pkl", "wb") as f:
            pickle.dump(self.faces, f, pickle.HIGHEST_PROTOCOL)
                
    
    def transform(self, t):
        self.vertices = np.dot(self.vertices, t)
    
    def get_triangles(self):
        for face in self.faces:
            yield (self.vertices[face[0]][:3], \
                   self.vertices[face[1]][:3], \
                   self.vertices[face[2]][:3],)
    
    def xmax(self):
        return np.amax(self.vertices, axis = 0)[0]

    def xmin(self):
        return np.amin(self.vertices, axis = 0)[0]

    def ymax(self):
        return np.amax(self.vertices, axis = 0)[2]

    def ymin(self):
        return np.amin(self.vertices, axis = 0)[2]

    def zmax(self):
        return np.amax(self.vertices, axis = 0)[1]

    def zmin(self):
        return np.amin(self.vertices, axis = 0)[1]

def parse_objects():
    """
    parse .obj objects and save them to pickle files
    """
    data_dir = utils.get_data_root_dir()
    obj_dir = data_dir + "/suncg_data/object/"
    print("Parsing SUNCG object files...")
    l = len(os.listdir(obj_dir))
    for (i, modelId) in enumerate(os.listdir(obj_dir)):
        print(f"{i+1} of {l}...", end="\r")
        if not modelId in ["mgcube", ".DS_Store"]:
            o = Obj(modelId, from_source = True)
            o.save()
            o = Obj(modelId, from_source = True, mirror = True)
            o.save()
    print()

if __name__ == "__main__":
    parse_objects()



