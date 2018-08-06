import csv
import os
import numpy as np
import utils

class ObjectCategories():
    """
    Determine which categories does each object belong to
    """
    def __init__(self):
        fname = "ModelCategoryMapping.csv"
        self.model_to_categories = {}

        root_dir = os.path.dirname(os.path.abspath(__file__))
        model_cat_file = f"{root_dir}/{fname}"

        with open(model_cat_file, "r") as f:
            categories = csv.reader(f)
            for l in categories:
                self.model_to_categories[l[1]] = [l[2],l[3],l[5]]

    def get_fine_category(self, model_id):
        model_id = model_id.replace("_mirror","")
        return self.model_to_categories[model_id][0]
    
    def get_coarse_category(self, model_id):
        model_id = model_id.replace("_mirror","")
        return self.model_to_categories[model_id][1]

    def get_final_category(self, model_id):
        """
        Final categories used in the generated dataset
        Minor tweaks from fine categories
        """
        model_id = model_id.replace("_mirror","")
        category = self.model_to_categories[model_id][0]
        if model_id == "199":
            category = "dressing_table_with_stool"
        if category == "nightstand":
            category = "stand"
        if category == "bookshelf":
            category = "shelving"
        return category

class ObjectData():
    """
    Various information associated with the objects
    """
    def __init__(self):
        self.model_to_data = {}

        root_dir = os.path.dirname(os.path.abspath(__file__))
        model_data_file = f"{root_dir}/Models.csv"

        with open(model_data_file, "r") as f:
            data = csv.reader(f)
            for l in data:
                if l[0] != 'id':  # skip header row
                    self.model_to_data[l[0]] = l[1:]

    def get_front(self, model_id):
        model_id = model_id.replace("_mirror","")
        # TODO compensate for mirror (can have effect if not axis-aligned in model space)
        return [float(a) for a in self.model_to_data[model_id][0].split(",")]

    def get_aligned_dims(self, model_id):
        """Return canonical alignment dimensions of model *in meters*"""
        model_id = model_id.replace('_mirror', '')  # NOTE dims don't change since mirroring is symmetric on yz plane
        return [float(a)/100.0 for a in self.model_to_data[model_id][4].split(',')]

    def get_model_semantic_frame_matrix(self, model_id):
        """Return canonical semantic frame matrix for model.
           Transforms from semantic frame [0,1]^3, [x,y,z] = [right,up,back] to raw model coordinates."""
        up = np.array([0, 1, 0])  # NOTE: up is assumed to always be +Y for SUNCG objects
        front = np.array(self.get_front(model_id))
        has_mirror = '_mirror' in model_id
        model_id = model_id.replace('_mirror', '')
        hdims = np.array(self.get_aligned_dims(model_id)) * 0.5
        p_min = np.array([float(a) for a in self.model_to_data[model_id][2].split(',')])
        p_max = np.array([float(a) for a in self.model_to_data[model_id][3].split(',')])
        if has_mirror:
            p_max[0] = -p_max[0]
            p_min[0] = -p_min[0]
        model_space_center = (p_max + p_min) * 0.5
        m = np.identity(4)
        m[:3, 0] = np.cross(front, up) * hdims[0]  # +x = right
        m[:3, 1] = np.array(up) * hdims[1]         # +y = up
        m[:3, 2] = -front * hdims[2]               # +z = back = -front
        m[:3, 3] = model_space_center              # origin = center
        # r = np.identity(3)
        # r[:3, 0] = np.cross(front, up)  # +x = right
        # r[:3, 1] = np.array(up)         # +y = up
        # r[:3, 2] = -front               # +z = back = -front
        # s = np.identity(3)
        # s[0, 0] = hdims[0]
        # s[1, 1] = hdims[1]
        # s[2, 2] = hdims[2]
        # sr = np.matmul(s, r)
        # m = np.identity(4)
        # m[:3, :3] = sr
        # m[:3, 3] = model_space_center
        return m

    def get_alignment_matrix(self, model_id):
        """
        Since some models in the dataset are not aligned in the way we want
        Generate matrix that realign them
        """
        #alignment happens BEFORE mirror, so make sure no mirrored 
        #object will ever call this!
        #model_id = model_id.replace("_mirror","")
        if self.get_front(model_id) == [0,0,1]:
            return None
        else:
            #Let's just do case by case enumeration!!!
            if model_id in ["106", "114", "142", "323", "333", "363", "364",
                            "s__1782", "s__1904"]:
                M = [[-1,0,0,0],
                     [0,1,0,0],
                     [0,0,-1,0],
                     [0,0,0,1]]
            elif model_id in ["s__1252", "s__400", "s__885"]:
                M = [[0,0,-1,0],
                     [0,1,0,0],
                     [1,0,0,0],
                     [0,0,0,1]]
            elif model_id in ["146", "190", "s__404", "s__406"]:
                M = [[0,0,1,0],
                     [0,1,0,0],
                     [-1,0,0,0],
                     [0,0,0,1]]
            else:
                print(model_id)
                raise NotImplementedError

            return np.asarray(M)
    
    def get_setIds(self, model_id):
        model_id = model_id.replace("_mirror","")
        return [a for a in self.model_to_data[model_id][8].split(",")]

