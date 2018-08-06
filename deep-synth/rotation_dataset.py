from torch.utils import data
import torch
from PIL import Image
from torch.autograd import Variable
import numpy as np
import scipy.misc as m
import random
import math
import pickle
import os
from data import ObjectCategories, RenderedScene, RenderedComposite, Obj, TopDownView, ProjectionGenerator

class RotationDataset():
    """
    Dataset for training/testing the instance/orientation network
    """
    pgen = ProjectionGenerator()
    possible_models = None

    def __init__(self, data_dir, data_root_dir, scene_indices=(0,4000), seed=None, ablation=None):
        """
        Parameters
        ----------
        data_root_dir (String): root dir where all data lives
        data_dir (String): directory where this dataset lives (relative to data_root_dir)
        scene_indices (tuple[int, int]): list of indices of scenes (in data_dir) that are considered part of this set
        p_auxiliary (int): probability that a auxiliary category is chosen
            Note that since (existing centroid) is sparse, it is actually treated as non-auxiliary when sampling
        seed (int or None, optional): random seed, to replicate stuff, if set
        ablation (string or None, optional): see data.RenderedComposite.get_composite, and paper
        """
        self.category_map = ObjectCategories()
        self.seed = seed
        self.data_dir = data_dir
        self.data_root_dir = data_root_dir
        self.scene_indices = scene_indices
        self.ablation = ablation

    def __len__(self):
        return self.scene_indices[1]-self.scene_indices[0]
    
    def __getitem__(self,index):
        if self.seed:
            random.seed(self.seed)

        i = index+self.scene_indices[0]
        scene = RenderedScene(i, self.data_dir, self.data_root_dir)
        composite = scene.create_composite() #get empty composite

        object_nodes = scene.object_nodes
        #Since we need to at least rotate one object, this differs from location dataset slightly
        num_objects = random.randint(0, len(object_nodes)-1) 
        num_categories = len(scene.categories)

        for i in range(num_objects):
            node = object_nodes[i]
            composite.add_node(node)
        
        #Select the node we want to rotate
        node = object_nodes[num_objects]
        
        modelId = node["modelId"]
        #Just some made up distribution of different cases
        #Focusing on 180 degree, then 90, then others
        ran = random.uniform(0,1)
        if ran < 0.2:
            r = math.pi
            target = 0
        elif ran < 0.4:
            r = math.pi / 2 * random.randint(1,3)
            target = 0
        elif ran < 0.6:
            r = math.pi / 8 * random.randint(1,15)
            target = 0
        else:
            r = 0
            target = 1

        o = Obj(modelId)
        #Get the transformation matrix from object space to scene space
        t = RotationDataset.pgen.get_projection(scene.room).to_2d(np.asarray(node["transform"]).reshape(4,4))
        #Since centered already in object space, rotating the object in object space is the easier option
        sin, cos = math.sin(r), math.cos(r)
        t_rot = np.asarray([[cos, 0, -sin, 0], \
                            [0, 1, 0, 0], \
                            [sin, 0, cos, 0], \
                            [0, 0, 0, 1]])
        o.transform(np.dot(t_rot,t))
        #Render the rotated view of the object
        rotated = torch.from_numpy(TopDownView.render_object_full_size(o, composite.size))
        #Calculate the relevant info needed to composite it to the input
        sin, cos = composite.get_transformation(node["transform"])
        original_r = math.atan2(sin, cos)
        sin = math.sin(original_r + r)
        cos = math.cos(original_r + r)
        composite.add_height_map(rotated, node["category"], sin, cos)
        
        inputs = composite.get_composite(ablation=self.ablation)
        #Add attention channel, which is just the outline of the targeted object
        rotated[rotated>0] = 1
        inputs[-1] = rotated

        return inputs, target
