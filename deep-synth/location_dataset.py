from torch.utils import data
import torch
import random
import math
from data import ObjectCategories, RenderedScene, RenderedComposite

class LocationDataset():
    """
    Dataset for training/testing the location/category network
    """
    def __init__(self, data_dir, data_root_dir, scene_indices=(0,4000), p_auxiliary=0.7, seed=None, ablation=None):
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
        self.p_auxiliary = p_auxiliary
        self.ablation = ablation

    def __len__(self):
        return self.scene_indices[1]-self.scene_indices[0]

    def __getitem__(self,index):
        if self.seed:
            random.seed(self.seed)

        i = index+self.scene_indices[0]
        scene = RenderedScene(i, self.data_dir, self.data_root_dir)
        composite = scene.create_composite() #get empty composite
        
        #Select a subset of objects randomly. Number of objects is uniformly
        #distributed between [0, total_number_of_objects]
        #Doors and windows do not count here
        object_nodes = scene.object_nodes
        num_objects = random.randint(0, len(object_nodes))
        
        num_categories = len(scene.categories)
        OUTSIDE = num_categories + 2
        EXISTING = num_categories + 1
        NOTHING = num_categories

        centroids = []
        existing_categories = torch.zeros(num_categories)
        future_categories = torch.zeros(num_categories)
        #Process existing objects
        for i in range(num_objects):
            #Add object to composite
            node = object_nodes[i]
            composite.add_node(node)
            #Add existing centroids
            xmin, _, ymin, _ = node["bbox_min"]
            xmax, _, ymax, _ = node["bbox_max"]
            centroids.append(((xmin+xmax)/2, (ymin+ymax)/2, EXISTING))
            existing_categories[node["category"]] += 1
        
        inputs = composite.get_composite(ablation=self.ablation)
        size = inputs.shape[1]
        #Process removed objects
        for i in range(num_objects, len(object_nodes)):
            node = object_nodes[i]
            xmin, _, ymin, _ = node["bbox_min"]
            xmax, _, ymax, _ = node["bbox_max"]
            centroids.append(((xmin+xmax)/2, (ymin+ymax)/2, node["category"]))
            future_categories[node["category"]] += 1
        
        resample = True
        if random.uniform(0,1) > self.p_auxiliary: #Sample an object at random
            x,y,output_centroid = random.choice(centroids)
            x = int(x)
            y = int(y)
        else: #Or sample an auxiliary category
            while resample:
                x, y = random.randint(0,511), random.randint(0,511) #Should probably remove this hardcoded size at somepoint
                good = True
                for (xc, yc, _) in centroids: 
                    #We don't want to sample an empty space that's too close to a centroid
                    #That is, if it can fall within the ground truth attention mask of an object
                    if x-4 < xc < x+5 and y-4 < yc < y+5:
                        good = False 

                if good:
                    if not inputs[0][x][y]:
                        output_centroid = OUTSIDE #Outside of room
                        if random.uniform(0,1) > 0.8: #Just some hardcoded stuff simple outside room is learned very easily
                            resample = False
                    else:
                        output_centroid = NOTHING #Inside of room
                        resample = False

        #Attention mask
        xmin = max(x - 4, 0)
        xmax = min(x + 5, size)
        ymin = max(y - 4, 0)
        ymax = min(y + 5, size)
        inputs[-1, xmin:xmax, ymin:ymax] = 1 #Create attention mask
        
        #Compute weight for L_Global
        #If some objects in this cateogory are removed, weight is zero
        #Otherwise linearly scaled based on completeness of the room
        #See paper for details
        penalty = torch.zeros(num_categories)
        penalty[future_categories==0] = num_objects/len(object_nodes)

        return inputs, output_centroid, existing_categories, penalty
