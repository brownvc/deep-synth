from torch.utils import data
from data import ObjectCategories, RenderedScene, RenderedComposite
import random
import math
import torch

import _pickle as pickle

class ShouldContinueDataset():
    """
    Dataset for training/testing the "should continue" network
    """
    def __init__(self, data_root_dir, data_dir, scene_indices=(0,4000), num_per_epoch=1, complete_prob=0.5, seed=None, ablation=None):
        """
        Parameters
        ----------
        data_root_dir (String): root dir where all data lives
        data_dir (String): directory where this dataset lives (relative to data_root_dir)
        scene_indices (tuple[int, int]): list of indices of scenes (in data_dir) that are considered part of this set
        num_per_epoch (int): number of random variants of each scene that will be used per training epoch
        complete_prob (float): probability of sampling the complete scene, as opposed to some incomplete variant of it
        """
        self.data_root_dir = data_root_dir
        #self.data_dir = data_root_dir + '/' + data_dir
        self.data_dir = data_dir
        self.scene_indices = scene_indices
        self.num_per_epoch = num_per_epoch
        self.complete_prob = complete_prob

        # Load up the map between SUNCG model IDs and category names
        #self.category_map = ObjectCategories(data_root_dir + '/suncg_data/ModelCategoryMapping.csv')
        # Also load up the list of coarse categories used in this particular dataset
        #self.categories = self.get_coarse_categories()
        # Build a reverse map from category to index
        #self.cat_to_index = {self.categories[i]:i for i in range(len(self.categories))}
        self.seed = seed
        self.ablation = ablation

    def __len__(self):
        return (self.scene_indices[1]-self.scene_indices[0]) * self.num_per_epoch

    def __getitem__(self, index):
        if self.seed:
            random.seed(self.seed)

        i = int(index+self.scene_indices[0] / self.num_per_epoch)
        scene = RenderedScene(i, self.data_dir, self.data_root_dir)
        composite = scene.create_composite()

        num_categories = len(scene.categories)
        existing_categories = torch.zeros(num_categories)
        # Flip a coin for whether we're going remove objects or treat this as a complete scene
        is_complete = random.random() < self.complete_prob
        if not is_complete:
            # If we decide to remove objects, then remove a random number of them
            num_objects = random.randint(0, len(scene.object_nodes) - 1)
        else:
            num_objects = len(scene.object_nodes)

        for i in range(num_objects):
            node = scene.object_nodes[i]
            composite.add_node(node)
            existing_categories[node["category"]] += 1

        inputs = composite.get_composite(num_extra_channels=0, ablation=self.ablation)
        # Output is a boolean for "should we continue adding objects?"
        output = not is_complete
        return inputs, output, existing_categories
