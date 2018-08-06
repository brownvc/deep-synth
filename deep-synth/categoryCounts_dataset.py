from torch.utils import data
from data import RenderedScene
import numpy as np
import os.path
import pickle


class CategoryCountsDataset():
    """
    Dataset for training the baseline NADE model for predicting object category counts
    """
    def __init__(self, data_root_dir, data_dir, scene_indices):
        """
        data_root_dir: root dir where all data lives
        data_dir: directory where this dataset lives (relative to data_root_dir)
        scene_indices: list of indices of scenes (in data_dir) that are considered part of this set
        """
        self.data_root_dir = data_root_dir
        self.data_dir = data_dir
        self.scene_indices = scene_indices
        self.categories = self.get_scene(0).categories
        self.data_size = len(self.categories)
        self.data_domain_sizes = self.get_domain_sizes()

    def __len__(self):
        return self.scene_indices[1]-self.scene_indices[0]

    def get_scene(self, index):
        return RenderedScene(index, self.data_dir, self.data_root_dir, shuffle=False)

    def __getitem__(self, index):
        i = index+self.scene_indices[0]
        scene = self.get_scene(i)
        # Note: scene.categories is already ordered by decreasing frequency
        cat_counts = np.zeros(self.data_size).astype(int)
        for node in scene.object_nodes:
            cat_index = node['category']
            cat_counts[cat_index] += 1
        return cat_counts

    def get_domain_sizes(self):
        """
        Search for a file containing domain sizes. If we don't find it, then compute it
        """
        domain_size_filename = f"{self.data_root_dir}/{self.data_dir}/domain_sizes.pkl"
        if not os.path.exists(domain_size_filename):
            domain_sizes = self.compute_domain_sizes()
            pkl_file = open(domain_size_filename, 'wb')
            pickle.dump(domain_sizes, pkl_file)
            pkl_file.close()
        else:
            pkl_file = open(domain_size_filename, 'rb')
            domain_sizes = pickle.load(pkl_file)
            pkl_file.close()

        # Convert from numpy.int64 array into a listof ints
        domain_sizes = domain_sizes.tolist()
        # Each entry of this list is the max number of instances of a category that appear 
        #    in any scene. The domain size is this value + 1 (to account for the possibility
        #    of having zero instances of that category)
        return [count + 1 for count in domain_sizes]

    def compute_domain_sizes(self):
        """
        Sweep through all the scenes once and compute the maximum number of times each category
           occurs in any single scene.
        """
        print('Computing domain sizes...')
        domain_sizes = np.zeros(self.data_size).astype(int)
        for i in range(0, len(self)):
            cat_counts = self[i]
            print(cat_counts)
            domain_sizes = np.maximum(domain_sizes, cat_counts)
        return domain_sizes
