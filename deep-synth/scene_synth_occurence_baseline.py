from data import *
import random
import scipy.misc as m
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from models import *
from models.nade import *
from torch.autograd import Variable
from PIL import Image
import copy
from model_prior import *
from scene_synth import *
from categoryCounts_dataset import CategoryCountsDataset

class SceneSynthOccurenceBaseline(SceneSynth):

    def __init__(self, counts_epoch, train_size, *args, **kwargs):
        super(SceneSynthOccurenceBaseline, self).__init__(*args, **kwargs)
        self.model_counts = self._load_category_counts_model(counts_epoch, train_size)
    
    def _load_category_counts_model(self, epoch, train_size):
        counts_dir = f"{self.model_dir}/categoryCounts_epoch_{epoch}.pt"
        dataset = CategoryCountsDataset(
            data_root_dir = self.data_root_dir,
            data_dir = self.data_dir_relative,
            scene_indices = (0, train_size),
        )

        data_size = dataset.data_size
        data_domain_sizes = dataset.data_domain_sizes

        model_counts = DiscreteNADEModule(
            data_size = data_size,
            data_domain_sizes = data_domain_sizes,
            hidden_size = data_size
        )
        model_counts.load_state_dict(torch.load(counts_dir))
        model_counts.eval()
        model_counts.cuda()
        return model_counts

    def synth_room(self, room_id, trial, size, samples, \
                   save_dir, temperature_cat, temperature_pixel, \
                   min_p, max_collision):
        room = SynthedRoomOccurenceBaseline(room_id, trial, size, samples, self, temperature_cat, temperature_pixel, min_p, max_collision)
        category_counts = self.model_counts.sample().cpu().data.numpy()[0]
        print(category_counts)
        print(category_counts.sum())
        
        #For nodes in sampled_nodes....
        for i in range(category_counts.sum()):
            room.save_top_down_view(save_dir)
            room.save_json(save_dir)
            print(category_counts)
            room.add_node(category_counts)
        room.save_top_down_view(save_dir, final=True)
        room.save_json(save_dir, final=True)


class SynthedRoomOccurenceBaseline(SynthedRoom):
    
    def __init__(self, *args, **kwargs):
        super(SynthedRoomOccurenceBaseline, self).__init__(*args, **kwargs)
    
    def add_node(self, category_counts):
        self.location_category_map = None
        self.current_room = self.composite.get_composite()

        self.existing_collisions = self._get_collisions()
        
        self.count = 0
        best_x, best_y, best_p, best_r, best_modelId = None, None, -100, None, None
        best_category = None
        while True:
            self.count += 1
            gridx,gridy,category = self._sample_location_category(category_counts)
            
            print(f"Choosing type {self._get_category_name(category)} at grid {gridx}, {gridy}")

            x,y = self._sample_exact_location(gridx, gridy, category)

            #print(f"Try placing an object at image space coordinate {x}, {y}")
                
            modelId, r, p = self._sample_model_rotation(x, y, category)
            print(p)
            if p > best_p:
                best_p = p
                best_x = x
                best_y = y
                best_modelId = modelId
                best_r = r
                best_category = category
            if p > self.min_p or self.count > 10:
                print(f"Choosing model {modelId} rotated by {r} radians")
                if self.count > 10:
                    self.failures += 1000 #bad bad
                break
            else:
                self.failures += 1
                print(f"Best probability is {p}, resample location-category")

        category_counts[best_category] -= 1
        new_obj = SynthNode(best_modelId, best_category, best_x, best_y, best_r, self)
        self.composite.add_height_map(new_obj.get_render(), best_category, math.sin(best_r), math.cos(best_r))
        self.object_nodes.append(new_obj)


    def _sample_location_category(self, category_counts):
        if self.location_category_map is None:
            self.location_category_map = self._create_location_category_map(category_counts)

        total_p = self.location_category_map.sum()
        max_p = np.max(self.location_category_map)
        
        x,y,category = self._sample_location_category_helper(self.location_category_map, total_p)
        self.location_category_map[category][x][y] = 0

        return x,y,category 
    
    def _create_location_category_map(self, category_counts):
        location_category_map = super(SynthedRoomOccurenceBaseline, self)._create_location_category_map()
        
        for k in range(self.synthesizer.num_categories):
            if category_counts[k] == 0:
                location_category_map[k] = 0

        return location_category_map

