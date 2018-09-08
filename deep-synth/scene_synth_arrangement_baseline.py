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

class SceneSynthArrangementBaseline(SceneSynth):

    def __init__(self, *args, **kwargs):
        super(SceneSynthArrangementBaseline, self).__init__(*args, **kwargs)
    
    def synth_room(self, room_id, trial, size, samples, \
                   save_dir, temperature_cat, temperature_pixel, \
                   min_p, max_collision, save=True, save_heatmap=False):
        room = SynthedRoomArrangementBaseline(room_id, trial, size, samples, self, temperature_cat, temperature_pixel, min_p, max_collision)

        while room.should_continue():
            room.save_top_down_view(save_dir)
            room.save_json(save_dir)
            room.add_node()
        room.save_top_down_view(save_dir, final=True)
        room.save_json(save_dir, final=True)


class SynthedRoomArrangementBaseline(SynthedRoom):
    
    def __init__(self, room_id, trial, size, samples, synthesizer, temperature_cat, temperature_pixel, min_p, max_collision):
        self.__dict__.update(locals())
        del self.self #Of course I don't care about readability

        self.scene = RenderedScene(index = room_id, \
                                   data_dir = synthesizer.data_dir_relative, \
                                   data_root_dir = synthesizer.data_root_dir, \
                                   load_objects = True)

        self._parse_original_nodes(self.scene.object_nodes)
        self.scene.object_nodes = []

        self.composite = self.scene.create_composite()
        self.door_window_nodes = self.scene.door_window_nodes
        self.object_nodes = []
        self.empty_house_json = None
        self.failures = 0
        self.obj_data = ObjectData()

    def _parse_original_nodes(self, nodes):
        category_dict = ObjectCategories()
        self.category_counts = np.zeros(len(self.synthesizer.categories))
        self.remaining_models = [[] for i in range(len(self.synthesizer.categories))]
        for node in nodes:
            modelId = node["modelId"]
            category = node["category"]
            self.category_counts[category] += 1
            self.remaining_models[category].append(modelId)
        #print(self.category_counts)
        #print(self.remaining_models)
            #print(modelId, category, category2, self.synthesizer.categories[category2])
    
    def should_continue(self):
        return self.category_counts.sum() > 0
    
    def add_node(self):
        print(self.category_counts)
        print(self.remaining_models)
        self.location_category_map = None
        self.current_room = self.composite.get_composite()

        self.existing_collisions = self._get_collisions()

        category_counts = self.category_counts
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
        self.remaining_models[best_category].remove(best_modelId)
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
        location_category_map = super(SynthedRoomArrangementBaseline, self)._create_location_category_map()
        
        for k in range(self.synthesizer.num_categories):
            if category_counts[k] == 0:
                location_category_map[k] = 0

        return location_category_map

    def _sample_model_rotation(self, x, y, category, trials=10, num_orientations=16):
        num_categories = self.synthesizer.num_categories
        synthesizer = self.synthesizer
        
        best_p = -100
        best_orientation = 0
        best_model = None

        models = list(set(self.remaining_models[category]))
            
        #print("Trying different models and orientation...")
        for (trial, modelId) in enumerate(models):
            inputs = []
            for orientation in range(num_orientations):
                r = math.pi*orientation/num_orientations*2
                new_obj = SynthNode(modelId, category, x, y, r, self).get_render()
                new_room = self.composite.add_and_get_composite \
                                (new_obj, category, math.sin(r), math.cos(r))
                new_obj[new_obj>0] = 1
                new_room[-1] = new_obj

                inputs.append(new_room)
            with torch.no_grad():
                inputs = torch.stack(inputs)
                inputs = inputs.cuda()
                outputs = synthesizer.model_rotation(inputs)
                outputs = synthesizer.softmax(outputs).cpu().data.numpy()

            for orientation in range(num_orientations):
                #print(orientation, outputs[orientation][1])
                if outputs[orientation][1] > best_p:
                    p = outputs[orientation][1]
                    r = math.pi*best_orientation/num_orientations*2
                    new_node = SynthNode(modelId, category, x, y, r, self)
                    collisions = self._get_collisions([new_node])
                    if (len(collisions) - len(self.existing_collisions)) > 0:
                        p -= 1
                    if p > best_p:
                        best_p = p
                        best_model = modelId
                        best_orientation = orientation

            print(f"Testing model {trial+1} of {len(models)}...", end = "\r")
        print()

        return best_model, math.pi*best_orientation/num_orientations*2, best_p
