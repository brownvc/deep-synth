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
from torch.autograd import Variable
from PIL import Image
import copy
from model_prior import *
from priors.observations import ObjectCollection
from utils import stdout_redirected

class SceneSynth():
    """
    Class that synthesizes scenes
    based on the trained models
    """
    def __init__(self, location_epoch, rotation_epoch, continue_epoch, \
                 data_dir, model_dir, data_root_dir=None, model_root_dir=None, size=512):
        """
        Parameters
        ----------
        location_epoch, rotation_epoch, continue_epoch (int):
            the epoch number of the respective trained models to be loaded
        data_dir (string): location of the dataset relative to data_root_dir
        model_dir (string): location of the trained models relative to model_root_dir
        data_root_dir (string or None, optional): if not set, use the default data location,
            see utils.get_data_root_dir
        model_root_dir (string or None, optional): if not set, use the directory of this script
        size (int): size of the input image
        """
        Node.warning = False
        self.data_dir_relative = data_dir #For use in RenderedScene
        if not data_root_dir:
            self.data_root_dir = utils.get_data_root_dir()
        self.data_dir = f"{self.data_root_dir}/{data_dir}"
        if not model_root_dir:
            model_root_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = f"{model_root_dir}/{model_dir}"
        
        #Loads category and model information
        self.categories, self.cat_to_index = self._load_category_map()
        self.num_categories = len(self.categories)
        self.possible_models = self._load_possible_models()
        self.model_set_list = self._load_model_set_list()
        
        #Loads trained models and build up NNs
        self.model_location, self.fc_location = self._load_location_model(location_epoch)
        self.model_rotation = self._load_rotation_model(rotation_epoch)
        self.model_continue, self.fc_continue = self._load_continue_model(continue_epoch)

        self.softmax = nn.Softmax(dim=1)
        self.softmax.cuda()
        
        #Misc Handling
        self.pgen = ProjectionGenerator()

        self.model_sampler = ModelPrior()
        self.model_sampler.load(self.data_dir)

        self.object_collection = ObjectCollection()
    
    def _load_category_map(self):
        with open(f"{self.data_dir}/final_categories_frequency", "r") as f:
            lines = f.readlines()
        cats = [line.split()[0] for line in lines]
        categories = [cat for cat in cats if cat not in set(['window', 'door'])]
        cat_to_index = {categories[i]:i for i in range(len(categories))}

        return categories, cat_to_index
    
    def _load_location_model(self, epoch):
        location_dir = f"{self.model_dir}/location_{epoch}.pt"
        model_location = resnet101(num_input_channels=self.num_categories+9, \
                                   use_fc=False)
        model_location.load_state_dict(torch.load(location_dir))
        model_location.eval()
        model_location.cuda()

        location_dir_fc = f"{self.model_dir}/location_fc_{epoch}.pt"
        fc_location = FullyConnected(2048+self.num_categories, self.num_categories+3)
        fc_location.load_state_dict(torch.load(location_dir_fc))
        fc_location.cuda()

        return model_location, fc_location

    def _load_rotation_model(self, epoch):
        rotation_dir = f"{self.model_dir}/rotation_{epoch}.pt"
        model_rotation = resnet101(num_classes=2, \
                                   num_input_channels=self.num_categories+9)
        model_rotation.load_state_dict(torch.load(rotation_dir))
        model_rotation.eval()
        model_rotation.cuda()

        return model_rotation

    def _load_continue_model(self, epoch):
        continue_dir = f"{self.model_dir}/continue_{epoch}.pt"

        model_continue = resnet101(num_input_channels=self.num_categories+8, \
                                   use_fc=False)
        model_continue.load_state_dict(torch.load(continue_dir))
        model_continue.eval()
        model_continue.cuda()

        continue_dir_fc = f"{self.model_dir}/continue_fc_{epoch}.pt"
        fc_continue = FullyConnected(2048+self.num_categories, 2)
        fc_continue.load_state_dict(torch.load(continue_dir_fc))
        fc_continue.cuda()
        fc_continue.eval()

        return model_continue, fc_continue

    def _load_possible_models(self, model_freq_threshold=0.01):
        #model_freq_threshold: discards models with frequency less than the threshold
        category_dict = ObjectCategories()
        possible_models = [[] for i in range(self.num_categories)]
        with open(f"{self.data_dir}/model_frequency") as f:
            models = f.readlines()

        models = [l[:-1].split(" ") for l in models]
        models = [(l[0], int(l[1])) for l in models]
        for model in models:
            category = category_dict.get_final_category(model[0])
            if not category in ["door", "window"]:
                possible_models[self.cat_to_index[category]].append(model)

        for i in range(self.num_categories):
            total_freq = sum([a[1] for a in possible_models[i]])
            possible_models[i] = [a[0] for a in possible_models[i] if a[1]/total_freq > model_freq_threshold]

        return possible_models
    
    def _load_model_set_list(self):
        possible_models = self.possible_models
        obj_data = ObjectData()
        model_set_list = [None for i in range(self.num_categories)]
        for category in range(self.num_categories):
            tmp_dict = {}
            for model in possible_models[category]:
                setIds = [a for a in obj_data.get_setIds(model) if a != '']
                for setId in setIds:
                    if setId in tmp_dict:
                        tmp_dict[setId].append(model)
                    else:
                        tmp_dict[setId] = [model]
            model_set_list[category] = \
                [value for key,value in tmp_dict.items() if len(value) > 1]
        
        return model_set_list

    def get_relevant_models(self, category, modelId):
        """
        Given a category and a modelId, return all models that are relevant to it
        Which is: the mirrored version of the model,
        plus all the models that belong to the same model set
        that appear more than model_freq_threshold (set to 0.01)
        See _load_possible_models and _load_model_set_list

        Parameters
        ----------
        category (int): category of the object
        modelId (String): modelId of the object

        Return
        ------
        set[String]: set of all relevant modelIds
        """
        relevant = set()
        if "_mirror" in modelId:
            mirrored = modelId.replace("_mirror", "")
        else:
            mirrored = modelId + "_mirror"
        if mirrored in self.possible_models[category]:
            relevant.add(mirrored)

        for model_set in self.model_set_list[category]:
            if modelId in model_set:
                relevant |= set(model_set)
        
        return relevant


    def synth(self, room_ids, trials=2, size=512, samples=32, save_dir=".", \
              temperature_cat=0.25, temperature_pixel=0.4, min_p=0.5, max_collision=-0.1):
        """
        Synthesizes the rooms!

        Parameters
        ----------
        room_ids (list[int]): indices of the room to be synthesized, loads their
            room arthicture, plus doors and windows, and synthesize the rest
        trials (int): number of layouts to synthesize per room
        size (int): size of the top-down image
        samples (int): size of the sample grid (for location and category)
        save_dir (str): location where the synthesized rooms are saved
        temperature_cat, temperature_pixel (float): temperature for tempering,
            refer to the paper for more details
        min_p (float): minimum probability where a model instance + orientation can be accepted
        max_collision (float): max number of collision penetration, in meters, that are allowed to occur
            This is not the only collision criteria, two more are hard coded, see SynthedRoom._get_collisions
        """
        for room_id in room_ids:
            for trial in range(trials):
                self.synth_room(room_id, trial, size, samples, save_dir, temperature_cat, temperature_pixel, min_p, max_collision)

    def synth_room(self, room_id, trial, size, samples, \
                   save_dir, temperature_cat, temperature_pixel, \
                   min_p, max_collision):
        """
        Synthesize a single room, see synth for explanation of some most paramters
        """
        room = SynthedRoom(room_id, trial, size, samples, self, temperature_cat, temperature_pixel, min_p, max_collision)

        while room.should_continue():
            room.save_top_down_view(save_dir)
            room.save_json(save_dir)
            room.add_node()
        room.save_top_down_view(save_dir, final=True)
        room.save_json(save_dir, final=True)

class SynthedRoom():
    """
    Class that synthesize a single room and keeps its record
    """

    def __init__(self, room_id, trial, size, samples, synthesizer, temperature_cat,
                 temperature_pixel, min_p, max_collision):
        """
        Refer to SceneSynth.synth for explanations for most parameters

        Parameters
        ----------
        synthesizer (SceneSynth): links back to SceneSynth so we can use the loaded models
        """
        self.__dict__.update(locals())
        del self.self #Of course I don't care about readability

        self.scene = RenderedScene(index = room_id, \
                                   data_dir = synthesizer.data_dir_relative, \
                                   data_root_dir = synthesizer.data_root_dir, \
                                   load_objects = False)

        self.composite = self.scene.create_composite()
        self.door_window_nodes = self.scene.door_window_nodes
        self.object_nodes = []
        self.empty_house_json = None
        self.failures = 0
        self.obj_data = ObjectData()

    def should_continue(self):
        """
        Stop either when continue predictor predicts stopping
        Or if more than 20 failed insertion has occured
        Or if no possible insertion can be found after 10 attempts
        at any step. (Possible means probability predicted by instance orientation
        network is greater than min_p, and there are no collisions)

        Return
        ------
        bool: should continue adding objects or not
        """
        synthesizer = self.synthesizer
        
        if self.failures >= 20:
            return False

        current_room = self.composite.get_composite(num_extra_channels=0)
        with torch.no_grad():
            inputs = current_room.unsqueeze(0).float()
            inputs = inputs.cuda()

            existing = Variable(self._get_existing_categories().unsqueeze(0).cuda())

            outputs = synthesizer.model_continue(inputs)
            outputs = torch.cat([outputs, existing], 1)
            outputs = synthesizer.fc_continue(outputs)

            outputs = synthesizer.softmax(outputs).cpu().numpy()
        print(f"Continue probability is {outputs[0][1]:.3f}")
        return outputs[0][1] >= 0.5 and len(self.object_nodes) < 20

    def add_node(self):
        """
        Add a new object into the current room
        """
        self.location_category_map = None
        self.current_room = self.composite.get_composite()
        #Get existing collisions, which should not be considered later
        self.existing_collisions = self._get_collisions()
        #Number of attempts
        self.count = 0
        #Info about best insertion so far
        best_x, best_y, best_p, best_r, best_modelId = None, None, -100, None, None
        best_category = None
        while True:
            self.count += 1
            gridx,gridy,category = self._sample_location_category()
            
            print(f"Choosing type {self._get_category_name(category)} at grid {gridx}, {gridy}")

            x,y = self._sample_exact_location(gridx, gridy, category)

            #print(f"Try placing an object at image space coordinate {x}, {y}")
                
            modelId, r, p = self._sample_model_rotation(x, y, category)
            #print(f"Insertion probability is {p}")
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
                if best_p < 0:
                    print(f"No collision free insertions found, resample location-category")
                else:
                    print(f"Best probability is {p}, resample location-category")

        new_obj = SynthNode(best_modelId, best_category, best_x, best_y, best_r, self)
        self.composite.add_height_map(new_obj.get_render(), best_category, math.sin(best_r), math.cos(best_r))
        self.object_nodes.append(new_obj)

    def _get_existing_categories(self):
        #Category count to be used by networks
        existing_categories = torch.zeros(self.synthesizer.num_categories)
        for node in self.object_nodes:
            existing_categories[node.category] += 1
        return existing_categories

    def _get_category_name(self, index):
        #Return name of the category, given index
        return self.synthesizer.categories[index]

    def _sample_location_category(self):
        #Creates location category map if it haven't been created
        #Otherwise just sample from the existing one
        if self.location_category_map is None:
            self.location_category_map = self._create_location_category_map()

        total_p = self.location_category_map.sum()
        max_p = np.max(self.location_category_map)
        
        x,y,category = self._sample_location_category_helper(self.location_category_map, total_p)
        #Clears sampled location so it does not get resampled again
        self.location_category_map[category][x][y] = 0

        return x,y,category 

    def _sample_location_category_helper(self, distribution, total):
        p = random.uniform(0,total)
        cumu = 0
        for i in range(self.samples):
            for j in range(self.samples):
                for category in range(self.synthesizer.num_categories):
                    cumu += distribution[category][i][j]
                    if cumu > p:
                        return (i,j,category)
        return (i,j,category)


    def _create_location_category_map(self):
        synthesizer = self.synthesizer
        num_categories = synthesizer.num_categories
        size = self.size
        samples = self.samples

        #print("Creating location category map...")

        location_category_map = np.zeros((num_categories, samples, samples))

        for i in range(samples):
            #Batching inputs by rows
            inputs = self.current_room.unsqueeze(0).repeat(self.samples,1,1,1)
            for j in range(samples):
                x = int(size/samples*(i+0.5))
                y = int(size/samples*(j+0.5))

                xmin = max(x-4,0)
                xmax = min(x+5,size)
                ymin = max(y-4,0)
                ymax = min(y+5,size)

                inputs[j,-1, xmin:xmax, ymin:ymax] = 1
            
            with torch.no_grad():
                inputs = inputs.cuda()

                existing = self._get_existing_categories().unsqueeze(0).repeat(samples,1)
                existing = Variable(existing.cuda())

                outputs = synthesizer.model_location(inputs)
                outputs = torch.cat([outputs, existing], 1)
                outputs = synthesizer.fc_location(outputs)

                outputs = outputs.cpu().numpy()

            for j in range(samples):
                #Very unoptimized but too lazy to change
                #Should really use a softmax here
                x = int(size/samples*(i+0.5))
                y = int(size/samples*(j+0.5))
                output = list(outputs[j])
                #print(output)
                output = [math.e ** o for o in output]
                sum_o = sum(output)
                if sum_o == 0 or self.current_room[0][x][y] == 0:
                    output = [0 for o in output]
                else:
                    output = [o / sum_o for o in output]
                for k in range(num_categories):
                    location_category_map[k][i][j] = output[k]

            #print(f"Batch {i+1} of {samples}...", end = "\r")
        
        #Temper by category (step 1)
        p_category = [location_category_map[k].sum() for k in range(num_categories)]
        for k in range(num_categories):
            location_category_map[k] = (location_category_map[k]**(1/self.temperature_cat))
            location_category_map[k] = location_category_map[k]/location_category_map[k].sum()*p_category[k]
        
        #Temper by location (step 2)
        location_map = location_category_map.sum(axis=0)
        location_map = location_map**(1/self.temperature_pixel)
        for i in range(samples):
            for j in range(samples):
                if location_category_map[:,i,j].sum() != 0:
                    location_category_map[:,i,j] /= location_category_map[:,i,j].sum()
                location_category_map[:,i,j] *= location_map[i,j]

        location_category_map = location_category_map / location_category_map.sum()

        return location_category_map
    
    def _sample_exact_location(self, gridx, gridy, category):
        #Picks the exact location given the coarse grid
        subsamples = int(self.size / self.samples)
        synthesizer = self.synthesizer
        num_categories = synthesizer.num_categories
        size = self.size

        subx = gridx * subsamples
        suby = gridy * subsamples
    
        exact_location_map = np.zeros((subsamples,subsamples))
        #print("Creating exact location map...")
        for i in range(subsamples):
            inputs = self.current_room.unsqueeze(0).repeat(subsamples,1,1,1)
            for j in range(subsamples):
                x = i + subx
                y = j + suby

                xmin = max(x-4,0)
                xmax = min(x+5,size)
                ymin = max(y-4,0)
                ymax = min(y+5,size)
                
                inputs[j,-1, xmin:xmax, ymin:ymax] = 1
            
            with torch.no_grad():
                inputs = inputs.cuda()

                existing = self._get_existing_categories().unsqueeze(0).repeat(subsamples,1)
                existing = Variable(existing.cuda())

                outputs = synthesizer.model_location(inputs)
                outputs = torch.cat([outputs, existing], 1)
                outputs = synthesizer.fc_location(outputs)
                outputs = outputs.cpu().numpy()

            for j in range(subsamples):
                output = list(outputs[j])
                #print(output)
                output = [math.e ** o for o in output]
                sum_o = sum(output)
                if sum_o == 0:
                    output = [0 for o in output]
                else:
                    output = [o / sum_o for o in output]
                exact_location_map[i][j] = output[category]

            #print(f"Batch {i+1} of {subsamples}...", end = "\r")
        #print()
        
        max_p = exact_location_map.max()
        i,j = list(np.where(exact_location_map==max_p))

        x,y = subx + i[0] + 0.5, suby + j[0] + 0.5

        return x,y

    def _sample_model_rotation(self, x, y, category, trials=10, num_orientations=16):
        num_categories = self.synthesizer.num_categories
        synthesizer = self.synthesizer
        
        best_p = -100
        best_orientation = 0
        best_model = None
        
        #Summarize existing objects to be fed into the
        #Super naive bigram model prior
        #Important objects are those with no more than 100 pixels apart
        #Or those belonging to the same category as the to be added category
        #see model_prior.ModelPrior.get_models
        important = []
        others = []
        for node in self.object_nodes:
            if node.category == category:
                important.append(node.modelId)
            elif ((node.x - x) ** 2 + (node.y - y) ** 2) < 10000:
                others.append(node.modelId)
        
        #models = self.synthesizer.model_sampler.get_models(category, [], [])
        #print(sorted(models))
        models = self.synthesizer.model_sampler.get_models(category, important, others) #Get possible models
        #print(sorted(models))
        set_augmented_models = set(models)
        for modelId in models: #Augment each of them using get_relevant_models
            set_augmented_models |= self.synthesizer.get_relevant_models(category, modelId)
        set_augmented_models = list(set_augmented_models)
        #print(sorted(set_augmented_models))
            
        #print("Trying different models and orientation...")
        for (trial, modelId) in enumerate(set_augmented_models):
            inputs = []
            #Batching possible orientations
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
                outputs = synthesizer.softmax(outputs).cpu().numpy()

            for orientation in range(num_orientations):
                if outputs[orientation][1] > best_p:
                    p = outputs[orientation][1]
                    r = math.pi*best_orientation/num_orientations*2
                    new_node = SynthNode(modelId, category, x, y, r, self)
                    collisions = self._get_collisions([new_node])
                    if (len(collisions) - len(self.existing_collisions)) > 0:
                        p -= 1 #Brute force way to enforce no collisions!
                    if p > best_p:
                        best_p = p
                        best_model = modelId
                        best_orientation = orientation

            print(f"Testing model {trial+1} of {len(set_augmented_models)}...", end = "\r")
        print()

        return best_model, math.pi*best_orientation/num_orientations*2, best_p
    
    def save_top_down_view(self, save_dir, final=False):
        """
        Save the top down view of the current room

        Parameters
        ----------
        save_dir (String): location to be saved
        final (bool, optional): If true, mark the saved one as finalized
            and include number of failures, to be processed by other scripts
        """
        current_room = self.composite.get_composite(num_extra_channels=0)
        img = m.toimage(current_room[3].numpy(), cmin=0, cmax=1)
        if final:
            img.save(f"{save_dir}/{self.room_id}_{self.trial}_{len(self.object_nodes)}_final_{self.failures}.png")
        else:
            img.save(f"{save_dir}/{self.room_id}_{self.trial}_{len(self.object_nodes)}.png")
    
    def _create_empty_house_json(self):
        #Preprocess a json containing only the empty ROOM (oops I named the method wrong)
        #Since this is constant across the entire synthesis process
        house_original = House(include_support_information=False, \
                               file_dir=f"{self.synthesizer.data_dir}/json/{self.room_id}.json")
        room_original = house_original.rooms[0] #Assume one room
        house = {}
        house["version"] = "suncg@1.0.0"
        house["id"] = house_original.id
        house["up"] = [0,1,0]
        house["front"] = [0,0,1]
        house["scaleToMeters"] = 1
        level = {}
        level["id"] = "0"
        room = {}
        room["id"] = "0_0"
        room["type"] = "Room"
        room["valid"] = 1
        room["modelId"] = room_original.modelId
        room["nodeIndices"] = []
        room["roomTypes"] = room_original.roomTypes
        room["bbox"] = room_original.bbox
        level["nodes"] = [room]
        house["levels"] = [level]
        count = 1
        for node in self.door_window_nodes:
            node_json = {}
            room["nodeIndices"].append(str(count))
            node_json["id"] = f"0_{count}"
            node_json["type"] = "Object"
            node_json["valid"] = 1
            modelId = node["modelId"]
            transform = node["transform"]
            if "mirror" in modelId:
                transform = np.asarray(transform).reshape(4,4)
                t_reflec = np.asarray([[-1, 0, 0, 0], \
                                  [0, 1, 0, 0], \
                                  [0, 0, 1, 0], \
                                  [0, 0, 0, 1]])
                transform = np.dot(t_reflec, transform)
                transform = list(transform.flatten())
                modelId = modelId.replace("_mirror","")
            node_json["transform"] = transform
            node_json["modelId"] = modelId

            level["nodes"].append(node_json)
            count += 1

        self.empty_house_json = house
        self.projection = self.synthesizer.pgen.get_projection(room_original)
        #Camera parameters, for orthographic render
        house["camera"] = {}
        ortho_param = self.projection.get_ortho_parameters()
        orthographic = {"left" : ortho_param[0],
                        "right" : ortho_param[1],
                        "bottom" : ortho_param[2],
                        "top" : ortho_param[3],
                        "far" : ortho_param[4],
                        "near" : ortho_param[5]}
        house["camera"]["orthographic"] = orthographic


    def save_json(self, save_dir, final=False):
        """
        Save the json file, see save_top_down_view
        """
        house = self.get_json()
        if final:
            with open(f"{save_dir}/{self.room_id}_{self.trial}_{len(self.object_nodes)}_final_{self.failures}.json", "w") as f:
                json.dump(house, f)
        else:
            with open(f"{save_dir}/{self.room_id}_{self.trial}_{len(self.object_nodes)}.json", "w") as f:
                json.dump(house, f)

    def get_json(self, additional_nodes=None):
        """
        Get the json of the current room, plus additional_nodes

        Parameters
        ----------
        additional_nodes (list[SynthNode]): objects to be included
            in addition to the current room, this is used
            to compute the collisions, since those codes are based
            on the original SUNCG json format
        """
        if self.empty_house_json is None:
            self._create_empty_house_json()

        house = copy.deepcopy(self.empty_house_json)
        level = house["levels"][0]
        room = level["nodes"][0]

        if additional_nodes is None:
            object_nodes = self.object_nodes
        else:
            object_nodes = self.object_nodes + additional_nodes

        count = len(self.door_window_nodes) + 1
        for node in object_nodes:
            node_json = {}
            room["nodeIndices"].append(str(count))
            node_json["id"] = f"0_{count}"
            node_json["type"] = "Object"
            node_json["valid"] = 1
            modelId = node.modelId
            transformation_3d = self.projection.to_3d(node.get_transformation())
            if "mirror" in modelId:
                t_reflec = np.asarray([[-1, 0, 0, 0], \
                                  [0, 1, 0, 0], \
                                  [0, 0, 1, 0], \
                                  [0, 0, 0, 1]])
                transformation_3d = np.dot(t_reflec, transformation_3d)
                modelId = modelId.replace("_mirror","")
            
            alignment_matrix = self.obj_data.get_alignment_matrix(modelId)
            if alignment_matrix is not None:
                transformation_3d = np.dot(alignment_matrix, transformation_3d)
            node_json["modelId"] = modelId
            node_json["transform"] = list(transformation_3d.flatten())

            level["nodes"].append(node_json)
            count += 1

        return house

    def _get_collisions(self, additional_nodes=None):
        with stdout_redirected():
            oc = self.synthesizer.object_collection
            oc.reset()
            oc.init_from_house(House(house_json=self.get_json(additional_nodes)))
            contacts = oc.get_collisions(include_collision_with_static=True)
            collisions = []
        for (_, contact_record) in contacts.items():
            #print(collision_pair)
            if contact_record.idA != contact_record.idB:
                #If contact with the room geometry, be more lenient and allow anything with 
                #less than 0.25m overlap
                if "0_0" in contact_record.idA or "0_0" in contact_record.idB:
                    if contact_record.distance < -0.25:
                        collisions.append(contact_record)
                else:
                    #Else, check if collision amount is more than max_collision, if, then it is a collision
                    if contact_record.distance < self.max_collision:
                        collisions.append(contact_record)
                    #Else, we do an additional check to see overlap of two objects along either axes
                    #Are greater than 1/5 of the smaller object
                    #Just a rough heuristics to make sure small objects don't overlap too much
                    #Since max_collision can be too large for those
                    elif contact_record.distance < -0.02:
                        idA = contact_record.idA
                        idB = contact_record.idB
                        aabbA = oc._objects[idA].obb.to_aabb()
                        aabbB = oc._objects[idB].obb.to_aabb()
                        def check_overlap(amin,amax,bmin,bmax):
                            return max(0, min(amax, bmax) - max(amin, bmin))
                        x_overlap = check_overlap(aabbA[0][0],aabbA[1][0],aabbB[0][0],aabbB[1][0])
                        y_overlap = check_overlap(aabbA[0][2],aabbA[1][2],aabbB[0][2],aabbB[1][2])
                        x_overlap /= min((aabbA[1][0]-aabbA[0][0]), (aabbB[1][0]-aabbB[0][0]))
                        y_overlap /= min((aabbA[1][2]-aabbA[0][2]), (aabbB[1][2]-aabbB[0][2]))
                        if (x_overlap > 0.2 and y_overlap > 0.2):
                            collisions.append(contact_record)

        return collisions

class SynthNode():
    """
    Representing a node in synthesis time
    """
    def __init__(self, modelId, category, x, y, r, room):
        self.__dict__.update(locals())
        del self.self
        self.render = None
    
    def get_render(self):
        """
        Get the top-down render of the object
        """
        o = Obj(self.modelId)
        o.transform(self.get_transformation())
        render = torch.from_numpy(TopDownView.render_object_full_size(o, self.room.size))
        self.render = render
        return render

    def get_transformation(self):
        """
        Get the transformation matrix
        Used to render the object
        and to save in json files
        """
        x,y,r = self.x, self.y, self.r
        xscale = self.room.synthesizer.pgen.xscale
        yscale = self.room.synthesizer.pgen.yscale
        zscale = self.room.synthesizer.pgen.zscale
        zpad = self.room.synthesizer.pgen.zpad

        sin, cos = math.sin(r), math.cos(r)

        t = np.asarray([[cos, 0, -sin, 0], \
                        [0, 1, 0, 0], \
                        [sin, 0, cos, 0], \
                        [0, zpad, 0, 1]])
        t_scale = np.asarray([[xscale, 0, 0, 0], \
                              [0, zscale, 0, 0], \
                              [0, 0, xscale, 0], \
                              [0, 0, 0, 1]])
        t_shift = np.asarray([[1, 0, 0, 0], \
                              [0, 1, 0, 0], \
                              [0, 0, 1, 0], \
                              [x, 0, y, 1]])
        
        return np.dot(np.dot(t,t_scale), t_shift)
