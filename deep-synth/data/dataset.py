"""
Tools used to process multiple houses in SUNCG
A pickle version of the entire dataset should be created first
by calling create_dataset (or python -m data.dataset)
The pickle version can then be handled by the Dataset class
"""
import os
import json
import pickle
from data import ObjectData, House, ObjectCategories, TopDownView
from abc import ABC, abstractmethod
import scipy.misc as m
import copy
import utils
import numpy as np

def create_dataset(source="SUNCG", dest="main", \
                   num_houses=-1, batch_size=1000):
    """
    Create a pickled version of the dataset from the json files

    Parameters
    ----------
    dest (string, optional): directory to save to
    num_houses (int, optional): If -1, then all the houses will be loaded, otherwise
        the first num_houses houses will be loaded
    batch_size (int, optional): number of houses in one .pkl file
    """
    data_dir = utils.get_data_root_dir()
    dest_dir = f"{data_dir}/{dest}"
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    to_save = []
    cur_index = 0
    def pkl_name(index):
        return f"{dest_dir}/{cur_index}.pkl"

    if source == "SUNCG":
        house_dir = f"{data_dir}/suncg_data/house"
        house_ids = dict(enumerate(os.listdir(house_dir)))
        print(f"There are {len(house_ids)} houses in the dataset.")
        num_houses = len(house_ids) if num_houses == -1 else num_houses
        start_house_i = 0
        while os.path.exists(pkl_name(cur_index)):
            print(f'Batch file {pkl_name(cur_index)} exists, skipping batch')
            cur_index += 1
            start_house_i = cur_index * batch_size
        for i in range(start_house_i, num_houses):
            print(f"Now loading house id={house_ids[i]} {i+1}/{num_houses}...", end="\r")
            house = House(i)
            if house.rooms:
                to_save.append(house)
                if len(to_save) == batch_size:
                    with open(pkl_name(cur_index), "wb") as f:
                        pickle.dump(to_save, f, pickle.HIGHEST_PROTOCOL)
                        to_save = []
                        cur_index += 1
        print()

        with open(pkl_name(cur_index), "wb") as f:
            pickle.dump(to_save, f, pickle.HIGHEST_PROTOCOL)
    else:
        print("Currently only supports loading SUNCG")


class Dataset():
    """
    Class that processed the pickled version of the SUNCG dataset
    """
    def __init__(self, actions, source="main", num_batches=0):
        """
        Parameters
        ----------
        action (list[DatasetAction]): a list of methods that are applied to each
            loaded house sequentially
        source (string): location of pickled dataset to be loaded
        num_batches (int): if not 0, only the first num_batches bathces of houses 
            will be loaded
        """
        data_dir = utils.get_data_root_dir()
        source_dir = f"{data_dir}/{source}"
        if not isinstance(actions, list):
            actions = [actions]
        self.actions = actions

        files = sorted([s for s in os.listdir(source_dir) if "pkl" in s], \
                        key = lambda x: int(x.split(".")[0]))

        if num_batches > 0: files = files[0:num_batches]
        self.files = [f"{source_dir}/{f}" for f in files]

    def run(self):
        for (i, fname) in enumerate(self.files):
            with open(fname, "rb") as f:
                print(f"Loading part {i}...", end="\r")
                houses = pickle.load(f)
                for action in self.actions:
                    houses = action.step(houses)
                for house in houses:
                    _ = house
        for action in self.actions:
            action.final()

class DatasetAction(ABC):
    """
    Abstract base class defining actions that can be done to the Dataset
    """
    def __init__(self):
        pass
    
    @abstractmethod
    def step(self, houses):
        """
        Called for every batch of houses loaded by the dataset
        If multiple DatasetAction is passed to the dataset, then they will be
        called sequentially
        
        Yield
        -----
        generator[House]: since multiple DatasetAction can be used, make sure to 
            return a generator of houses to be used later
        """
        pass

    def final(self):
        """
        Called once after all batches of houses are processed
        """
        pass

class DatasetFilter(DatasetAction):
    """
    Filters every node in the House based on a set of node_filters
    Then filters every room (with nodes filtered already)
        in the House based on a set of room_filters
    Finally, a House is filtered if either:
        -it does not pass any of the house_filter
        -it contains no rooms after filtering rooms
    """
    def __init__(self, room_filters=[], node_filters=[], house_filters=[]):
        """
        Parameters
        ----------
        house_filters (list[house_filter]): where a house_filter takes a House as input
            and returns whether the House should be included or not
        room_filters (list[room_filter]): see house.House.filter_rooms
        node_filters (list[node_filter]): see house.Room.filter_nodes
        """
        self.room_filters = room_filters
        self.node_filters = node_filters
        self.house_filters = house_filters

    def step(self, houses):
        for house in houses:
            if self.house_filters:
                if any(not f(house) for f in self.house_filters):
                    continue
            if self.node_filters:
                for room in house.rooms:
                    room.filter_nodes(self.node_filters)
            if self.room_filters:
                house.filter_rooms(self.room_filters)
            if house.rooms:
                yield house

class DatasetRenderer(DatasetAction):
    """
    Pre-render top-down view of
    each room in the house (floor, walls and objects)
    """
    def __init__(self, dest, size=512):
        self.dest = dest
        self.size = size
        self.count = 0
        self.renderer = TopDownView(size=self.size)
        data_dir = utils.get_data_root_dir()
        self.dest_dir = f"{data_dir}/{dest}"
        if not os.path.exists(self.dest_dir):
            os.makedirs(self.dest_dir)

    def step(self, houses):
        for house in houses:
            if house.rooms:
                for room in house.rooms:
                    img, data = self.renderer.render(room)
                    with open(f"{self.dest_dir}/{self.count}.pkl","wb") as f:
                        pickle.dump((data, room), f, pickle.HIGHEST_PROTOCOL)

                    img = m.toimage(img, cmin=0, cmax=1)
                    img.save(f"{self.dest_dir}/{self.count}.jpg")
                    print(f"Rendering room {self.count}...", end="\r")
                    self.count += 1
            yield house
        print()

class DatasetToJSON(DatasetAction):
    """
    Converts the dataset back to the original json format
    """
    def __init__(self, dest):
        data_dir = utils.get_data_root_dir()
        self.dest_dir = f"{data_dir}/{dest}/json"
        self.count = 0
        if not os.path.exists(self.dest_dir):
            os.makedirs(self.dest_dir)

    def step(self, houses):
        for house in houses:
            rooms = house.rooms
            del house.rooms
            del house.filters
            del house.nodes
            del house.node_dict
            house.levels = []

            #new_house["id"] = f"room_{self.count}"
            for room in rooms:
                new_house = copy.deepcopy(house.__dict__)
                l = room.id.split("_")[0]
                cur_level = {}
                cur_level["id"] = l
                cur_level["nodes"] = []
                new_house["levels"].append(cur_level)
                nodes = room.nodes
                del room.nodes
                room.nodeIndices = []
                for node in nodes:
                    if hasattr(node, 'parent'):
                        del node.parent, node.child
                        del node.xmin, node.xmax
                        del node.ymin, node.ymax
                        del node.zmin, node.zmax
                        #Remove _mirror tag and apply the corresponding reverse transformation
                        if "_mirror" in node.modelId:
                            node.modelId = node.modelId.replace("_mirror", "")
                            t = np.asarray(node.transform).reshape(4,4)
                            t_reflec = np.asarray([[-1, 0, 0, 0], \
                                                  [0, 1, 0, 0], \
                                                  [0, 0, 1, 0], \
                                                  [0, 0, 0, 1]])
                            t = np.dot(t_reflec, t)
                            node.transform = list(t.flatten())
                        #Same for the special cases
                        if node.modelId in ["146", "142", "106"]:
                            t = np.asarray(node.transform).reshape(4,4)
                            d = ObjectData()
                            t = np.dot(d.get_alignment_matrix(node.modelId), t)
                            node.transform = list(t.flatten())
                    cur_level["nodes"].append(node.__dict__)
                    room.nodeIndices.append(node.id.split("_")[1])
                
                del room.filters
                del room.house_id
                del room.parent, room.child
                del room.xmin, room.xmax
                del room.ymin, room.ymax
                del room.zmin, room.zmax
                
                cur_level["nodes"].append(room.__dict__)

                with open(f"{self.dest_dir}/{self.count}.json", "w") as f:
                    json.dump(new_house, f)

                self.count += 1
            yield house

class DatasetSaver(DatasetAction):
    """
    Save the (usually filtered) dataset into pickle files
    """
    def __init__(self, dest, batch_size=1000, trim=False):
        """
        Parameters
        ----------
        dest (string): directory to save to
        batch_size (int, optional): number of houses in one .pkl file
        trim (bool, optional): If true, removes certain unused attributes
            to reduce file size
        """
        data_dir = utils.get_data_root_dir()
        self.dest_dir = f"{data_dir}/{dest}"
        self.to_save = []
        self.cur_index = 0
        self.batch_size = batch_size
        self.trim = trim
        if not os.path.exists(self.dest_dir):
            os.makedirs(self.dest_dir)
    
    def pkl_name(self, index):
        return f"{self.dest_dir}/{index}.pkl"

    def step(self, houses):
        for house in houses:
            if house.rooms:
                if self.trim:
                    house.trim()
                self.to_save.append(house)
                if len(self.to_save) == self.batch_size:
                    with open(self.pkl_name(self.cur_index), "wb") as f:
                        pickle.dump(self.to_save, f, pickle.HIGHEST_PROTOCOL)
                        self.to_save = []
                        self.cur_index += 1
            yield house

    def final(self):
        with open(self.pkl_name(self.cur_index), "wb") as f:
            pickle.dump(self.to_save, f, pickle.HIGHEST_PROTOCOL)

class DatasetStats(DatasetAction):
    """
    Gather stats of the houses
    Useful to get an idea of what's in the dataset
    And must be called to generate the model frequency information that's used
    in the NN modules
    """
    def __init__(self, details=False, model_details=False, save_freq=True, save_dest=""):
        """
        Parameters
        ----------
        details (bool, optional): If true, then frequency information will be shown on screen
        model_details (bool, optional): since there are so many model ids, this additional bool
            controls if those should be printed
        save_freq (bool, optional): if true, then the category frequency information will be saved
        save_dest (string, optional): directory to which frequency information is saved
        """
        self.room_count = 0
        self.object_count = 0
        self.room_types_count = {}
        self.fine_categories_count = {}
        self.coarse_categories_count = {}
        self.final_categories_count = {}
        self.models_count = {}
        self.object_category = ObjectCategories()
        self.floor_node_only = False
        self.details = details
        self.model_details = model_details
        self.save_freq = save_freq
        data_dir = utils.get_data_root_dir()
        self.save_dest = f"{data_dir}/{save_dest}"
    
    def step(self, houses):
        for house in houses:
            self.room_count += len(house.rooms)
            for room in house.rooms:
                room_types = room.roomTypes
                for room_type in room_types:
                    self.room_types_count[room_type] = \
                        self.room_types_count.get(room_type, 0) + 1
            filters = [floor_node_filter] if self.floor_node_only else []
            nodes = list(set([node for nodes in [room.get_nodes(filters) \
                              for room in house.rooms] for node in nodes \
                              if node.type == "Object"]))
            for node in nodes:
                self.object_count += 1
                fine_category = self.object_category.get_fine_category(node.modelId)           
                coarse_category = self.object_category.get_coarse_category(node.modelId)
                final_category = self.object_category.get_final_category(node.modelId)

                self.fine_categories_count[fine_category] = \
                    self.fine_categories_count.get(fine_category, 0) + 1
                self.coarse_categories_count[coarse_category] = \
                    self.coarse_categories_count.get(coarse_category, 0) + 1
                self.final_categories_count[final_category] = \
                    self.final_categories_count.get(final_category, 0) + 1
                self.models_count[node.modelId] = \
                    self.models_count.get(node.modelId, 0) + 1
            yield house
    
    def final(self):
        print(f"\nPrinting Results...")
        print(f"\nThere are {self.room_count} non-empty rooms in the selection.")
        print(f"There are {self.object_count} objects in the rooms.")
        print(f"On average, there are {self.object_count/self.room_count:.3f} objects for each room\n")
        
        print(f"There are {len(self.fine_categories_count)} fine categories among these objects.")
        
        if self.details:
            print(f"\n{'Model Category':40s}{'Occurence'}")
            for category in sorted(list((self.fine_categories_count.items())), key=lambda x: -x[1]):
                print(f"{category[0]:40s}{category[1]}")
        
        print(f"\nThere are {len(self.coarse_categories_count)} coarse categories among these objects.")
        if self.details:
            print(f"\n{'Coarse Category':40s}{'Occurence'}")
            for category in sorted(list((self.coarse_categories_count.items())), key=lambda x: -x[1]):
                print(f"{category[0]:40s}{category[1]}")
                
        print(f"\nThere are {len(self.final_categories_count)} final categories among these objects.")
        if self.details:
            print(f"\n{'Final Category':40s}{'Occurence'}")
            for category in sorted(list((self.final_categories_count.items())), key=lambda x: -x[1]):
                print(f"{category[0]:40s}{category[1]}")
        
        print(f"\nThere are {len(self.models_count)} unique models among these objects.")
        if self.details and self.model_details:
            print(f"\n{'Model':40s}{'Occurence'}")
            for category in sorted(list((self.models_count.items())), key=lambda x: -x[1]):
                print(f"{category[0]:40s}{category[1]}")

        if self.save_freq:
            with open(f"{self.save_dest}/fine_categories_frequency", "w") as f:
                for cat in sorted(list((self.fine_categories_count.items())), key=lambda x: -x[1]):
                    f.write(f"{cat[0]} {cat[1]}\n")
            with open(f"{self.save_dest}/coarse_categories_frequency", "w") as f:
                for cat in sorted(list((self.coarse_categories_count.items())), key=lambda x: -x[1]):
                    f.write(f"{cat[0]} {cat[1]}\n")
            with open(f"{self.save_dest}/final_categories_frequency", "w") as f:
                for cat in sorted(list((self.final_categories_count.items())), key=lambda x: -x[1]):
                    f.write(f"{cat[0]} {cat[1]}\n")
            with open(f"{self.save_dest}/model_frequency", "w") as f:
                for cat in sorted(list((self.models_count.items())), key=lambda x: -x[1]):
                    f.write(f"{cat[0]} {cat[1]}\n")
        
if __name__ == "__main__":
    create_dataset()
