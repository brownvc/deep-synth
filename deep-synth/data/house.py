"""
Three level House-Level-Node/Room representation of SUNCG
"""
import os
import json
import numpy as np
from data import ObjectData
import utils

class House():
    """
    Represents a House
    See SUNCG Toolbox for a detailed description of the structure of the json file
    describing a house
    """
    object_data = ObjectData()
    def __init__(self, index=0, id_=None, house_json=None, file_dir=None,
                 include_support_information=False, include_arch_information=False):
        """
        Get a set of rooms from the house which satisfies a certain criteria

        Parameters
        ----------
        index (int): The index of the house among all houses sorted in alphabetical order
            default way of loading a house
        id_ (string, optional): If set, then the house with the specified directory name is chosen
        house_json(json, optional): If set, then the specified json object
            is used directly to initiate the house
        file_dir (string, optional): If set, then the json pointed to by file_dir will be loaded
        include_support_information(bool): If true, then support information is loaded from suncg_data/house_relations
            might not be available, so defaults to False
        include_arch_information (bool): If true, then arch information is loaded from suncg_data/wall
            might not be available, so defaults to False
        """
        if house_json is None:
            if file_dir is None:
                data_dir = utils.get_data_root_dir()
                house_dir = data_dir + "/suncg_data/house/"

                if id_ is None:
                    houses = dict(enumerate(os.listdir(house_dir)))
                    self.__dict__ = json.loads(open(house_dir+houses[index]+"/house.json", 'r').read())
                else:
                    self.__dict__ = json.loads(open(house_dir+id_+"/house.json", 'r').read())
            else:
                self.__dict__ = json.loads(open(file_dir, 'r').read())
        else:
            self.__dict__ = house_json

        self.filters = []
        self.levels = [Level(l,self) for l in self.levels]
        self.rooms = [r for l in self.levels for r in l.rooms]
        self.nodes = [n for l in self.levels for n in l.nodes]
        self.node_dict = {id_: n for l in self.levels for id_,n in l.node_dict.items()}
        if include_support_information:
            house_stats_dir = data_dir + "/suncg_data/house_relations/"
            stats = json.loads(open(house_stats_dir+self.id+"/"+self.id+".stats.json", 'r').read())
            supports = [(s["parent"],s["child"]) for s in stats["relations"]["support"]]
            for parent, child in supports:
                if child not in self.node_dict:
                    print(f'Warning: support relation {supports} involves not present {child} node')
                    continue
                if "f" in parent:
                    self.get_node(child).parent = "Floor"
                elif "c" in parent:
                    self.get_node(child).parent = "Ceiling"
                elif len(parent.split("_")) > 2:
                    self.get_node(child).parent = "Wall"
                else:
                    if parent not in self.node_dict:
                        print(f'Warning: support relation {supports} involves not present {parent} node')
                        continue
                    self.get_node(parent).child.append(self.get_node(child))
                    self.get_node(child).parent = self.get_node(parent)
        if include_arch_information:
            house_arch_dir = data_dir + '/suncg_data/wall/'
            arch = json.loads(open(house_arch_dir+self.id+'/'+self.id+'.arch.json', 'r').read())
            self.walls = [w for w in arch['elements'] if w['type'] == 'Wall']
    
    def get_node(self, id_):
        return self.node_dict[id_]

    def get_rooms(self, filters=None):
        """
        Get a set of rooms from the house which satisfies a certain criteria

        Parameters
        ----------
        filters (list[room_filter]): room_filter is tuple[Room,House] which returns
            if the Room should be included
        
        Returns
        -------
        list[Room]
        """
        if filters is None: filters = self.filters
        if not isinstance(filters, list): filters = [filters]
        rooms = self.rooms
        for filter_ in filters:
            rooms = [room for room in rooms if filter_(room, self)]
        return rooms
    
    def filter_rooms(self, filters):
        """
        Similar to get_rooms, but overwrites self.node instead of returning a list
        """
        self.rooms = self.get_rooms(filters)
    
    def trim(self):
        """
        Get rid of some intermediate attributes
        """
        nodes = list(self.node_dict.values())
        if hasattr(self, 'rooms'):
            nodes.extend(self.rooms)
        for n in nodes:
            for attr in ['xform', 'obb', 'frame', 'model2world']:
                if hasattr(n, attr):
                    delattr(n, attr)
        self.nodes = None
        self.walls = None
        for room in self.rooms:
            room.filters = None
        self.levels = None
        self.node_dict = None
        self.filters = None


class Level():
    """
    Represents a floor level in the house
    Currently mostly just used to parse the list of nodes and rooms
    Might change in the future
    """
    def __init__(self, dict_, house):
        self.__dict__ = dict_
        self.house = house
        invalid_nodes = [n["id"] for n in self.nodes if (not n["valid"]) and "id" in n]
        self.nodes = [Node(n,self) for n in self.nodes if n["valid"]]
        self.node_dict = {n.id: n for n in self.nodes}
        self.nodes = list(self.node_dict.values())  # deduplicate nodes with same id
        self.rooms = [Room(n, ([self.node_dict[i] for i in [f"{self.id}_{j}" \
                      for j in list(set(n.nodeIndices))] if i not in invalid_nodes]), self) \
                      for n in self.nodes if n.isRoom() and hasattr(n, 'nodeIndices')]

class Room():
    """
    Represents a room in the house
    """
    def __init__(self, room, nodes, level):
        self.__dict__ = room.__dict__
        #self.room = room
        self.nodes = nodes
        #self.level = level
        self.filters = []
        self.house_id = level.house.id
    
    def get_nodes(self, filters=None):
        """
        Get a set of nodes from the room which satisfies a certain criteria

        Parameters
        ----------
        filters (list[node_filter]): node_filter is tuple[Node,Room] which returns
            if the Node should be included
        
        Returns
        -------
        list[Node]
        """
        if filters is None: filters = self.filters
        if not isinstance(filters, list): filters = [filters]
        nodes = self.nodes
        for filter_ in filters:
            nodes = [node for node in nodes if filter_(node, self)]
        return nodes
    
    def filter_nodes(self, filters):
        """
        Similar to get_nodes, but overwrites self.node instead of returning a list
        """
        self.nodes = self.get_nodes(filters)

class Node():
    """
    Basic unit of representation of SUNCG
    Usually a room or an object
    Refer to SUNCG toolbox for the possible set of attributes
    """
    warning = True
    def __init__(self, dict_, level):
        self.__dict__ = dict_
        #self.level = level
        self.parent = None
        self.child = []
        if hasattr(self, 'bbox'):
            (self.xmin, self.zmin, self.ymin) = self.bbox["min"]
            (self.xmax, self.zmax, self.ymax) = self.bbox["max"]
            (self.width, self.length) = sorted([self.xmax - self.xmin, self.ymax - self.ymin])
            self.height = self.zmax - self.zmin
        else:  # warn and populate with default bbox values
            if self.warning:
                print(f'Warning: node id={self.id} is valid but has no bbox, setting default values')
            (self.xmin, self.zmin, self.ymin) = (0, 0, 0)
            (self.xmax, self.zmax, self.ymax) = (0, 0, 0)
        if hasattr(self, 'transform') and hasattr(self, 'modelId'):
            t = np.asarray(self.transform).reshape(4,4)
            #Special cases of models with were not aligned in the way we want
            alignment_matrix = House.object_data.get_alignment_matrix(self.modelId)
            if alignment_matrix is not None:
                t = np.dot(np.linalg.inv(alignment_matrix), t)
                self.transform = list(t.flatten())
                
            #If a reflection is present, switch to the mirrored model
            #And adjust transform accordingly
            if np.linalg.det(t) < 0:
                t_reflec = np.asarray([[-1, 0, 0, 0], \
                                      [0, 1, 0, 0], \
                                      [0, 0, 1, 0], \
                                      [0, 0, 0, 1]])
                t = np.dot(t_reflec, t)
                self.modelId += "_mirror"
                self.transform = list(t.flatten())

    def isRoom(self):
        return self.type == "Room"

if __name__ == "__main__":
    a = House(id_ = "f53c0878c4db848bfa43163473b74245")
    for room in a.rooms:
        if room.id == "0_7":
            print(room.__dict__)
