from data.house import *
from data.dataset import DatasetFilter
from data.object_data import ObjectCategories
from .global_category_filter import *
import utils

"""
Bedroom filter
"""

def bedroom_filter(version, source):
    data_dir = utils.get_data_root_dir()
    with open(f"{data_dir}/{source}/coarse_categories_frequency", "r") as f:
        coarse_categories_frequency = ([s[:-1] for s in f.readlines()])
        coarse_categories_frequency = [s.split(" ") for s in coarse_categories_frequency]
        coarse_categories_frequency = dict([(a,int(b)) for (a,b) in coarse_categories_frequency])
    category_map = ObjectCategories()

    if version == "final":
        filtered, rejected, door_window = GlobalCategoryFilter.get_filter()
        with open(f"{data_dir}/{source}/final_categories_frequency", "r") as f:
            frequency = ([s[:-1] for s in f.readlines()])
            frequency = [s.split(" ") for s in frequency]
            frequency = dict([(a,int(b)) for (a,b) in frequency])

        def node_criteria(node, room):
            category = category_map.get_final_category(node.modelId)
            if category in filtered: return False
            return True

        def room_criteria(room, house):
            node_count = 0
            bed_count = 0 #Must have one bed
            for node in room.nodes:
                category = category_map.get_final_category(node.modelId)
                if category in rejected:
                    return False
                if not category in door_window:
                    node_count += 1

                    t = np.asarray(node.transform).reshape((4,4)).transpose()
                    a = t[0][0]
                    b = t[0][2]
                    c = t[2][0]
                    d = t[2][2]
                    
                    xscale = (a**2 + c**2)**0.5
                    yscale = (b**2 + d**2)**0.5
                    zscale = t[1][1]
                    
                    if not 0.8<xscale<1.2: #Reject rooms where any object is scaled by too much
                        return False
                    if not 0.8<yscale<1.2:
                        return False
                    if not 0.8<zscale<1.2:
                        return False

                if "bed" in category:
                    bed_count += 1

                if frequency[category] < 500: return False

            if node_count < 5 or node_count > 20: return False
            if bed_count < 1: return False

            return True

    else:
        raise NotImplementedError

    dataset_f = DatasetFilter(room_filters = [room_criteria], node_filters = [node_criteria])

    return dataset_f


