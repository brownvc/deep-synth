from data.house import *
from data.dataset import DatasetFilter
from data.object_data import ObjectCategories

"""
Floor node if either:
Parent is floor (from house_relations, not loaded by default in the released version of the code, 
    so parent is always None and this gets ignored )
Distance from floor is less than 0.1 meters
Is a door/window (since those have to be included)
"""

def floor_node_filter():
    category = ObjectCategories()
    def node_criteria(node, room):
        return (node.parent and node.parent == "Floor") or (node.zmin - room.zmin < 0.1) or (hasattr(node, "modelId") and category.get_final_category(node.modelId) in ["door", "window"])

    dataset_f = DatasetFilter(node_filters = [node_criteria])
    return dataset_f
