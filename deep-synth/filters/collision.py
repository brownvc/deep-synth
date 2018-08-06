from data.house import *
from data.dataset import DatasetFilter
from data.object_data import ObjectCategories
import copy
import os
from utils import stdout_redirected

"""
Filters out rooms with large collisions
"""
def collision_filter(oc):
    category_map = ObjectCategories()
    def room_criteria(room, house):
        with stdout_redirected():
            oc.reset()
            oc.init_from_room(copy.deepcopy(house), room.id)
            collisions = oc.get_collisions()

        has_contact = False
        for (collision_pair, contact_record) in collisions.items():
            if contact_record.distance < -0.15:
                idA = contact_record.idA
                idB = contact_record.idB
                if idA == idB:
                    node = next(node for node in room.nodes if node.id==idA)
                    if category_map.get_final_category(node.modelId) not in ["window", "door"]:
                        has_contact = True
                else:
                    has_contact = True

        return not has_contact

    dataset_f = DatasetFilter(room_filters = [room_criteria])
    return dataset_f
