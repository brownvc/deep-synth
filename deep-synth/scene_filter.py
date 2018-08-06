from data import *
import os
import pickle
import math
import pickle

"""
Performs various kind of operations to the dataset
"""
filter_description = []
#==================EDIT HERE====================
#filter_description should be a list of tuples in the form of (filter_description, *params for filter)
#Here are some examples

#filter_description = [("floor_node",)]
#filter_description = [("room_type", ["Office"]), ("floor_node",), ("renderable",)]
filter_description = [("bedroom", "final"), ("collision",)]
#filter_description = [("good_house",)]
source = "bedroom" #Source and dest are relative to utils.get_data_root_dir()
dest = "bedroom_fin"
#I hate typing True and False
stats = 1 #If true, print stats about the dataset
save_freq = 1 #If true, save category frequency count to the dest directory
render = 1 #If true, render the dataset
save = 0 #If true, save the filtered dataset back as pkl files
json = 1 #If true, save the json files
#There seems to be a KNOWN BUG (I can't remember) that prevents using render/json together with save
#So avoid using them together to be save
render_size = 512
#===============================================

def get_filter(source, filter_type, *args):
    #Just brute force enumerate possible implemented filters
    #See filters/
    if filter_type == "good_house":
        from filters.good_house import good_house_criteria
        house_f = good_house_criteria
        dataset_f = DatasetFilter(house_filters = [house_f])
    elif filter_type == "room_type":
        from filters.room_type import room_type_criteria
        room_f = room_type_criteria(*args)
        dataset_f = DatasetFilter(room_filters = [room_f])
    elif filter_type == "bedroom":
        from filters.bedroom import bedroom_filter
        dataset_f = bedroom_filter(*args, source)
    elif filter_type == "office":
        from filters.office import office_filter
        dataset_f = office_filter(*args, source)
    elif filter_type == "livingroom":
        from filters.livingroom import livingroom_filter
        dataset_f = livingroom_filter(*args, source)
    elif filter_type == "floor_node":
        from filters.floor_node import floor_node_filter
        dataset_f = floor_node_filter(*args)
    elif filter_type == "renderable":
        from filters.renderable import renderable_room_filter
        dataset_f = renderable_room_filter(*args)
    elif filter_type == "collision":
        from filters.collision import collision_filter
        from priors.observations import ObjectCollection
        oc = ObjectCollection()
        dataset_f = collision_filter(oc)
    else:
        raise NotImplementedError
    return dataset_f

def run_filter(filter_description, source, dest, stats, save_freq, render, save, json, render_size=512):
    """
    Parameters
    ----------
    source, dest (String): location of source/dest relative (should be) to utils.get_data_root_dir()
    stats (bool): If true, print stats about the dataset
    save_freq (bool): If true, save category frequency count to the dest directory
    render (bool): If true, render the dataset
    save (bool): If true, save the filtered dataset back as pkl files
    json (bool): If true, save the json files
    render_size (int): size of the top down render
    """
    actions = []
    for description in filter_description:
        actions.append(get_filter(source, *description))
    if stats:
        actions.append(DatasetStats(save_freq=save_freq, details=False, model_details=False, save_dest=dest))
    if save:
        actions.append(DatasetSaver(dest=dest, batch_size=1000))
    if render:
        actions.append(DatasetRenderer(dest=dest, size=render_size))
    if json:
        actions.append(DatasetToJSON(dest=dest))
    
    d = Dataset(actions, source=source)
    d.run()

if __name__ == "__main__":
    run_filter(filter_description, source, dest, stats, save_freq, render, save, json, render_size)
