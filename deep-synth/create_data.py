"""
Creates all the necessary dataset from raw SUNCG
Make sure to read README
IMPORTANT: make sure you don't have a directory named `temp`
under SCENESYNTH_ROOT_PATH, since that will be removed relentlessly
"""
#Didn't implement any checkpoints, comment out parts as you wish...
import shutil
from utils import get_data_root_dir
from data.object import parse_objects

root_dir = get_data_root_dir()

parse_objects()
from data.dataset import create_dataset
print("Creating initial dataset...")
create_dataset()

from scene_filter import get_filter, run_filter
from model_prior import ModelPrior
print("Extracting houses with acceptable quality...")
filter_description = [("good_house",)]
run_filter(filter_description, "main", "good", 1, 1, 0, 1, 0)
print()

print("Creating bedroom dataset...")
filter_description = [("room_type", ["Bedroom"]), ("floor_node",), ("renderable",)]
run_filter(filter_description, "good", "temp", 1, 1, 0, 1, 0)
filter_description = [("bedroom", "final"), ("collision",)]
run_filter(filter_description, "temp", "bedroom", 1, 1, 1, 0, 1)
print()
print("Creating model prior for bedroom...")
mp = ModelPrior()
mp.learn("bedroom")
mp.save()
print()
shutil.rmtree(f"{root_dir}/temp")

print("Creating living room dataset...")
filter_description = [("room_type", ["Living_Room"]), ("floor_node",), ("renderable",)]
run_filter(filter_description, "good", "temp", 1, 1, 0, 1, 0)
filter_description = [("livingroom", "final"), ("collision",)]
run_filter(filter_description, "temp", "living", 1, 1, 1, 0, 1)
print()
print("Creating model prior for living room...")
mp = ModelPrior()
mp.learn("living")
mp.save()
print()
shutil.rmtree(f"{root_dir}/temp")

print("Creating office dataset...")
filter_description = [("room_type", ["Office"]), ("floor_node",), ("renderable",)]
run_filter(filter_description, "good", "temp", 1, 1, 0, 1, 0)
filter_description = [("office", "final"), ("collision",)]
run_filter(filter_description, "temp", "office", 1, 1, 1, 0, 1)
print()
print("Creating model prior for office...")
mp = ModelPrior()
mp.learn("office")
mp.save()
print()
shutil.rmtree(f"{root_dir}/temp")
