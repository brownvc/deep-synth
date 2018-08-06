from scene_synth import *
import argparse
import utils
"""
Sample script to call scene_synth
Modify as wanted
"""
parser = argparse.ArgumentParser(description='Synth parameter search')
parser.add_argument('--temperature-cat', type=float, default=0.25, metavar='N')
parser.add_argument('--temperature-pixel', type=float, default=0.4, metavar='N')
parser.add_argument('--min-p', type=float, default=0.5, metavar='N')
parser.add_argument('--max-collision', type=float, default=-0.1, metavar='N')
parser.add_argument('--save-dir', type=str, default="synth", metavar='S')
parser.add_argument('--data-dir', type=str, default="bedroom", metavar='S')
parser.add_argument('--model-dir', type=str, default="train/bedroom", metavar='S')
parser.add_argument('--continue-epoch', type=int, default=50, metavar='N')
parser.add_argument('--location-epoch', type=int, default=300, metavar='N')
parser.add_argument('--rotation-epoch', type=int, default=300, metavar='N')
parser.add_argument('--start', type=int, default=0, metavar='N')
parser.add_argument('--end', type=int, default=1, metavar='N')
parser.add_argument('--trials', type=int, default=1, metavar='N')
args = parser.parse_args()

#All the SceneSynth parameters that can be controlled 
params = {'temperature_cat' : args.temperature_cat,
          'temperature_pixel' : args.temperature_pixel,
          'min_p' : args.min_p,
          'max_collision' : args.max_collision}

print(params)
save_dir = args.save_dir
utils.ensuredir(save_dir)

s = SceneSynth(args.location_epoch, args.rotation_epoch, args.continue_epoch, args.data_dir, args.model_dir)
s.synth(range(args.start, args.end), trials=args.trials, save_dir=args.save_dir, **params)
