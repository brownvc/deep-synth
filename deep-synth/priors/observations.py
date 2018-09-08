#!/usr/bin/env python3

import argparse
import csv
import glob
import os
from random import random, randint
import utils
from collections import namedtuple
from data import ObjectCategories, ObjectData, House
from math_utils import *
from math_utils.OBB import OBB
from math_utils.Simulator import Simulator


class ObjectCollection:
    """Provides observation information for a collection of objects"""
    def __init__(self, categorization_type='final', sim_mode='direct'):
        self._object_data = ObjectData()
        self._object_categories = ObjectCategories()
        self._objects = {}
        self._room = None
        self._categorization_type = categorization_type
        self._sim = Simulator(mode=sim_mode)

    @property
    def simulator(self):
        return self._sim

    @property
    def room(self):
        return self._room

    @property
    def objects(self):
        return self._objects

    def add_object(self, o):
        if o.id in self._objects and self._objects[o.id] != o:
            print(f'Warning: ignoring node with duplicate node id={o.id}')
            return None
        if hasattr(o, 'type') and o.type == 'Room':  # room node
            self._room = o
            self._sim.add_room(o, wall=True, floor=True, ceiling=False)
        else:  # other nodes
            self._sim.add_object(o)
        self.update(o, update_sim=True)
        return o.id

    def _remove(self, obj_id):
        if obj_id not in self._objects:
            print(f'Warning: tried to remove not present object with id={obj_id}')
        else:
            del self._objects[obj_id]
            self._sim.remove(obj_id)

    def update(self, o, xform=None, update_sim=False):
        if not hasattr(o, 'category'):
            o.category = self.category(o.modelId, scheme=self._categorization_type)
        model_id = o.modelId if hasattr(o, 'modelId') else None
        o.model2world = self.semantic_frame_matrix(model_id)
        o.xform = xform if xform else Transform.from_node(o)
        if hasattr(o, 'transform'):
            o.transform = o.xform.as_mat4_flat_row_major()
        o.obb = OBB.from_local2world_transform(np.matmul(o.xform.as_mat4(), o.model2world))
        o.frame = self.node_to_semantic_frame(o, o.obb)
        self._objects[o.id] = o
        # room geometries pre-transformed, so after obb computation above is done, set back to identity transform
        if hasattr(o, 'type') and o.type == 'Room':
            o.xform = Transform()
            o.transform = o.xform.as_mat4_flat_row_major()
        if update_sim:
            self._sim.set_state(obj_id=o.id, position=o.xform.translation, rotation_q=o.xform.rotation)

    def randomize_object_transforms(self):
        for o in self._objects.values():
            if o is self._room:
                continue
            t = self._room.obb.sample()
            t[1] = o.xform.translation[1]
            o.xform.set_translation(t)
            r = random() * 2 * math.pi
            o.xform.set_rotation(radians=r)
            self.update(o, o.xform)

    def init_from_room(self, house, room_id, only_architecture=False, update_sim=True):
        self.reset()
        room = next(r for r in house.rooms if r.id == room_id)
        self.add_object(room)
        if not only_architecture:
            for o in room.nodes:
                self.add_object(o)
        # if update_sim:
        #     self._sim.add_house_room_only(house, room, only_architecture=only_architecture, no_ceil=True, no_floor=True)

    def init_from_house(self, house, update_sim=True):
        self.reset()
        for o in house.nodes:
            self.add_object(o)
        # if update_sim:
        #     self._sim.add_house(house, no_ceil=True, no_floor=True)

    def as_house(self):
        room_nodes = [dict(n.__dict__) for n in self._objects.values() if n.type != 'Room']
        for i, n in enumerate(room_nodes):
            n['originalId'] = n.originalId if hasattr(n, 'originalId') else n['id']
            n['id'] = f'0_{str(i + 1)}'  # overwrite id with linearized id
        room = {
            'id': '0_0',
            'originalId': self.room.originalId if hasattr(self.room, 'originalId') else self.room.id,
            'type': 'Room',
            'valid': 1,
            'modelId': self.room.modelId,
            'nodeIndices': list(range(1, len(room_nodes) + 1)),
            'roomTypes': self.room.roomTypes,
            'bbox': self.room.bbox,
        }
        house_dict = {
            'version': 'suncg@1.0.2',
            'id': self.room.house_id,
            'up': [0, 1, 0],
            'front': [0, 0, 1],
            'scaleToMeters': 1,
            'levels': [{'id': '0', 'nodes': [room] + room_nodes}]
        }
        return House(house_json=house_dict)

    def reset(self):
        self._objects = {}
        self._sim.reset()

    def reinit_simulator(self, wall=True, floor=True, ceiling=False):
        self._sim.reset()
        for o in self._objects.values():
            if hasattr(o, 'type') and o.type == 'Room':  # room node:
                self._sim.add_room(self.room, wall=wall, floor=floor, ceiling=ceiling)
            else:  # other nodes
                self._sim.add_object(o)
            self._sim.set_state(obj_id=o.id, position=o.xform.translation, rotation_q=o.xform.rotation)

    def get_relative_observations(self, room_id, filter_ref_obj, ignore_categories):
        out = {}
        ref_objects = [filter_ref_obj] if filter_ref_obj else self._objects.values()
        for o_r in ref_objects:
            if o_r.category in ignore_categories:
                continue
            for o_i in self._objects.values():
                if o_i is o_r:
                    continue
                if o_i.category in ignore_categories:
                    continue
                out[(o_i.id, o_r.id)] = self.object_frames_to_relative_observation(o_i.frame, o_r.frame, room_id)
        return out

    def get_collisions(self, include_collision_with_static=True, obj_id_a=None):
        # update sim state to match state of this ObjectCollection
        for o_id, o in self._objects.items():
            self._sim.set_state(obj_id=o.id, position=o.xform.translation, rotation_q=o.xform.rotation)
        self._sim.step()  # sim step needed to create contacts
        return self._sim.get_contacts(obj_id_a=obj_id_a, include_collision_with_static=include_collision_with_static)

    def get_observation_key(self, observation):
        room_node = self._objects[observation.room_id]
        room_types = '-'.join(room_node.roomTypes) if hasattr(room_node, 'roomTypes') else ''
        obj_node = self._objects[observation.obj_id]
        obj_cat = self.category(obj_node.modelId, scheme=self._categorization_type)
        ref_node = self._objects[observation.ref_id]
        ref_cat = self.category(ref_node.modelId, scheme=self._categorization_type)
        key = ObservationCategory(room_types=room_types, obj_category=obj_cat, ref_obj_category=ref_cat)
        return key

    def category(self, model_id, scheme):
        if 'rm' in model_id:
            return 'room'
        if scheme == 'coarse':
            return self._object_categories.get_coarse_category(model_id)
        elif scheme == 'fine':
            return self._object_categories.get_fine_category(model_id)
        elif scheme == 'final':
            return self._object_categories.get_final_category(model_id)
        else:
            raise RuntimeError(f'Unknown categorization type: {scheme}')

    def semantic_frame_matrix(self, model_id):
        if model_id in self._object_data.model_to_data:
            return self._object_data.get_model_semantic_frame_matrix(model_id)
        else:  # not a model, so assume identity semantic frame
            return np.identity(4)

    def object_frames_to_relative_observation(self, frame, ref_frame, room_id):
        ref_dims = ref_frame['obb'].half_dimensions
        rel_centroid = ref_frame['obb'].transform_point(frame['obb'].centroid)
        rel_min = ref_frame['obb'].transform_point(frame['aabb']['min'])
        rel_max = ref_frame['obb'].transform_point(frame['aabb']['max'])
        rel_up = ref_frame['obb'].transform_direction(frame['obb'].rotation_matrix[:3, 1])
        rel_front = ref_frame['obb'].transform_direction(-frame['obb'].rotation_matrix[:3, 2])  # note: +z = back
        cp = self._sim.get_closest_point(obj_id_a=frame['obj_id'], obj_id_b=ref_frame['obj_id'])
        rel_cp = ref_frame['obb'].transform_point(cp.positionOnAInWS)
        # NOTE: below approximate closest point calls are for removing pybullet call and debugging memory leak
        # cp = frame['obb'].closest_point(ref_frame['obb'].centroid)
        # rel_cp = ref_frame['obb'].transform_point(cp)
        out = RelativeObservation(room_id=room_id, obj_id=frame['obj_id'], ref_id=ref_frame['obj_id'],
                                  ref_dims=ref_dims, centroid=rel_centroid, min=rel_min, max=rel_max, closest=rel_cp,
                                  front=rel_front, up=rel_up)
        return out

    @staticmethod
    def node_to_semantic_frame(node, obb):
        aabb_min, aabb_max = obb.to_aabb()
        out = {
            'obj_id': node.id,
            'obb': obb,
            'aabb': {'min': aabb_min, 'max': aabb_max}
        }
        return out


# an observation of object id relative to a reference object ref_id
class RelativeObservation(namedtuple('RelativeObservation',
                                     ['room_id', 'obj_id', 'ref_id', 'ref_dims', 'centroid', 'min', 'max', 'closest',
                                      'front', 'up'])):
    __slots__ = ()  # prevent per-instance dict

    def to_str_list(self):
        return [self.room_id, self.obj_id, self.ref_id, nparr2str_compact(self.ref_dims),
                nparr2str_compact(self.centroid), nparr2str_compact(self.min), nparr2str_compact(self.max),
                nparr2str_compact(self.closest), nparr2str_compact(self.front), nparr2str_compact(self.up)]

    @classmethod
    def fromstring(cls, s):
        return RelativeObservation(room_id=s['room_id'], obj_id=s['obj_id'], ref_id=s['ref_id'], ref_dims=s['ref_dims'],
                                   centroid=str2nparr(s['centroid']), min=str2nparr(s['min']), max=str2nparr(s['max']),
                                   closest=str2nparr(s['closest']), front=str2nparr(s['front']), up=str2nparr(s['up']))

    def parameterize(self, scheme):
        if scheme == 'dist_angles':
            return self._to_distance_angles()
        elif scheme == 'dist_cos_sin_angles':
            return self._to_distance_cos_sin_angles()
        elif scheme == 'offsets_cos_sin_angles':
            return self._to_offsets_cos_sin_angles()
        elif scheme == 'offsets_angles':
            return self._to_offsets_angles()
        else:
            raise RuntimeError(f'Unknown RelativeObservation parameterization scheme {scheme}')

    def _to_distance_angles(self):
        d_center, a_center = relative_pos_to_xz_distance_angle(self.centroid)
        d_closest, a_closest = relative_pos_to_xz_distance_angle(self.closest)
        a_front = relative_dir_to_xz_angle(self.front)
        return [d_center, d_closest, a_center, a_closest, a_front]

    def _to_distance_cos_sin_angles(self):
        d_center, a_center = relative_pos_to_xz_distance_angle(self.centroid)
        d_closest, a_closest = relative_pos_to_xz_distance_angle(self.closest)
        a_front = relative_dir_to_xz_angle(self.front)
        return [d_center, d_closest, math.cos(a_center), math.sin(a_center), math.cos(a_closest), math.sin(a_closest),
                math.cos(a_front), math.sin(a_front)]

    def _to_offsets_angles(self):
        a_front = relative_dir_to_xz_angle(self.front)
        return [self.centroid[0], self.centroid[2], self.closest[0], self.closest[2], a_front]

    def _to_offsets_cos_sin_angles(self):
        a_front = relative_dir_to_xz_angle(self.front)
        return [self.centroid[0], self.centroid[2], self.closest[0], self.closest[2],
                math.cos(a_front), math.sin(a_front)]


# a record categorizing a relative observation by originating room type, object category and reference object category
ObservationCategory = namedtuple('ObservationCategory', ['room_types', 'obj_category', 'ref_obj_category'])


class RelativeObservationsDatabase:
    """Encapsulates a set of object arrangement priors"""

    def __init__(self, name, priors_dir, verbose=False, categorization_type='final'):
        self._name = name
        self._priors_dir = priors_dir
        utils.ensuredir(self._priors_dir)
        self._verbose = verbose
        self._objects = ObjectCollection(categorization_type=categorization_type)
        self._semantic_frames = {}  # house_id -> {obj_id: SemanticFrame}
        self._observations = {}  # house_id -> {(obj_id,ref_obj_id): RelativeObservation}
        self._grouped_observations = {}  # {ObservationCategory: [RelativeObservation]}

    def clear(self):
        self._semantic_frames = {}
        self._observations = {}
        self._grouped_observations = {}

    @property
    def num_houses(self):
        return len(self._observations.keys())

    @property
    def grouped_observations(self):
        return self._grouped_observations

    def collect_observations(self, houses):
        for h in houses:
            self._observations[h.id] = self._get_observations_from_house(h)

    def save_observations_by_house(self, prefix='', save_frames=False):
        for (house_id, house_observations) in self._observations.items():
            basename = prefix + '_' + house_id
            csv_file = os.path.join(self._priors_dir, basename + '.relpos.csv')
            f = csv.writer(open(csv_file, 'w'))
            f.writerow(RelativeObservation._fields)
            for o in house_observations.values():
                f.writerow(o.to_str_list())

        if save_frames:
            for (house_id, house_frames) in self._semantic_frames.items():
                basename = prefix + '_' + house_id
                csv_file = os.path.join(self._priors_dir, basename + '.semframes.csv')
                f = csv.writer(open(csv_file, 'w'))
                f.writerow(['obj_id', 'local2world', 'world_aabb_min', 'world_aabb_max'])
                for frame in house_frames.values():
                    local2world_flat = nparr2str_compact(np.squeeze(np.asarray(frame['obb'].local2world.flatten())))
                    f.writerow([frame['obj_id'], local2world_flat,
                                nparr2str_compact(frame['aabb']['min']), nparr2str_compact(frame['aabb']['max'])])

    def load_observations(self, filenames, house_dir, load_frames=False):
        counts_house_ids = list(map(lambda f: f.split('_'), filenames))
        num_observations = len(counts_house_ids)
        for i, (obs_id, house_id) in enumerate(counts_house_ids):
            print(f'Loading observation obs_id={obs_id} house id={house_id}, {i}/{num_observations}')
            house_observations = self._observations.get(house_id, {})
            num_rows = 0
            for row in csv.DictReader(open(os.path.join(self._priors_dir, obs_id+'_'+house_id + '.relpos.csv'))):
                if len(row) == 0:
                    continue
                obs = RelativeObservation.fromstring(row)
                house_observations[(obs.obj_id, obs.ref_id)] = obs
                num_rows += 1
            if num_rows > 0:
                self._observations[obs_id] = house_observations

            if load_frames:
                house_frames = self._semantic_frames.get(house_id, {})
                for row in csv.DictReader(open(os.path.join(self._priors_dir, obs_id+'_'+house_id + '.semframes.csv'))):
                    local2world = np.matrix(str2nparr(row['local2world'])).reshape(4, 4)
                    frame = {'obj_id': row['obj_id'], 'obb': OBB.from_local2world_transform(local2world),
                             'aabb': {'min': row['world_aabb_min'], 'max': row['world_aabb_max']}}
                    house_frames[frame['obj_id']] = frame
                self._semantic_frames[house_id] = house_frames

        print('Grouping observations by categories...')
        groups = {}
        num_obs = len(self._observations)
        for i, (obs_id, house_id) in enumerate(counts_house_ids):
            print(f'Observation in house id={house_id} {i}/{num_obs}')
            house = House(file_dir=os.path.join(house_dir, obs_id + '.json'), include_support_information=False)
            self._objects.init_from_house(house, update_sim=False)
            for observation in self._observations[obs_id].values():
                key = self._objects.get_observation_key(observation)
                key_bin = groups.get(key, [])
                key_bin.append(observation)
                groups[key] = key_bin
        print('Done grouping')

        self._grouped_observations = groups

    def save(self):
        pkl_file = os.path.join(self._priors_dir, self._name + '.priors.pkl.gz')
        utils.pickle_dump_compressed(self._grouped_observations, pkl_file)

    def load(self):
        pkl_file = os.path.join(self._priors_dir, self._name + '.priors.pkl.gz')
        self._grouped_observations = utils.pickle_load_compressed(pkl_file)

    def _get_observations_from_house(self, house):
        if self._verbose:
            print(f'Observations for house id={house.id}...')

        self._objects.reset()
        self._objects.simulator.add_house(house, no_ceil=True, no_floor=True)
        relative_observations = {}  # (node_i.id,node_j.id) -> observation of node_i relative to node_j

        if len(house.rooms) == 0:
            print(f'House id={house.id} has no rooms, skipping.')
            return relative_observations

        for room in house.rooms:
            if self._verbose:
                print(f'Room id={room.id}, roomTypes={room.roomTypes}')
            # prepare node and room metadata
            self._objects.add_object(room)
            nodes = [n for n in room.nodes if n.type == 'Object' and n.valid]
            for o in nodes:
                self._objects.add_object(o)
            room_relative_observations = self._objects.get_relative_observations(room.id, 
                filter_ref_obj=None,
                ignore_categories=list())
            relative_observations.update(room_relative_observations)

            return relative_observations


if __name__ == '__main__':
    module_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='Compute arrangement priors')
    parser.add_argument('--task', type=str, required=True, help='<Required> task [collect|save_pkl]')
    parser.add_argument('--input', type=str, help='house json or directory with house json files')
    parser.add_argument('--priors_dir', type=str, required=True, help='priors data directory')
    parser.add_argument('--house_dir', type=str, help='house data directory')
    args = parser.parse_args()

    rod = RelativeObservationsDatabase(name='suncg_priors', priors_dir=args.priors_dir, verbose=True)

    if os.path.isdir(args.input):
        house_files = sorted(glob.glob(os.path.join(args.input, '**/*.json'), recursive=True))
    else:
        house_files = [args.input]

    if args.task == 'collect':
        for f in house_files:
            house = House(file_dir=f, include_support_information=False)
            prefix = os.path.splitext(os.path.basename(f))[0]
            priors_file = os.path.join(args.priors_dir, prefix + '_' + house.id + '.relpos.csv')
            if os.path.exists(priors_file):
                print(f'Priors already exist at {priors_file}')
            else:
                rod.collect_observations([house])
                rod.save_observations_by_house(prefix)
                rod.clear()

    if args.task == 'save_pkl':
        relpos_files = filter(lambda p: '.relpos.csv' in p, os.listdir(args.input))
        relpos_files = list(map(lambda p: os.path.basename(p).split('.')[0], relpos_files))
        print(relpos_files)
        rod.load_observations(relpos_files, args.house_dir, load_frames=False)
        # reduced = {k: rod.grouped_observations[k] for k in [
        #     ObservationCategory('Bedroom', 'chair', 'desk'), ObservationCategory('Bedroom', 'desk', 'chair'),
        # ]}
        # rod._grouped_observations = reduced
        rod.save()

    print('DONE')
