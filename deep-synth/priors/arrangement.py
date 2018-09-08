#!/usr/bin/env python3

import argparse
import math
import os
import utils

import numpy as np


from data import House, DatasetToJSON
from math_utils import Transform
from priors.observations import ObjectCollection, ObservationCategory, RelativeObservation
from priors.pairwise import PairwiseArrangementPrior  # needed for pickle laod
from pyquaternion import Quaternion
from random import random


class ArrangementPriors:
    """
    A collection of object arrangement priors for estimating probability of an arrangement and sampling arrangements
    """
    def __init__(self, priors_file, w_dist=1.0, w_clearance=1.0, w_same_category=1.0, w_closest=1.0, w_orientation=1.0):
        self._priors = utils.pickle_load_compressed(priors_file)
        self._num_priors = 0.
        self._num_observations = 0.
        self._w_dist = w_dist
        self._w_clearance = w_clearance
        self._w_same_category = w_same_category
        self._w_closest = w_closest
        self._w_orientation = w_orientation
        for p_key, p in self._priors.items():
            self._priors[p_key] = p
            self._num_priors += 1
            self._num_observations += p.num_observations
            p.trim()

    @property
    def num_observations(self):
        """
        Return total number of observations in this ArrangementPriors set
        """
        return self._num_observations

    @property
    def pairwise_priors(self):
        """
        Return all pairwise priors
        """
        return self._priors

    def get(self, category_key):
        """
        Return pairwise prior for given category_key
        """
        return self._priors[category_key]

    def pairwise_occurrence_prob(self, category_key):
        """
        Return probability of pairwise observation with given category_key
        """
        norm = self._num_observations + self._num_priors  # add one smoothing
        if category_key in self._priors:
            return (self._priors[category_key].num_observations + 1) / norm
        else:
            return 1. / norm

    def log_prob(self, pairwise_observations_by_key):
        """
        Return log probability of given set of pairwise relative observations under arrangement priors
        """
        sum_lp = 0

        for key in pairwise_observations_by_key:
            # get observations and lp for this pairwise prior
            observations = pairwise_observations_by_key[key]

            # parameterize observations
            records = map(lambda o: o.parameterize(scheme='offsets_angles'), observations)
            x = np.stack(records, axis=0)
            # print(observations)
            # print(x)

            if key not in self._priors:
                lp = math.log(0.5)
            else:  # evaluate each observation against relevant prior
                prior = self._priors[key]
                if key.obj_category == 'room':  # distance and angle to closest point on wall
                    lp = self._w_clearance * prior.log_prob_closest(x)
                elif key.obj_category == key.ref_obj_category:  # same category instance
                    lp = self._w_same_category * self._w_closest * prior.log_prob_closest(x)
                else:  # all other categories
                    lp = self._w_closest * prior.log_prob_closest(x)
                lp += self._w_orientation * prior.log_prob_orientation(x)

            if self._w_dist > 0:
                dist = np.sum(np.square(x[:, 2:4]), axis=1, keepdims=True)
                lp = lp - (self._w_dist * dist/2)  # log(exp(-x^2/2))

            pairwise_occ_p = self.pairwise_occurrence_prob(key)
            lp = lp * pairwise_occ_p

            # sum up lps
            sum_lp += np.sum(lp, axis=0)

        if hasattr(sum_lp, 'shape'):
            sum_lp = np.sum(sum_lp)
        return sum_lp


class ArrangementGreedySampler:
    """
    Iterative optimization of object arrangements using greedy sampling of ArrangementPriors
    """
    def __init__(self, arrangement_priors, num_angle_divisions=8, num_pairwise_priors=-1, sim_mode='direct'):
        self._objects = ObjectCollection(sim_mode=sim_mode)
        self.room_id = None
        self._priors = arrangement_priors
        self._num_angle_divisions = num_angle_divisions
        self._num_pairwise_priors = num_pairwise_priors

    @property
    def objects(self):
        return self._objects

    def init(self, house, only_architecture=True, room_id=None):
        if not room_id:
            room_id = house.rooms[0].id
        self._objects.init_from_room(house, room_id, only_architecture=only_architecture)
        self.room_id = room_id

    def log_prob(self, filter_ref_obj=None, ignore_categories=list()):
        observations = self._objects.get_relative_observations(self.room_id, filter_ref_obj=filter_ref_obj,
                                                               ignore_categories=ignore_categories)
        observations_by_key = {}

        # top_k_prior_categories = None
        # if filter_ref_obj and num_pairwise_priors specified, filter observations to only those in top k priors
        # if filter_ref_obj and self._num_pairwise_priors > 0:
        #     category = self._objects.category(filter_ref_obj.modelId, scheme='final')
        #     priors = list(filter(lambda p: p.ref_obj_category == category, self._priors.pairwise_priors))
        #     k = min(self._num_pairwise_priors, len(priors))
        #     priors = list(sorted(priors, key=lambda p: self._priors.pairwise_occurrence_log_prob(p)))[-k:]
        #     top_k_prior_categories = set(map(lambda p: p.obj_category, priors))

        for o in observations.values():
            # only pairwise observations in which filter_ref_obj is the reference
            if filter_ref_obj and o.ref_id != filter_ref_obj.id:
                continue
            key = self._objects.get_observation_key(o)
            os_key = observations_by_key.get(key, [])
            os_key.append(o)
            observations_by_key[key] = os_key
        return self._priors.log_prob(observations_by_key)

    def get_candidate_transform(self, node, max_iterations=100):
        num_checks = 0
        zmin = self._objects.room.zmin
        while True:
            num_checks += 1
            p = self._objects.room.obb.sample()
            ray_from = [p[0], zmin - .5, p[2]]
            ray_to = [p[0], zmin + .5, p[2]]
            intersection = ags._objects.simulator.ray_test(ray_from, ray_to)
            if intersection.id == self.room_id + 'f' or num_checks > max_iterations:
                break
        xform = Transform()
        xform.set_translation([p[0], zmin + .1, p[2]])
        angle = random() * 2 * math.pi
        angular_resolution = 2 * math.pi / self._num_angle_divisions
        angle = round(angle / angular_resolution) * angular_resolution
        xform.set_rotation(radians=angle)
        return xform

    def sample_placement(self, node, n_samples, houses_log=None, max_attempts_per_sample=10, ignore_categories=list(),
                         collision_threshold=0):
        """
        Sample placement for given node
        """
        self._objects.add_object(node)
        max_lp = -np.inf
        max_xform = None
        max_house = None
        num_noncolliding_samples = 0
        for i in range(max_attempts_per_sample*n_samples):
            xform = self.get_candidate_transform(node)
            self._objects.update(node, xform=xform, update_sim=True)
            collisions = self._objects.get_collisions(obj_id_a=node.id)
            # print(f'i={i}, samples_so_far={num_noncolliding_samples}, n_samples={n_samples},'
            #       f'max_attempts_per_sample={max_attempts_per_sample}')
            if collision_threshold > 0:
                if min(collisions.values(), key=lambda c: c.distance).distance < -collision_threshold:
                    continue
            elif len(collisions) > 0:
                continue
            lp = self.log_prob(filter_ref_obj=node, ignore_categories=ignore_categories)
            print(f'lp={lp}')
            if lp > max_lp:
                max_xform = xform
                max_lp = lp
                if houses_log is not None:
                    max_house = self._objects.as_house()
            num_noncolliding_samples += 1
            if num_noncolliding_samples == n_samples:
                break
        if houses_log is not None:
            houses_log.append(max_house)
        self._objects.update(node, xform=max_xform, update_sim=True)

    def placeable_objects_sorted_by_size(self, house):
        objects = []
        fixed_objects = []
        for n in house.levels[0].nodes:
            if n.type != 'Object':
                continue
            category = self._objects.category(n.modelId, scheme='final')
            if category == 'door' or category == 'window':
                fixed_objects.append(n)
                continue
            objects.append(n)

        for o in objects:
            # to_delete = []
            # for k in o.__dict__:
            #     if k not in ['id', 'modelId', 'transform', 'type', 'valid', 'bbox']:
            #         to_delete.append(k)
            # for k in to_delete:
            #     delattr(o, k)
            dims = self._objects._object_data.get_aligned_dims(o.modelId)
            o.volume = dims[0] * dims[1] * dims[2]
        objects = list(sorted(objects, key=lambda x: x.volume, reverse=True))
        return objects, fixed_objects


if __name__ == '__main__':
    module_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='Arrangement model')
    parser.add_argument('--task', type=str, default='arrange', help='task [arrange]')
    parser.add_argument('--input', type=str, required=True, help='input scene to arrange')
    parser.add_argument('--output_dir', type=str, required=True, help='output directory to save arranged house states')
    parser.add_argument('--priors_dir', type=str, required=True, help='base directory to load priors')
    parser.add_argument('--only_final_state', action='store_true', help='save only final')
    parser.add_argument('--sim_mode', dest='sim_mode', default='direct', help='sim server [gui|direct|shared_memory]')
    parser.add_argument('--restart_sim_every_round', action='store_true', help='restart sim after every object placed')
    parser.add_argument('--max_objects', type=int, default=np.inf, help='maximum number of objects to place')
    parser.add_argument('--num_samples_per_object', type=int, default=100, help='placement samples per object')
    parser.add_argument('--ignore_doors_windows', action='store_true', help='ignore door and window priors')
    parser.add_argument('--collision_threshold', type=float, default=0, help='how much collision penetration to allow')
    args = parser.parse_args()

    if args.task == 'arrange':
        # initialize arrangement priors and sampler
        filename = os.path.join(args.priors_dir, f'priors.pkl.gz')
        ap = ArrangementPriors(priors_file=filename,
                               w_dist=0.0,
                               w_clearance=1.0,
                               w_closest=1.0,
                               w_orientation=1.0,
                               w_same_category=1.0)
        ags = ArrangementGreedySampler(arrangement_priors=ap, sim_mode=args.sim_mode)

        # load house and set architecture
        original_house = House(file_dir=args.input, include_support_information=False)
        ags.init(original_house, only_architecture=True)

        # split into placeable objects and fixed portals (doors, windows)
        placeables, fixed_objects = ags.placeable_objects_sorted_by_size(original_house)

        # add fixed objects (not to be arranged)
        for n in fixed_objects:
            ags.objects.add_object(n)

        # place objects one-by-one
        houses_states = []
        num_placeables = len(placeables)
        for (i, n) in enumerate(placeables):
            print(f'Placing {i+1}/{num_placeables}')
            if args.sim_mode == 'gui' and i < 2:  # center view on first couple of placements if gui_mode
                ags.objects.simulator.set_gui_rendering(enabled=True)
            houses_log = None if args.only_final_state else houses_states
            ags.sample_placement(node=n, n_samples=args.num_samples_per_object, houses_log=houses_log,
                                 ignore_categories=['window', 'door'] if args.ignore_doors_windows else [])
            if args.restart_sim_every_round:
                ags.objects.simulator.connect()
                ags.objects.reinit_simulator()
            if i+1 >= args.max_objects:
                break

        # append final house state
        house_final = ags._objects.as_house()
        houses_states.append(house_final)

        # save out states
        input_id = os.path.splitext(os.path.basename(args.input))[0]
        output_dir = os.path.join(args.output_dir, input_id)
        print(f'Saving results to {output_dir}...')
        dj = DatasetToJSON(dest=output_dir)
        empty_is = []
        for i, h in enumerate(houses_states):
            if h:
                h.trim()
            else:
                print(f'Warning: empty house state for step {i}, ignoring...')
        houses_states = list(filter(lambda h: h is not None, houses_states))
        for (i, s) in enumerate(dj.step(houses_states)):
            print(f'Saved {i}')

    print('DONE')
