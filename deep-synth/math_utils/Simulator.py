import os
import numpy as np
import pybullet as p
import subprocess as sp
import time

from pyquaternion import Quaternion
from collections import namedtuple
from itertools import groupby
from math_utils import Transform
import utils

# handle to a simulated rigid body
Body = namedtuple('Body', ['id', 'bid', 'vid', 'cid', 'static'])

# a body-body contact record
Contact = namedtuple(
    'Contact', ['flags', 'idA', 'idB', 'linkIndexA', 'linkIndexB',
                'positionOnAInWS', 'positionOnBInWS',
                'contactNormalOnBInWS', 'distance', 'normalForce']
)

# ray intersection record
Intersection = namedtuple('Intersection', ['id', 'linkIndex', 'ray_fraction', 'position', 'normal'])


class Simulator:
    def __init__(self, mode='direct', bullet_server_binary=None, data_dir_base=None, verbose=False):
        self._mode = mode
        self._verbose = verbose
        module_dir = os.path.dirname(os.path.abspath(__file__))
        data_root_dir = utils.get_data_root_dir()
        if data_dir_base:
            self._data_dir_base = data_dir_base
        else:
            self._data_dir_base = os.path.join(data_root_dir, 'suncg_data')
        if bullet_server_binary:
            self._bullet_server_binary = bullet_server_binary
        else:
            self._bullet_server_binary = os.path.join(module_dir, '..', 'bullet_shared_memory_server')
        self._obj_id_to_body = {}
        self._bid_to_body = {}
        self._pid = None
        self._bullet_server = None
        self.connect()

    def connect(self):
        # disconnect and kill existing servers
        if self._pid:
            p.disconnect(physicsClientId=self._pid)
            self._pid = None
        if self._bullet_server:
            print(f'Restarting by killing bullet server pid={self._bullet_server.pid}...')
            self._bullet_server.kill()
            time.sleep(1)  # seems necessary to prevent deadlock on re-connection attempt
            self._bullet_server = None

        # reconnect to appropriate server type
        if self._mode == 'gui':
            self._pid = p.connect(p.GUI)
        elif self._mode == 'direct':
            self._pid = p.connect(p.DIRECT)
        elif self._mode == 'shared_memory':
            print(f'Restarting bullet server process...')
            self._bullet_server = sp.Popen([self._bullet_server_binary])
            time.sleep(1)  # seems necessary to prevent deadlock on connection attempt
            self._pid = p.connect(p.SHARED_MEMORY)
        else:
            raise RuntimeError(f'Unknown simulator server mode={self._mode}')

        # reset and initialize gui if needed
        p.resetSimulation(physicsClientId=self._pid)
        if self._mode == 'gui':
            p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1, physicsClientId=self._pid)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=self._pid)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=self._pid)
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=self._pid)
            # disable rendering during loading -> much faster
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=self._pid)

    def __del__(self):
        if self._bullet_server:
            print(f'Process terminating. Killing bullet server pid={self._bullet_server.pid}...')
            self._bullet_server.kill()

    def run(self):
        if self._mode == 'gui':
            # infinite ground plane and gravity
            # plane_cid = p.createCollisionShape(p.GEOM_PLANE, planeNormal=[0, 1, 0])
            # plane_bid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=plane_cid)
            p.setGravity(0, -10, 0, physicsClientId=self._pid)
            p.setRealTimeSimulation(1, physicsClientId=self._pid)
            self.set_gui_rendering(enabled=True)
            while True:
                contacts = self.get_contacts(include_collision_with_static=True)
                if self._verbose:
                    print(f'#contacts={len(contacts)}, contact_pairs={contacts.keys()}')
        else:
            self.step()
            contacts = self.get_contacts(include_collision_with_static=True)
            if self._verbose:
                for (k, c) in contacts.items():
                    cp = self.get_closest_point(obj_id_a=c.idA, obj_id_b=c.idB)
                    print(f'contact pair={k} record={cp}')
                print(f'#contacts={len(contacts)}, contact_pairs={contacts.keys()}')

    def set_gui_rendering(self, enabled):
        if not self._mode == 'gui':
            return False
        center = np.array([0.0, 0.0, 0.0])
        num_obj = 0
        for obj_id in self._obj_id_to_body.keys():
            pos, _ = self.get_state(obj_id)
            if not np.allclose(pos, [0, 0, 0]):  # try to ignore room object "identity" transform
                num_obj += 1
                center += pos
        center /= num_obj
        p.resetDebugVisualizerCamera(cameraDistance=10.0,
                                     cameraYaw=45.0,
                                     cameraPitch=-30.0,
                                     cameraTargetPosition=center,
                                     physicsClientId=self._pid)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1 if enabled else 0, physicsClientId=self._pid)
        return enabled

    def add_mesh(self, obj_id, obj_file, transform, vis_mesh_file=None, static=False):
        if static:
            cid = p.createCollisionShape(p.GEOM_MESH, fileName=obj_file, meshScale=transform.scale,
                                         flags=p.GEOM_FORCE_CONCAVE_TRIMESH, physicsClientId=self._pid)
        else:
            cid = p.createCollisionShape(p.GEOM_MESH, fileName=obj_file, meshScale=transform.scale,
                                         physicsClientId=self._pid)
        vid = -1
        if vis_mesh_file:
            vid = p.createVisualShape(p.GEOM_MESH, fileName=vis_mesh_file, meshScale=transform.scale,
                                      physicsClientId=self._pid)
        rot_q = np.roll(transform.rotation.elements, -1)  # w,x,y,z -> x,y,z,w (which pybullet expects)
        mass = 0 if static else 1
        bid = p.createMultiBody(baseMass=mass,
                                baseCollisionShapeIndex=cid,
                                baseVisualShapeIndex=vid,
                                basePosition=transform.translation,
                                baseOrientation=rot_q,
                                physicsClientId=self._pid)
        body = Body(id=obj_id, bid=bid, vid=vid, cid=cid, static=static)
        self._obj_id_to_body[obj_id] = body
        self._bid_to_body[bid] = body
        return body

    def add_box(self, obj_id, half_extents, transform, static=False):
        cid = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=self._pid)
        rot_q = np.roll(transform.rotation.elements, -1)  # w,x,y,z -> x,y,z,w (which pybullet expects)
        mass = 0 if static else 1
        bid = p.createMultiBody(baseMass=mass,
                                baseCollisionShapeIndex=cid,
                                basePosition=transform.translation,
                                baseOrientation=rot_q,
                                physicsClientId=self._pid)
        body = Body(id=obj_id, bid=bid, vid=-1, cid=cid, static=static)
        self._obj_id_to_body[obj_id] = body
        self._bid_to_body[bid] = body
        return body

    # House-specific functions
    def add_object(self, node, create_vis_mesh=False, static=False):
        model_id = node.modelId.replace('_mirror', '')  # TODO: need to otherwise account for mirror?
        object_dir = os.path.join(self._data_dir_base, 'object')
        basename = f'{object_dir}/{model_id}/{model_id}'
        vis_obj_filename = f'{basename}.obj' if create_vis_mesh else None
        col_obj_filename = f'{basename}.vhacd.obj'
        if not os.path.exists(col_obj_filename):
            print('WARNING: collision mesh {col_obj_filename} unavailable, using visual mesh instead.')
            col_obj_filename = f'{basename}.obj'
        return self.add_mesh(obj_id=node.id, obj_file=col_obj_filename, transform=Transform.from_node(node),
                             vis_mesh_file=vis_obj_filename, static=static)

    def add_wall(self, node):
        h = node['height']
        p0 = np.transpose(np.matrix(node['points'][0]))
        p1 = np.transpose(np.matrix(node['points'][1]))
        c = (p0 + p1) * 0.5
        c[1] = h * 0.5
        dp = p1 - p0
        dp_l = np.linalg.norm(dp)
        dp = dp / dp_l
        angle = np.arccos(dp[0])
        rot_q = Quaternion(axis=[0, 1, 0], radians=angle)
        half_extents = np.array([dp_l, h, node['depth']]) * 0.5
        return self.add_box(obj_id=node['id'], half_extents=half_extents,
                            transform=Transform(translation=c, rotation=rot_q), static=True)

    def add_room(self, node, wall=True, floor=True, ceiling=False):
        def add_architecture(n, obj_file, suffix):
            return self.add_mesh(obj_id=n.id + suffix, obj_file=obj_file, transform=Transform(), vis_mesh_file=None,
                                 static=True)
        room_id = node.modelId
        room_dir = os.path.join(self._data_dir_base, 'room')
        basename = f'{room_dir}/{node.house_id}/{room_id}'
        body_ids = []
        if wall:
            body_wall = add_architecture(node, f'{basename}w.obj', '')  # treat walls as room (=room.id, no suffix)
            body_ids.append(body_wall)
        if floor:
            body_floor = add_architecture(node, f'{basename}f.obj', 'f')
            body_ids.append(body_floor)
        if ceiling:
            body_ceiling = add_architecture(node, f'{basename}c.obj', 'c')
            body_ids.append(body_ceiling)
        return body_ids

    def add_house(self, house, no_walls=False, no_ceil=False, no_floor=False, use_separate_walls=False, static=False):
        for node in house.nodes:
            if not node.valid:
                continue
            if not hasattr(node, 'body'):
                node.body = None
            if node.type == 'Object':
                node.body = self.add_object(node, static=static)
            if node.type == 'Room':
                ceil = False if no_ceil else not (hasattr(node, 'hideCeiling') and node.hideCeiling == 1)
                wall = False if (no_walls or use_separate_walls) else not (hasattr(node, 'hideWalls') and node.hideWalls == 1)
                floor = False if no_floor else not (hasattr(node, 'hideFloor') and node.hideFloor == 1)
                node.body = self.add_room(node, wall=wall, floor=floor, ceiling=ceil)
            if node.type == 'Box':
                half_widths = list(map(lambda x: 0.5 * x, node.dimensions))
                node.body = self.add_box(obj_id=node.id, half_extents=half_widths, transform=Transform.from_node(node),
                                         static=static)
        if use_separate_walls and not no_walls:
            for wall in house.walls:
                wall['body'] = self.add_wall(wall)

    def add_house_room_only(self, house, room, no_walls=False, no_ceil=True, no_floor=False, use_separate_walls=False,
                            only_architecture=False, static=False):
        #walls, ceil, floor logic not fleshed out due to current limited use case
        room_node = [node for node in house.nodes if node.id == room.id]
        if len(room_node) < 1:
            raise Exception("Missing Room")
        if only_architecture:
            house.nodes = room_node
        else:
            house.nodes = [node for node in room.nodes]
            house.nodes.append(room_node[0])

        for node in house.nodes:
            if not node.valid:
                continue
            if not hasattr(node, 'body'):
                node.body = None
            if node.type == 'Object':
                node.body = self.add_object(node, static=static)
            if node.type == 'Room':
                ceil = False if no_ceil else not (hasattr(node, 'hideCeiling') and node.hideCeiling == 1)
                wall = False if (no_walls or use_separate_walls) else not (hasattr(node, 'hideWalls') and node.hideWalls == 1)
                floor = False if no_floor else not (hasattr(node, 'hideFloor') and node.hideFloor == 1)
                node.body = self.add_room(node, wall=wall, floor=floor, ceiling=ceil)
            if node.type == 'Box':
                half_widths = list(map(lambda x: 0.5 * x, node.dimensions))
                node.body = self.add_box(obj_id=node.id, half_extents=half_widths, transform=Transform.from_node(node),
                                         static=static)
        if use_separate_walls and not no_walls:
            for wall in house.walls:
                wall['body'] = self.add_wall(wall)

    def remove(self, obj_id):
        body = self._obj_id_to_body[obj_id]
        p.removeBody(bodyUniqueId=body.bid, physicsClientId=self._pid)
        del self._obj_id_to_body[obj_id]
        del self._bid_to_body[body.bid]

    def set_state(self, obj_id, position, rotation_q):
        body = self._obj_id_to_body[obj_id]
        rot_q = np.roll(rotation_q.elements, -1)  # w,x,y,z -> x,y,z,w (which pybullet expects)
        p.resetBasePositionAndOrientation(bodyUniqueId=body.bid, posObj=position, ornObj=rot_q,
                                          physicsClientId=self._pid)

    def get_state(self, obj_id):
        body = self._obj_id_to_body[obj_id]
        pos, q = p.getBasePositionAndOrientation(bodyUniqueId=body.bid, physicsClientId=self._pid)
        rotation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        return pos, rotation

    def step(self):
        p.stepSimulation(physicsClientId=self._pid)

    def reset(self):
        p.resetSimulation(physicsClientId=self._pid)
        self._obj_id_to_body = {}
        self._bid_to_body = {}

    def get_closest_point(self, obj_id_a, obj_id_b, max_distance=np.inf):
        """
        Return record with distance between closest points between pair of nodes if within max_distance or None.
        """
        bid_a = self._obj_id_to_body[obj_id_a].bid
        bid_b = self._obj_id_to_body[obj_id_b].bid
        cps = p.getClosestPoints(bodyA=bid_a, bodyB=bid_b, distance=max_distance, physicsClientId=self._pid)
        cp = None
        if len(cps) > 0:
            closest_points = self._convert_contacts(cps)
            cp = min(closest_points, key=lambda x: x.distance)
        del cps  # NOTE force garbage collection of pybullet objects
        return cp

    def get_contacts(self, obj_id_a=None, obj_id_b=None, only_closest_contact_per_pair=True,
                     include_collision_with_static=True):
        """
        Return all current contacts. When include_collision_with_statics is true, include contacts with static bodies
        """
        bid_a = self._obj_id_to_body[obj_id_a].bid if obj_id_a else -1
        bid_b = self._obj_id_to_body[obj_id_b].bid if obj_id_b else -1
        cs = p.getContactPoints(bodyA=bid_a, bodyB=bid_b, physicsClientId=self._pid)
        contacts = self._convert_contacts(cs)
        del cs  # NOTE force garbage collection of pybullet objects

        if not include_collision_with_static:
            def not_contact_with_static(c):
                static_a = self._obj_id_to_body[c.idA].static
                static_b = self._obj_id_to_body[c.idB].static
                return not static_a and not static_b
            contacts = filter(not_contact_with_static, contacts)
            # print(f'#all_contacts={len(all_contacts)} to #non_static_contacts={len(non_static_contacts)}')

        if only_closest_contact_per_pair:
            def bid_pair_key(x):
                return str(x.idA) + '_' + str(x.idB)
            contacts = sorted(contacts, key=bid_pair_key)
            min_dist_contact_by_pair = {}
            for k, g in groupby(contacts, key=bid_pair_key):
                min_dist_contact = min(g, key=lambda x: x.distance)
                min_dist_contact_by_pair[k] = min_dist_contact
            contacts = min_dist_contact_by_pair.values()

        # convert into dictionary of form (id_a, id_b) -> Contact
        contacts_dict = {}
        for c in contacts:
            key = (c.idA, c.idB)
            contacts_dict[key] = c

        return contacts_dict

    def _convert_contacts(self, contacts):
        out = []
        for c in contacts:
            bid_a = c[1]
            bid_b = c[2]
            if bid_a not in self._bid_to_body or bid_b not in self._bid_to_body:
                continue
            id_a = self._bid_to_body[bid_a].id
            id_b = self._bid_to_body[bid_b].id
            o = Contact(flags=c[0], idA=id_a, idB=id_b, linkIndexA=c[3], linkIndexB=c[4],
                        positionOnAInWS=c[5], positionOnBInWS=c[6], contactNormalOnBInWS=c[7],
                        distance=c[8], normalForce=c[9])
            out.append(o)
        return out

    def ray_test(self, from_pos, to_pos):
        hit = p.rayTest(rayFromPosition=from_pos, rayToPosition=to_pos, physicsClientId=self._pid)
        intersection = Intersection._make(*hit)
        del hit  # NOTE force garbage collection of pybullet objects
        if intersection.id >= 0:  # if intersection, replace bid with id
            intersection = intersection._replace(id=self._bid_to_body[intersection.id].id)
        return intersection


if __name__ == '__main__':
    from data import House
    sim = Simulator(mode='gui', verbose=True)
    h = House(id_='d119e6e0bd567d923aea774c2a984bf0', include_arch_information=False)
    # h = House(index=5)
    sim.add_house(h, no_walls=False, no_ceil=True, use_separate_walls=False, static=False)
    sim.run()
