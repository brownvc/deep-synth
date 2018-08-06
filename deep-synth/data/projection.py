import numpy as np
import math
import scipy.misc as m

class ProjectionGenerator():
    """
    Generates projection between original 3D space and rendered 2D image space
    Given the position of the room
    """
    def __init__(self, room_size_cap=(6.05,4.05,6.05), zpad=0.5, img_size=512):
        """
        See top_down.TopDownView for explanation of room_size, zpad and img_size
        """
        self.room_size_cap = room_size_cap
        self.zpad = zpad
        self.img_size = img_size
        self.xscale = self.img_size / self.room_size_cap[0]
        self.yscale = self.img_size / self.room_size_cap[2]
        self.zscale = 1.0 / (self.room_size_cap[1] + self.zpad)

    def get_projection(self, room):
        """
        Generates projection matrices specific to a room,
        need to be room-specific since every room is located in a different position,
        but they are all rendered centered in the image
        """
        xscale, yscale, zscale = self.xscale, self.yscale, self.zscale

        xshift = -(room.xmin * 0.5 + room.xmax * 0.5 - self.room_size_cap[0] / 2.0)
        yshift = -(room.ymin * 0.5 + room.ymax * 0.5 - self.room_size_cap[2] / 2.0)
        zshift = -room.zmin + self.zpad

        t_scale = np.asarray([[xscale, 0, 0, 0], \
                              [0, zscale, 0, 0], \
                              [0, 0, yscale, 0], \
                              [0, 0, 0, 1]])
        
        t_shift = np.asarray([[1, 0, 0, 0], \
                              [0, 1, 0, 0], \
                              [0, 0, 1, 0], \
                              [xshift, zshift, yshift, 1]])

        t_3To2 = np.dot(t_shift, t_scale)
    
        t_scale = np.asarray([[1/xscale, 0, 0, 0], \
                              [0, 1/zscale, 0, 0], \
                              [0, 0, 1/yscale, 0], \
                              [0, 0, 0, 1]])
        
        t_shift = np.asarray([[1, 0, 0, 0], \
                              [0, 1, 0, 0], \
                              [0, 0, 1, 0], \
                              [-xshift, -zshift, -yshift, 1]])
        
        t_2To3 = np.dot(t_scale, t_shift)

        return Projection(t_3To2, t_2To3, self.img_size)

class Projection():
    def __init__(self, t_2d, t_3d, img_size):
        self.t_2d = t_2d
        self.t_3d = t_3d
        self.img_size = img_size
    
    def to_2d(self, t=None):
        """
        Parameters
        ----------
        t(Matrix or None): transformation matrix of the object
            if None, then returns room projection
        """
        if t is None:
            return self.t_2d
        else:
            return np.dot(t, self.t_2d)

    def to_3d(self, t=None):
        if t is None:
            return self.t_3d
        else:
            return np.dot(t, self.t_3d)

    def get_ortho_parameters(self):
        bottom_left = np.asarray([0,0,0,1])
        top_right = np.asarray([self.img_size,1,self.img_size,1])
        bottom_left = np.dot(bottom_left, self.t_3d)
        top_right = np.dot(top_right, self.t_3d)
        return (bottom_left[0], top_right[0], bottom_left[2], top_right[2], bottom_left[1], top_right[1])
