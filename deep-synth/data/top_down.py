from data import Obj, ProjectionGenerator, Projection
import numpy as np
import math
import scipy.misc as m
from numba import jit
import torch

class TopDownView():
    """
    Take a room, pre-render top-down views
    Of floor, walls and individual objects
    That can be used to generate the multi-channel views used in our pipeline
    """
    def __init__(self, height_cap=4.05, length_cap=6.05, size=512):
        #Padding by 0.05m to avoid problems with boundary cases
        """
        Parameters
        ----------
        height_cap (int): the maximum height (in meters) of rooms allowed, which will be rendered with
            a value of 1 in the depth channel. To separate the floor from empty spaces,
            floors will have a height of 0.5m. See zpad below
        length_cap (int): the maximum length/width of rooms allowed.
        size (int): size of the rendered top-down image

        Returns
        -------
        visualization (Image): a visualization of the rendered room, which is
            simply the superimposition of all the rendered parts
        (floor, wall, nodes) (Triple[torch.Tensor, torch.Tensor, list[torch.Tensor]): 
            rendered invidiual parts of the room, as 2D torch tensors
            this is the part used by the pipeline
        """
        self.size = size
        self.pgen = ProjectionGenerator(room_size_cap=(length_cap, height_cap, length_cap), \
                                        zpad=0.5, img_size=size)

    def render(self, room):
        projection = self.pgen.get_projection(room)
        
        visualization = np.zeros((self.size,self.size))
        nodes = []

        for node in room.nodes:
            modelId = node.modelId #Camelcase due to original json

            t = np.asarray(node.transform).reshape(4,4)

            o = Obj(modelId)
            t = projection.to_2d(t)
            o.transform(t)
            
            t = projection.to_2d()
            bbox_min = np.dot(np.asarray([node.xmin, node.zmin, node.ymin, 1]), t)
            bbox_max = np.dot(np.asarray([node.xmax, node.zmax, node.ymax, 1]), t)
            xmin = math.floor(bbox_min[0])
            ymin = math.floor(bbox_min[2])
            xsize = math.ceil(bbox_max[0]) - xmin + 1
            ysize = math.ceil(bbox_max[2]) - ymin + 1

            description = {}
            description["modelId"] = modelId
            description["transform"] = node.transform
            description["bbox_min"] = bbox_min
            description["bbox_max"] = bbox_max
            
            #Since it is possible that the bounding box information of a room
            #Was calculated without some doors/windows
            #We need to handle these cases
            if ymin < 0: 
                ymin = 0
            if xmin < 0: 
                xmin = 0

            rendered = self.render_object(o, xmin, ymin, xsize, ysize, self.size)
            description["height_map"] = torch.from_numpy(rendered).float()
            
            tmp = np.zeros((self.size, self.size))
            tmp[xmin:xmin+rendered.shape[0],ymin:ymin+rendered.shape[1]] = rendered
            visualization += tmp

            nodes.append(description)
        
        #Render the floor
        o = Obj(room.modelId+"f", room.house_id, is_room=True)
        t = projection.to_2d()
        o.transform(t)
        floor = self.render_object(o, 0, 0, self.size, self.size, self.size)
        visualization += floor
        floor = torch.from_numpy(floor).float()
    
        #Render the walls
        o = Obj(room.modelId+"w", room.house_id, is_room=True)
        t = projection.to_2d()
        o.transform(t)
        wall = self.render_object(o, 0, 0, self.size, self.size, self.size)
        visualization += wall
        wall = torch.from_numpy(wall).float()
        
        return (visualization, (floor, wall, nodes))
    
    @staticmethod
    @jit(nopython=True)
    def render_object_helper(triangles, xmin, ymin, xsize, ysize, img_size):
        result = np.zeros((img_size, img_size), dtype=np.float32)
        N, _, _ = triangles.shape

        for triangle in range(N):
            x0,z0,y0 = triangles[triangle][0]
            x1,z1,y1 = triangles[triangle][1]
            x2,z2,y2 = triangles[triangle][2]
            a = -y1*x2 + y0*(-x1+x2) + x0*(y1-y2) + x1*y2
            if a != 0:
                for i in range(max(0,math.floor(min(x0,x1,x2))), \
                               min(img_size,math.ceil(max(x0,x1,x2)))):
                    for j in range(max(0,math.floor(min(y0,y1,y2))), \
                                   min(img_size,math.ceil(max(y0,y1,y2)))):
                        x = i+0.5
                        y = j+0.5
                        s = (y0*x2 - x0*y2 + (y2-y0)*x + (x0-x2)*y)/a
                        t = (x0*y1 - y0*x1 + (y0-y1)*x + (x1-x0)*y)/a
                        if s < 0 and t < 0:
                            s = -s
                            t = -t
                        if 0 < s < 1 and 0 < t < 1 and s + t <= 1:
                            height = z0 *(1-s-t) + z1*s + z2*t
                            result[i][j] = max(result[i][j], height)

        return result[xmin:xmin+xsize, ymin:ymin+ysize]
    
    @staticmethod
    def render_object(o, xmin, ymin, xsize, ysize, img_size):
        """
        Render a cropped top-down view of object
        
        Parameters
        ----------
        o (list[triple]): object to be rendered, represented as a triangle mesh
        xmin, ymin (int): min coordinates of the bounding box containing the object,
            with respect to the full image
        xsize, ysze (int); size of the bounding box containing the object
        img_size (int): size of the full image
        """
        triangles = np.asarray(list(o.get_triangles()), dtype=np.float32)
        return TopDownView.render_object_helper(triangles, xmin, ymin, xsize, ysize, img_size)
    
    @staticmethod
    @jit(nopython=True)
    def render_object_full_size_helper(triangles, size):
        result = np.zeros((size, size), dtype=np.float32)
        N, _, _ = triangles.shape

        for triangle in range(N):
            x0,z0,y0 = triangles[triangle][0]
            x1,z1,y1 = triangles[triangle][1]
            x2,z2,y2 = triangles[triangle][2]
            a = -y1*x2 + y0*(-x1+x2) + x0*(y1-y2) + x1*y2
            if a != 0:
                for i in range(max(0,math.floor(min(x0,x1,x2))), \
                               min(size,math.ceil(max(x0,x1,x2)))):
                    for j in range(max(0,math.floor(min(y0,y1,y2))), \
                                   min(size,math.ceil(max(y0,y1,y2)))):
                        x = i+0.5
                        y = j+0.5
                        s = (y0*x2 - x0*y2 + (y2-y0)*x + (x0-x2)*y)/a
                        t = (x0*y1 - y0*x1 + (y0-y1)*x + (x1-x0)*y)/a
                        if s < 0 and t < 0:
                            s = -s
                            t = -t
                        if 0 < s < 1 and 0 < t < 1 and s + t <= 1:
                            height = z0 *(1-s-t) + z1*s + z2*t
                            result[i][j] = max(result[i][j], height)
        
        return result

    @staticmethod
    def render_object_full_size(o, size):
        """
        Render a full-sized top-down view of the object, see render_object
        """
        triangles = np.asarray(list(o.get_triangles()), dtype=np.float32)
        return TopDownView.render_object_full_size_helper(triangles, size)
