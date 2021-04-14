import csv
import pandas as pd
import bpy
import numpy as np
from matplotlib import cm
import math
import random
import array
import json
import jsonpickle
from json import JSONEncoder
from mathutils import Vector
import sys
import time
from scipy.spatial import ConvexHull


def screen_to_cam(screen_coords, z):
    x = -screen_coords[0] * z
    y = -screen_coords[1] * z
    return [x, y, z]


def compute_screen_coordinates(cam_coords):
    screen_coords = []
    screen_coords.append(cam_coords[0][0] / -cam_coords[2][0])
    screen_coords.append(cam_coords[1][0] / -cam_coords[2][0])
 
    return screen_coords


#----------------------------------------------------------------------------------------#

# From creotiv/computer_vision

import numpy as np
import cv2
import math
from scipy.spatial.distance import cdist, cosine
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt


class ShapeContext(object):

    def __init__(self, nbins_r=5, nbins_theta=12, r_inner=0.1250, r_outer=2.0):
        # number of radius zones
        self.nbins_r = nbins_r
        # number of angles zones
        self.nbins_theta = nbins_theta
        # maximum and minimum radius
        self.r_inner = r_inner
        self.r_outer = r_outer

    def _hungarian(self, cost_matrix):
        """
            Here we are solving task of getting similar points from two paths
            based on their cost matrixes. 
            This algorithm has dificulty O(n^3)
            return total modification cost, indexes of matched points
        """
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        total = cost_matrix[row_ind, col_ind].sum()
        indexes = zip(row_ind.tolist(), col_ind.tolist())
        return total, indexes
    
    def get_points_from_img(self, image, simpleto=100):
        """
            This is much faster version of getting shape points algo.
            It's based on cv2.findContours algorithm, which is basically return shape points
            ordered by curve direction. So it's gives better and faster result
        """
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        cnts = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        points = np.array(cnts[1][0]).reshape((-1, 2))
        if len(cnts[1]) > 1:
            points = np.concatenate([points, np.array(cnts[1][1]).reshape((-1, 2))], axis=0)
        points = points.tolist()
        step = len(points) / simpleto
        points = [points[i] for i in range(0, len(points), step)][:simpleto]
        if len(points) < simpleto:
            points = points + [[0, 0]] * (simpleto - len(points))
        return points

    '''def get_points_from_img(self, image, threshold=50, simpleto=100, radius=2):
        """
            That is not very good algorithm of choosing path points, but it will work for our case.
            Idea of it is just to create grid and choose points that on this grid.
        """
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dst = cv2.Canny(image, threshold, threshold * 3, 3)
        py, px = np.gradient(image)
        # px, py gradients maps shape can be smaller then input image shape
        points = [index for index, val in np.ndenumerate(dst)
                  if val == 255 and index[0] < py.shape[0] and index[1] < py.shape[1]]
        h, w = image.shape
        _radius = radius
        while len(points) > simpleto:
            newpoints = points
            xr = range(0, w, _radius)
            yr = range(0, h, _radius)
            for p in points:
                if p[0] not in yr and p[1] not in xr:
                    newpoints.remove(p)
                    if len(points) <= simpleto:
                        T = np.zeros((simpleto, 1))
                        for i, (y, x) in enumerate(points):
                            radians = math.atan2(py[y, x], px[y, x])
                            T[i] = radians + 2 * math.pi * (radians < 0)
                        return points, np.asmatrix(T)
            _radius += 1
        T = np.zeros((simpleto, 1))
        for i, (y, x) in enumerate(points):
            radians = math.atan2(py[y, x], px[y, x])
            T[i] = radians + 2 * math.pi * (radians < 0)
        return points, np.asmatrix(T)'''

    def _cost(self, hi, hj):
        cost = 0
        for k in range(self.nbins_theta * self.nbins_r):
            if (hi[k] + hj[k]):
                cost += ((hi[k] - hj[k])**2) / (hi[k] + hj[k])

        return cost * 0.5

    def cost_by_paper(self, P, Q, qlength=None):
        p, _ = P.shape
        p2, _ = Q.shape
        d = p2
        if qlength:
            d = qlength
        C = np.zeros((p, p2))
        for i in range(p):
            for j in range(p2):
                C[i, j] = self._cost(Q[j] / d, P[i] / p)

        return C

    def compute(self, points_raw):
        """
          Here we are computing shape context descriptor
        """
        t_points = len(points_raw)
        points = None

        for ind in range(t_points):
            if points is None:
                points = np.array([points_raw[ind].co])
            else:
                points = np.append(points, [points_raw[ind].co], axis = 0)

        # getting euclidian distance
        r_array = cdist(points, points)
        # getting two points with maximum distance to norm angle by them
        # this is needed for rotation invariant feature
        am = r_array.argmax()
        max_points = [am / t_points, am % t_points]
        # normalizing
        r_array_n = r_array / r_array.mean()
        # create log space
        r_bin_edges = np.logspace(np.log10(self.r_inner), np.log10(self.r_outer), self.nbins_r)
        r_array_q = np.zeros((t_points, t_points), dtype=int)
        # summing occurences in different log space intervals
        # logspace = [0.1250, 0.2500, 0.5000, 1.0000, 2.0000]
        # 0    1.3 -> 1 0 -> 2 0 -> 3 0 -> 4 0 -> 5 1
        # 0.43  0     0 1    0 2    1 3    2 4    3 5
        for m in range(self.nbins_r):
            r_array_q += (r_array_n < r_bin_edges[m])

        fz = r_array_q > 0

        # getting angles in radians
        theta_array = cdist(points, points, lambda u, v: math.atan2((v[1] - u[1]), (v[0] - u[0])))
        norm_angle = theta_array[int(max_points[0]), max_points[1]]
        # making angles matrix rotation invariant
        theta_array = (theta_array - norm_angle * (np.ones((t_points, t_points)) - np.identity(t_points)))
        # removing all very small values because of float operation
        theta_array[np.abs(theta_array) < 1e-7] = 0

        # 2Pi shifted because we need angels in [0,2Pi]
        theta_array_2 = theta_array + 2 * math.pi * (theta_array < 0)
        # Simple Quantization
        theta_array_q = (1 + np.floor(theta_array_2 / (2 * math.pi / self.nbins_theta))).astype(int)

        # building point descriptor based on angle and distance
        nbins = self.nbins_theta * self.nbins_r
        descriptor = np.zeros((t_points, nbins))
        for i in range(t_points):
            sn = np.zeros((self.nbins_r, self.nbins_theta))
            for j in range(t_points):
                if (fz[i, j]):
                    sn[r_array_q[i, j] - 1, theta_array_q[i, j] - 1] += 1
            descriptor[i] = sn.reshape(nbins)

        return descriptor

    def cosine_diff(self, P, Q):
        """
            Fast cosine diff.
        """
        P = P.flatten()
        Q = Q.flatten()
        assert len(P) == len(Q), 'number of descriptors should be the same'
        return cosine(P, Q)

    def diff(self, P, Q, qlength=None):
        """
            More precise but not very speed efficient diff.
            if Q is generalized shape context then it compute shape match.
            if Q is r point representative shape contexts and qlength set to 
            the number of points in Q then it compute fast shape match.
        """
        result = None
        C = self.cost_by_paper(P, Q, qlength)

        result = self._hungarian(C)

        return result


#----------------------------------------------------------------------------------------#


class Matching(object):

    def __init__(self, stroke_pairs, stroke_energies, stroke_centroids, frames, frame_centroids, delete0, delete1):
        self.stroke_pairs = stroke_pairs
        self.stroke_energies = stroke_energies
        self.stroke_centroids = stroke_centroids
        self.frames = frames
        self.frame_centroids = frame_centroids
        self.delete0 = delete0
        self.delete1 = delete1

    def writeJSON(self, ind):
        matJSON = jsonpickle.encode(self)

        with open("match." + str(ind) + "." + str(int(self.frames[0])) + "." + str(int(self.frames[1])) + ".txt", 'w') as f:  
            f.write(matJSON)


#----------------------------------------------------------------------------------------#


COPY_SPACING = 5
COPIES = 1
RAND_FACTOR = 1

RESAMPLE_LENGTH = 0.01
DISTANCE_THRESHOLD = 2.5


def distance(arr1, arr2):
    output = 0
    for ind in range(len(arr1)):
        output += math.pow(arr1[ind] - arr2[ind], 2)
    return math.sqrt(output)


# Calculates Rotation Matrix given euler angles.
def euler_to_matrix(theta):
    R_x = np.array([[1,         0,                  0],
                    [0,         math.cos(theta[0]), -math.sin(theta[0])],
                    [0,         math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])],
                    [0,                     1,      0],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def average(arr1, arr2):
    output = []
    for ind in range(len(arr1)):
        output.append((arr1[ind] + arr2[ind])/2)
    return output


# ratio reflects distance from earlier frame
def linear_interpolate(arr1, arr2, ratio):
    return np.add(ratio * arr2, (1 - ratio) * arr1)
    # output = []
    # for ind in range(len(arr1)):
    #     output.append(ratio * arr2[ind] + (1 - ratio) * arr1[ind])
    # return output


def flip_if_necessary(context, stroke0, stroke1, frame0, frame1, frame_centroids):
    # set context for dissolve
    # context.area.type = 'VIEW_3D'
    # bpy.ops.object.mode_set(mode='EDIT_GPENCIL')

    # frame0_num = frame0.frame_number
    frame1_num = frame1.frame_number
    # if len(stroke1.points) > len(stroke0.points):
    #     diff = len(stroke1.points) - len(stroke0.points)
    #     context.scene.frame_set(frame1_num)
    #     for _ in range(1, diff + 1):
    #         for stroke in frame0.strokes:
    #             stroke.select = False
    #             for point in stroke.points:
    #                 point.select = False
    #         stroke1.select = True
    #         stroke1.points[-int(np.random.random_sample() * (len(stroke1.points)))].select = True
    #         bpy.ops.gpencil.dissolve(type='POINTS')
    # elif len(stroke1.points) < len(stroke0.points):
    #     diff = len(stroke0.points) - len(stroke1.points)
    #     context.scene.frame_set(frame0_num)
    #     for _ in range(1, diff + 1):
    #         for stroke in frame1.strokes:
    #             stroke.select = False
    #             for point in stroke.points:
    #                 point.select = False
    #         stroke0.select = True
    #         stroke0.points[-int(np.random.random_sample() * (len(stroke0.points)))].select = True
    #         bpy.ops.gpencil.dissolve(type='POINTS')
    #     context.scene.frame_set(frame1_num)
        # for _ in range(1, diff + 1):
        #     for stroke in frame1.strokes:
        #         stroke.select = False
        #         for point in stroke.points:
        #             point.select = False
        #     stroke1.points[-(random.randrange(len(stroke1.points)))
        #                     ].select = True
        #     bpy.ops.gpencil.dissolve(type='POINTS')

    normal_sum = 0
    flipped_sum = 0

    stroke0_cp = None
    stroke1_cp = None

    for point in stroke0.points:
        if stroke0_cp is None:
            stroke0_cp = [point.co]
        else:
            stroke0_cp = np.append(stroke0_cp, [point.co], axis=0)

    for point in stroke1.points:
        if stroke1_cp is None:
            stroke1_cp = [point.co]
        else:
            stroke1_cp = np.append(stroke1_cp, [point.co], axis=0)

    # normalize points (avg magnitude = 1)
    stroke0_norms = [np.linalg.norm(point) for point in stroke0_cp]
    stroke1_norms = [np.linalg.norm(point) for point in stroke1_cp]
    mean_norm0 = np.mean(stroke0_norms)
    mean_norm1 = np.mean(stroke1_norms)

    for ind in range(len(stroke0.points)):
        stroke0_cp[ind] = stroke0_cp[ind] / mean_norm0
        stroke1_cp[ind] = stroke1_cp[ind] / mean_norm1

    for ind in range(len(stroke0.points)):
        stroke0_relative = np.subtract(stroke0_cp[ind], frame_centroids[0])
        stroke1_relative = np.subtract(stroke1_cp[ind], frame_centroids[1])
        stroke1_flipped_relative = np.subtract(
            stroke1_cp[len(stroke0.points) - 1 - ind], frame_centroids[1])
        normal_sum += distance(stroke0_relative, stroke1_relative)
        flipped_sum += distance(stroke0_relative, stroke1_flipped_relative)

    if flipped_sum < normal_sum:
        context.scene.frame_set(frame1_num)
        stroke1.select = True
        bpy.ops.gpencil.stroke_flip()


def calc_stroke_centroids(context, layer_frames, omit0=[], omit1=[]):
    frames = []
    frame_centroids = None
    stroke_centroids = {}
    
    for frame_ind in range(len(layer_frames)):
        points = None
        frame = layer_frames[frame_ind]
        frames = np.append(frames, int(frame.frame_number))
        context.scene.frame_set(frame.frame_number)

        camera = context.scene.camera

        # stroke_centroids[frame_ind] = frame_stroke_centroids
        frame_stroke_centroids_dict = {}
        frame_stroke_centroids = None

        for stroke_ind in range(len(frame.strokes)):
            if frame_ind == 0 and stroke_ind in omit0:
                # print("None added 0", frame_ind, stroke_ind)
                frame_stroke_centroids_dict[stroke_ind] = np.array([None, None, None])
                if frame_stroke_centroids is None:
                    frame_stroke_centroids = [np.array([None, None, None])]
                else:
                    frame_stroke_centroids = np.append(
                        frame_stroke_centroids, [np.array([None, None, None])], axis=0)
                continue
            elif frame_ind == 1 and stroke_ind in omit1:
                # print("None added 1", frame_ind, stroke_ind)
                frame_stroke_centroids_dict[stroke_ind] = np.array([None, None, None])
                if frame_stroke_centroids is None:
                    frame_stroke_centroids = [np.array([None, None, None])]
                else:
                    frame_stroke_centroids = np.append(
                        frame_stroke_centroids, [np.array([-1, -1, -1])], axis=0)
                continue
            stroke = frame.strokes[stroke_ind]
            stroke.select = True

            # calculate stroke centroids
            box_min = stroke.bound_box_min
            box_max = stroke.bound_box_max
            centroid = np.expand_dims(
                np.append(average(box_min, box_max), 1), 1)

            # convert from world to camera space
            M = euler_to_matrix(camera.rotation_euler)
            t = camera.location
            t = np.array([[t[0]], [t[1]], [t[2]]])
            cam_to_world = np.concatenate((M, t), axis=1)
            cam_to_world = np.concatenate(
                (cam_to_world, [[0, 0, 0, 1]]), axis=0)

            world_to_cam = np.linalg.inv(cam_to_world)

            centroid_cam = world_to_cam @ centroid

            for point in stroke.points:
                point_cam = world_to_cam @ np.expand_dims(np.append(context.object.matrix_world @ point.co, 1), 1)
                point_cam = np.transpose(point_cam[:3])[0]
                if points is None:
                    points = [point_cam]
                else:
                    points = np.append(points, [point_cam], axis = 0)

            # convert centroid to coord array
            centroid_cam = np.transpose(centroid_cam[:3])[0]

            frame_stroke_centroids_dict[stroke_ind] = centroid_cam

            if frame_stroke_centroids is None:
                frame_stroke_centroids = [centroid_cam]
            else:
                frame_stroke_centroids = np.append(
                    frame_stroke_centroids, [centroid_cam], axis=0)

        # calculate centroid of centroids
        if points is None:
            frame_centroid = None
        elif len(points) > 0:
            frame_centroid = np.mean(points, axis=0)
        else:
            frame_centroid = points

        if frame_centroids is None:
            frame_centroids = [frame_centroid]
        else:
            frame_centroids = np.append(
                frame_centroids, [frame_centroid], axis=0)

        for ind in range(len(frame_stroke_centroids)):
            if frame_stroke_centroids[ind][0] is not None:
                frame_stroke_centroids[ind] = np.subtract(
                    frame_stroke_centroids[ind], frame_centroid)

        for ind in frame_stroke_centroids_dict:
            if frame_stroke_centroids_dict[ind][0] is not None:
                frame_stroke_centroids_dict[ind] = np.subtract(
                    frame_stroke_centroids_dict[ind], frame_centroid)

        stroke_centroids[len(frames) - 1] = frame_stroke_centroids_dict

        bpy.ops.gpencil.stroke_sample(length=RESAMPLE_LENGTH)

    return stroke_centroids, frames, frame_centroids


def interpolate(context, layer, stroke_ind0, stroke_ind1, frame0_num, frame0_ind, frame1_num, frame1_ind, frame_centroids):
    stroke0 = layer.frames[frame0_ind].strokes[int(stroke_ind0)]
    stroke1 = layer.frames[frame1_ind].strokes[int(stroke_ind1)]

    if len(stroke0.points) != len(stroke1.points):
        # print("*1", len(stroke0.points), len(stroke1.points))
        # # resample
        # if len(stroke0.points) > len(stroke1.points):
        #     context.scene.frame_set(frame0_num)
        #     stroke0.select = True
        #     stroke1.select = False
        #     dist = RESAMPLE_LENGTH / len(stroke1.points) * len(stroke0.points)
        #     # print(dist)
        #     bpy.ops.gpencil.stroke_sample(length=dist)
        # else:
        #     context.scene.frame_set(frame1_num)
        #     stroke0.select = False
        #     stroke1.select = True
        #     dist = RESAMPLE_LENGTH / len(stroke0.points) * len(stroke1.points)
        #     # print(dist)
        #     bpy.ops.gpencil.stroke_sample(length=dist)
        
        # # print("2", len(stroke0.points), len(stroke1.points))

        # resolve off-by-one errors
        if len(stroke0.points) > 2*len(stroke1.points):
            context.scene.frame_set(frame0_num)
            for stroke in layer.frames[frame0_ind].strokes:
                stroke.select = False
                for point in stroke.points:
                    point.select = False
            to_select = range(0, len(stroke0.points), int(len(stroke0.points) / len(stroke1.points)))
            for ind in to_select:
                stroke0.points[ind].select = True
            bpy.ops.gpencil.dissolve(type='POINTS')
        elif len(stroke1.points) > 2*len(stroke0.points):
            context.scene.frame_set(frame1_num)
            for stroke in layer.frames[frame1_ind].strokes:
                stroke.select = False
                for point in stroke.points:
                    point.select = False
            to_select = range(0, len(stroke1.points), int(len(stroke1.points) / len(stroke0.points)))
            for ind in to_select:
                stroke1.points[ind].select = True
            bpy.ops.gpencil.dissolve(type='POINTS')
        
        if len(stroke0.points) > len(stroke1.points):
            context.scene.frame_set(frame0_num)
            diff = len(stroke0.points) - len(stroke1.points)
            for _ in range(1, diff + 1):
                for stroke in layer.frames[frame0_ind].strokes:
                    stroke.select = False
                    for point in stroke.points:
                        point.select = False
                stroke0.points[-(random.randrange(len(stroke0.points)))
                               ].select = True
                bpy.ops.gpencil.dissolve(type='POINTS')
        elif len(stroke1.points) > len(stroke0.points):
            context.scene.frame_set(frame1_num)
            diff = len(stroke1.points) - len(stroke0.points)
            for _ in range(1, diff + 1):
                for stroke in layer.frames[frame1_ind].strokes:
                    stroke.select = False
                    for point in stroke.points:
                        point.select = False
                stroke1.points[-(random.randrange(len(stroke1.points)))
                               ].select = True
                bpy.ops.gpencil.dissolve(type='POINTS')

    # print("before linear interpolate")

    # linearly interpolate between strokes
    for frame_ind in range(int(frame0_num) + 1, int(frame1_num)):
        ratio = (frame_ind - frame0_num)/(frame1_num - frame0_num)
        frame = layer.frames[int(frame_ind - frame0_num + frame0_ind)]

        # make sure same number of points
        curr_stroke = frame.strokes[int(stroke_ind0)]
        if len(curr_stroke.points) > len(stroke0.points):
            diff = len(curr_stroke.points) - len(stroke0.points)
            context.scene.frame_set(frame_ind)
            for _ in range(1, diff + 1):
                for stroke in frame.strokes:
                    stroke.select = False
                    for point in stroke.points:
                        point.select = False
                curr_stroke.points[-int(np.random.random_sample() * (len(curr_stroke.points)))].select = True
                bpy.ops.gpencil.dissolve(type='POINTS')
        elif len(curr_stroke.points) < len(stroke0.points):
            diff = len(stroke0.points) - len(curr_stroke.points)
            context.scene.frame_set(frame0_num)
            for _ in range(1, diff + 1):
                for stroke in layer.frames[frame0_ind].strokes:
                    stroke.select = False
                    for point in stroke.points:
                        point.select = False
                stroke0.points[-int(np.random.random_sample() * (len(stroke0.points)))].select = True
                bpy.ops.gpencil.dissolve(type='POINTS')
            context.scene.frame_set(frame1_num)
            for _ in range(1, diff + 1):
                for stroke in layer.frames[frame1_ind].strokes:
                    stroke.select = False
                    for point in stroke.points:
                        point.select = False
                stroke1.points[-(random.randrange(len(stroke1.points)))
                               ].select = True
                bpy.ops.gpencil.dissolve(type='POINTS')
        # print("before flip")
        if stroke_ind0 == 17 or stroke_ind0 == 18:
            flip_if_necessary(context, stroke0, stroke1, frame, layer.frames[frame1_ind], frame_centroids)
        # print("after flip")
        for ind_p in range(len(stroke0.points)):
            curr_stroke.points[ind_p].co = linear_interpolate(
                stroke0.points[ind_p].co, stroke1.points[ind_p].co, ratio)

    # print("after linear interpolate")


def propose_matching(context, layer, ind):

    SHAPE_CONTEXT_WEIGHT = 0
    POSITION_WEIGHT = 0
    DOT_PRODUCT_WEIGHT = 0

    if context.scene.shape_context:
        SHAPE_CONTEXT_WEIGHT = 1
    if context.scene.position:
        POSITION_WEIGHT = 1
    if context.scene.dot_product:
        DOT_PRODUCT_WEIGHT = 0.1

    frame_ind0 = ind
    frame_ind1 = ind + 1

    layer_frames = np.array([layer.frames[frame_ind0], layer.frames[frame_ind1]])

    bpy.ops.gpencil.stroke_sample(length=RESAMPLE_LENGTH)

    # stroke_centroids[frame_ind][stroke_ind] = centroid_coords
    stroke_centroids, frames, frame_centroids = calc_stroke_centroids(context, layer_frames)

    # identify strokes that are too far from everything else
    all_stroke_distances = {}
    arr_stroke_dists = []
    for frame_ind in range(len(stroke_centroids)):
        for stroke_ind in range(len(stroke_centroids[frame_ind])):
            all_stroke_distances[frame_ind * len(stroke_centroids[0]) + stroke_ind] = np.linalg.norm(
                stroke_centroids[frame_ind][stroke_ind])
            arr_stroke_dists = np.append(arr_stroke_dists, np.linalg.norm(
                stroke_centroids[frame_ind][stroke_ind]))

    std_dev = np.std(arr_stroke_dists)
    mean = np.mean(arr_stroke_dists)

    ignore = bpy.context.scene.collection.get('ignore', None)

    delete0 = array.array('i')
    delete1 = array.array('i')
    if ignore is not None:
        if str(layer_frames[0].frame_number) in ignore:
            user_specified0 = ignore[str(layer_frames[0].frame_number)][1:].split(" ")
            for ind in user_specified0:
                delete0 = np.append(delete0, int(ind))
        if str(layer_frames[1].frame_number) in ignore:
            user_specified1 = ignore[str(layer_frames[1].frame_number)][1:].split(" ")
            for ind in user_specified1:
                delete1 = np.append(delete1, int(ind))

    context.scene.frame_set(layer.frames[frame_ind0].frame_number)
    layer.frames[frame_ind0].select = True
    for stroke_ind in range(len(layer.frames[frame_ind0].strokes)):
        stroke = layer.frames[frame_ind0].strokes[stroke_ind]
        if len(stroke.points) < 2:
            delete0 = np.append(delete0, int(stroke_ind))

    context.scene.frame_set(layer.frames[frame_ind1].frame_number)
    layer.frames[frame_ind0].select = True
    for stroke_ind in range(len(layer.frames[frame_ind1].strokes)):
        stroke = layer.frames[frame_ind1].strokes[stroke_ind]
        if len(stroke.points) < 2:
            delete1 = np.append(delete1, int(stroke_ind))

    # print("-", delete0, delete1)

    for key in all_stroke_distances:
        value = all_stroke_distances[key]
        if value > mean + DISTANCE_THRESHOLD * std_dev or value < mean - DISTANCE_THRESHOLD * std_dev:
            if key < len(stroke_centroids[0]):
                # in frame 0
                if key not in delete0:
                    delete0 = np.append(delete0, key)
                    if ignore is None:
                        bpy.context.scene.collection["ignore"] = {}
                        bpy.context.scene.collection["ignore"][str(frames[0])] = " " + str(key)
                    elif str(frames[0]) not in ignore:
                        bpy.context.scene.collection["ignore"][str(frames[0])] = " " + str(key)
                    elif " " + str(key) not in bpy.context.scene.collection["ignore"][str(frames[0])]:
                        bpy.context.scene.collection["ignore"][str(frames[0])] += " " + str(key)
            else:
                # in frame 1
                if key not in delete1:
                    delete1 = np.append(delete1, key - len(stroke_centroids[0]))
                    if ignore is None:
                        bpy.context.scene.collection["ignore"] = {}
                        bpy.context.scene.collection["ignore"][str(frames[1])] = " " + str(key - len(stroke_centroids[0]))
                    elif str(frames[1]) not in ignore:
                        bpy.context.scene.collection["ignore"][str(frames[1])] = " " + str(key - len(stroke_centroids[0]))
                    elif " " + str(key - len(stroke_centroids[0])) not in bpy.context.scene.collection["ignore"][str(frames[1])]:
                        bpy.context.scene.collection["ignore"][str(frames[1])] += " " + str(key - len(stroke_centroids[0]))

    delete0 = []
    delete1 = []

    stroke_centroids, frames, frame_centroids = calc_stroke_centroids(context, 
        layer_frames, delete0, delete1)

    # initialize empty stroke pair dictionary (stroke 1 ind, stroke 2 ind, energy)
    stroke_pairs = {}
    stroke_energies = None

    # create ranking of pairwise stroke distances
    shape_context = ShapeContext()
    stroke0_sc = None
    stroke1_sc = None

    for stroke in layer_frames[0].strokes:
        if stroke0_sc is None:
            sc = shape_context.compute(np.array(stroke.points))
            stroke0_sc = np.array(sc / np.linalg.norm(sc))
        else:
            sc = shape_context.compute(np.array(stroke.points))
            stroke0_sc = np.append(stroke0_sc, sc / np.linalg.norm(sc), axis = 0)
    
    for stroke in layer_frames[1].strokes:
        if stroke1_sc is None:
            sc = shape_context.compute(np.array(stroke.points))
            stroke1_sc = np.array(sc / np.linalg.norm(sc))
        else:
            sc = shape_context.compute(np.array(stroke.points))
            stroke1_sc = np.append(stroke1_sc, sc / np.linalg.norm(sc), axis = 0)

    stroke_distances = None
    for stroke_ind0 in stroke_centroids[0]:
        if stroke_centroids[0][stroke_ind0][0] is None:
            continue
        null_sc = []
        avg_sc = 0
        for stroke_ind1 in stroke_centroids[1]:
            if stroke_centroids[1][stroke_ind1][0] is None:
                continue
            stroke0_rel = np.subtract(stroke_centroids[0][stroke_ind0], frame_centroids[0])
            stroke1_rel = np.subtract(stroke_centroids[1][stroke_ind1], frame_centroids[1])

            energy = POSITION_WEIGHT * distance(stroke_centroids[0][stroke_ind0], stroke_centroids[1][stroke_ind1]) - DOT_PRODUCT_WEIGHT * np.dot(stroke0_rel, stroke1_rel)
            this_sc = SHAPE_CONTEXT_WEIGHT * distance(stroke0_sc[stroke_ind0], stroke1_sc[stroke_ind1])
            
            if not math.isnan(this_sc):
                energy = energy + this_sc
                avg_sc = avg_sc + this_sc
            else:
                null_sc = np.append(null_sc, stroke_ind1)

            if stroke_distances is None:
                stroke_distances = np.array((stroke_ind0, stroke_ind1, energy), dtype=[
                                            ('first', 'i'), ('second', 'i'), ('energy', 'f')])
            else:
                stroke_distances = np.append(stroke_distances, np.array((stroke_ind0, stroke_ind1, energy), dtype=[('first', 'i'), ('second', 'i'), ('energy', 'f')]))
            # for row in stroke_distances:
            #     print(row)
        
        avg_sc = avg_sc / len(stroke_centroids[1])

        prev_row_ind = 0
        for ind1 in null_sc:
            for row_ind in range(prev_row_ind, len(stroke_distances)):
                row = stroke_distances[row_ind]
                if row['first'] == stroke_ind0 and row['second'] == ind1:
                    prev_row_ind = row_ind
                    break
            # condition = stroke_distances['first'] == stroke_ind0 and stroke_distances['second'] == ind1
            # extracted = np.where(condition, 1, 0)
            # element_ind = np.nonzero(extracted)[0]
            stroke_distances[row_ind]['energy'] += avg_sc

    # adjust to ensure all >= 0
    min_val = np.min(stroke_distances['energy'])
    if min_val < 0:
        if stroke_distances.size > 1:
            for row in stroke_distances:
                row['energy'] -= min_val
        else:
            stroke_distances['energy'] -= min_val

    if stroke_distances.size > 1:
        stroke_distances = np.sort(stroke_distances, order="energy")
    stroke_distances_cp = np.copy(stroke_distances)

    # save stroke_distances as complete stroke energies to scene
    bpy.context.scene.collection["matching_energies"] = ""
    if stroke_distances.size > 1:
        for row in stroke_distances:
            bpy.context.scene.collection["matching_energies"] = bpy.context.scene.collection["matching_energies"] + str(row['first']) + " " + str(row['second']) + " " + str(row['energy']) + ","
    else:
        bpy.context.scene.collection["matching_energies"] = str(stroke_distances['first']) + " " + str(stroke_distances['second']) + " " + str(stroke_distances['energy']) + ","
    bpy.context.scene.collection["matching_energies"] = bpy.context.scene.collection["matching_energies"][:-1]

    unmatched0 = []
    for item in stroke_centroids[0].keys():
        if stroke_centroids[0][item][0] is not None:
            unmatched0 = np.append(unmatched0,  item)
    unmatched1 = []
    for item in stroke_centroids[1].keys():
        if stroke_centroids[1][item][0] is not None:
            unmatched1 = np.append(unmatched1,  item)

    if bpy.context.scene.collection["pair"] is not None:
        pairs = bpy.context.scene.collection["pair"][:-1].split(",")
        for pair in pairs:
            inds = pair.split(" ")
            unmatched0 = unmatched0[unmatched0 != int(inds[0])]
            unmatched1 = unmatched1[unmatched1 != int(inds[1])]
            stroke_pairs[int(inds[0])] = int(inds[1])
            if stroke_energies is None:
                stroke_energies = np.array((int(inds[0]), int(inds[1]), float('-inf')), dtype=[
                                            ('first', 'i'), ('second', 'i'), ('energy', 'f')])
            else:
                stroke_energies = np.append(stroke_energies, np.array((int(inds[0]), int(inds[1]), float('-inf') ), dtype=[('first', 'i'), ('second', 'i'), ('energy', 'f')]))

            stroke_distances = stroke_distances[stroke_distances['first']
                                                != int(inds[0])]
            stroke_distances = stroke_distances[stroke_distances['second']
                                                != int(inds[1])]

    p75 = np.percentile(stroke_distances_cp['energy'], 75)
    iqr = p75 - np.percentile(stroke_distances_cp['energy'], 25)
    cutoff = p75 + iqr * float(bpy.context.scene.QueryProps.query)

    while stroke_distances.size > 0 and len(unmatched0) > 0 and len(unmatched1) > 0:
        if stroke_distances.size > 1:
            new_match = stroke_distances[0]
        else:
            new_match = stroke_distances

        # if clamp_90(stroke_distances_cp['energy'], new_match['energy']) == new_match['energy'] or True:
        if new_match['energy'] < cutoff or True:
            stroke_pairs[int(new_match['first'])] = int(new_match['second'])
            
            if stroke_energies is None:
                stroke_energies = np.array((int(new_match['first']), int(new_match['second']), float(new_match['energy'])), dtype=[
                                            ('first', 'i'), ('second', 'i'), ('energy', 'f')])
            else:
                stroke_energies = np.append(stroke_energies, np.array((int(new_match['first']), int(new_match['second']), float(new_match['energy'])), dtype=[('first', 'i'), ('second', 'i'), ('energy', 'f')]))

            unmatched0 = unmatched0[unmatched0 != new_match['first']]
            unmatched1 = unmatched1[unmatched1 != new_match['second']]

        stroke_distances = stroke_distances[stroke_distances['first']
                                                    != new_match['first']]
        stroke_distances = stroke_distances[stroke_distances['second']
                                                    != new_match['second']]

    # match remaining strokes
    if len(unmatched0) > 0:
        while len(unmatched0) > 0 and len(stroke_distances_cp) > 0:
            to_match = unmatched0[0]
            unmatched0 = unmatched0[unmatched0 != to_match]
            new_match = stroke_distances_cp[stroke_distances_cp['first']
                                            == to_match][0]

            # duplicate double-matched stroke
            match_ind = new_match['second']
            context.scene.frame_set(frames[1])
            for stroke in layer.frames[frame_ind1].strokes:
                stroke.select = False
            layer.frames[frame_ind1].strokes[match_ind].select = True
            bpy.ops.gpencil.duplicate()
            stroke_pairs[int(to_match)] = int(len(layer.frames[frame_ind1].strokes) - 1)
            stroke_centroids[1][int(len(layer.frames[frame_ind1].strokes) - 1)] = stroke_centroids[1][match_ind]
            if stroke_energies is None:
                stroke_energies = np.array((int(to_match), int(len(layer.frames[frame_ind1].strokes) - 1), float(new_match['energy'])), dtype=[
                                            ('first', 'i'), ('second', 'i'), ('energy', 'f')])
            else:
                stroke_energies = np.append(stroke_energies, np.array((int(to_match), int(len(layer.frames[frame_ind1].strokes) - 1), float(new_match['energy'])), dtype=[('first', 'i'), ('second', 'i'), ('energy', 'f')]))


    if len(unmatched1) > 0:
        while len(unmatched1) > 0 and len(stroke_distances_cp) > 0:
            to_match = unmatched1[0]
            unmatched1 = unmatched1[unmatched1 != to_match]
            new_match = stroke_distances_cp[stroke_distances_cp['second']
                                            == to_match][0]

            # duplicate double-matched stroke
            match_ind = new_match['first']
            context.scene.frame_set(frames[0])
            for stroke in layer.frames[frame_ind0].strokes:
                stroke.select = False
            layer.frames[frame_ind0].strokes[match_ind].select = True
            bpy.ops.gpencil.duplicate()
            stroke_pairs[int(len(layer.frames[frame_ind0].strokes) - 1)] = int(to_match)
            stroke_centroids[0][int(len(layer.frames[frame_ind1].strokes) - 1)] = stroke_centroids[0][match_ind]
            if stroke_energies is None:
                stroke_energies = np.array((int(len(layer.frames[frame_ind0].strokes) - 1), int(to_match), float(new_match['energy'])), dtype=[
                                            ('first', 'i'), ('second', 'i'), ('energy', 'f')])
            else:
                stroke_energies = np.append(stroke_energies, np.array((int(len(layer.frames[frame_ind0].strokes) - 1), int(to_match), float(new_match['energy'])), dtype=[('first', 'i'), ('second', 'i'), ('energy', 'f')]))

    # save most recent matching
    bpy.context.scene.collection["matching"] = {}
    for key in stroke_pairs:
        bpy.context.scene.collection["matching"][str(key)] = str(stroke_pairs[key])

    if stroke_energies.size > 1:
        stroke_energies = np.sort(stroke_energies, order="energy")
        bpy.context.scene.collection["range"] = str(stroke_energies[0]['energy']) + ' ' + str(stroke_energies[-1]['energy'])
    else:
        bpy.context.scene.collection["range"] = str(stroke_energies['energy']) + ' ' + str(stroke_energies['energy'])

    return Matching(stroke_pairs, stroke_energies, stroke_centroids, frames, frame_centroids, delete0, delete1)


def interpolate_frames(context, layer, ind, matching):

    stroke_pairs = matching.stroke_pairs
    frames = matching.frames
    frame_centroids = matching.frame_centroids
    delete0 = matching.delete0
    delete1 = matching.delete1

    context.area.type = 'VIEW_3D'
    bpy.ops.object.mode_set(mode='EDIT_GPENCIL')

    frame_ind0 = ind
    frame_ind1 = ind + 1

    # resample paired strokes if needed
    for stroke_ind0 in stroke_pairs:
        print(stroke_ind0)
        # print("in for loop")
        stroke_ind1 = stroke_pairs[stroke_ind0]
        stroke0 = layer.frames[frame_ind0].strokes[stroke_ind0]
        stroke1 = layer.frames[frame_ind1].strokes[stroke_ind1]

        temp_point_spacing = RESAMPLE_LENGTH
        if len(stroke0.points) > len(stroke1.points):
            # print("if")
            context.scene.frame_set(frames[1])
            layer.frames[frame_ind1].select = True
            stroke0.select = False
            stroke1.select = True
            while len(stroke0.points) > len(stroke1.points):
                temp_point_spacing = temp_point_spacing / 2
                # print("before subdivide", len(stroke0.points), len(stroke1.points))
                bpy.ops.gpencil.stroke_subdivide(only_selected=False)
                # print("after subdivide", len(stroke0.points), len(stroke1.points))
            temp_point_spacing = temp_point_spacing / 2
            bpy.ops.gpencil.stroke_subdivide(only_selected=False)
            # print("inner if complete")
        elif len(stroke0.points) < len(stroke1.points):
            # print("else")
            context.scene.frame_set(frames[0])
            layer.frames[frame_ind0].select = True
            stroke0.select = True
            stroke1.select = False
            while len(stroke0.points) < len(stroke1.points):
                temp_point_spacing = temp_point_spacing / 2
                print("before subdivide", len(stroke0.points), len(stroke1.points))
                bpy.ops.gpencil.stroke_subdivide(only_selected=False)
                print("after subdivide", len(stroke0.points), len(stroke1.points))
            # print("inner while complete")
            temp_point_spacing = temp_point_spacing / 2
            bpy.ops.gpencil.stroke_subdivide(only_selected=False)
            # print("inner else complete")

        # print("first if complete")

        if len(stroke0.points) != len(stroke1.points):
            # resample
            if len(stroke0.points) > len(stroke1.points):
                context.scene.frame_set(layer.frames[frame_ind0].frame_number)
                stroke0.select = True
                stroke1.select = False
                dist = temp_point_spacing / len(stroke1.points) * len(stroke0.points)
                bpy.ops.gpencil.stroke_sample(length=dist)
            else:
                context.scene.frame_set(layer.frames[frame_ind1].frame_number)
                stroke0.select = False
                stroke1.select = True
                dist = temp_point_spacing / len(stroke0.points) * len(stroke1.points)
                bpy.ops.gpencil.stroke_sample(length=dist)

            # resolve off-by-one errors
            if len(stroke0.points) > 2*len(stroke1.points):
                context.scene.frame_set(layer.frames[frame_ind0].frame_number)
                for stroke in layer.frames[frame_ind0].strokes:
                    stroke.select = False
                    for point in stroke.points:
                        point.select = False
                to_select = range(0, len(stroke0.points), int(len(stroke0.points) / len(stroke1.points)))
                for ind in to_select:
                    stroke0.points[ind].select = True
                bpy.ops.gpencil.dissolve(type='POINTS')
            elif len(stroke1.points) > 2*len(stroke0.points):
                context.scene.frame_set(layer.frames[frame_ind1].frame_number)
                for stroke in layer.frames[frame_ind1].strokes:
                    stroke.select = False
                    for point in stroke.points:
                        point.select = False
                to_select = range(0, len(stroke1.points), int(len(stroke1.points) / len(stroke0.points)))
                for ind in to_select:
                    stroke1.points[ind].select = True
                bpy.ops.gpencil.dissolve(type='POINTS')

            if len(stroke0.points) > len(stroke1.points):
                context.scene.frame_set(layer.frames[frame_ind0].frame_number)
                diff = len(stroke0.points) - len(stroke1.points)
                for _ in range(1, diff + 1):
                    for stroke in layer.frames[frame_ind0].strokes:
                        stroke.select = False
                        for point in stroke.points:
                            point.select = False
                    stroke0.points[-(random.randrange(len(stroke0.points)))
                                ].select = True
                    bpy.ops.gpencil.dissolve(type='POINTS')
            elif len(stroke1.points) > len(stroke0.points):
                context.scene.frame_set(layer.frames[frame_ind1].frame_number)
                diff = len(stroke1.points) - len(stroke0.points)
                for _ in range(1, diff + 1):
                    for stroke in layer.frames[frame_ind1].strokes:
                        stroke.select = False
                        for point in stroke.points:
                            point.select = False
                    stroke1.points[-(random.randrange(len(stroke1.points)))
                                ].select = True
                    bpy.ops.gpencil.dissolve(type='POINTS')

        # print("second if complete")

    # print("exit for loop")

    # initialize intermediate frames
    context.scene.frame_set(frames[0])
    layer.frames[frame_ind0].select = True
    for frame_ind in range(int(frames[0]) + 1, int(frames[1])):
        context.scene.frame_set(frame_ind)
        bpy.ops.gpencil.frame_duplicate()
    
    frame_ind1 = int(frame_ind0 + frames[1] - frames[0])

    # print("initialized intermediate")

    # interpolate paired strokes
    for stroke_ind0 in stroke_pairs:
        print(stroke_ind0,"*")
        stroke_ind1 = stroke_pairs[stroke_ind0]
        interpolate(context, layer, stroke_ind0, stroke_ind1, frames[0], frame_ind0, frames[1], frame_ind1, frame_centroids)

    # print("interpolated paired strokes")

    # interpolate outlier strokes
    if context.scene.animate_outliers_bool:
        for outlier in delete0:
            coords = layer.frames[frame_ind1 - 1].strokes[outlier].points[0].co
            context.scene.frame_set(frames[1] - 1)
            for point in layer.frames[frame_ind1 - 1].strokes[outlier].points:
                point.co = coords
            interpolate(context, layer, outlier, outlier,
                        frames[0], frame_ind0, frames[1] - 1, frame_ind1 - 1, frame_centroids)
        for outlier in delete1:
            coords = layer.frames[frame_ind1].strokes[outlier].points[0].co
            context.scene.frame_set(frames[0] + 1)
            for point in layer.frames[frame_ind1].strokes[outlier].points:
                point.co = coords
            interpolate(context, layer, outlier, outlier,
                        frames[1], frame_ind0 + 1, frames[1], frame_ind1, frame_centroids)

    # print("interpolated outlier strokes")

    return frame_ind1


def object_match(context):

    # SETUP
    obj = context.object
    og_name = obj.name
    context.area.type = 'VIEW_3D'
    layer = obj.data.layers[0]

    bpy.ops.object.mode_set(mode='EDIT_GPENCIL')

    if bpy.data.objects.get('vis.' + og_name + '.' + str(int(layer.frames[1].frame_number))) is None:

        # PAD EMPTY FRAMES
        context.scene.frame_set(layer.frames[0].frame_number - 1)
        layer.frames[0].select = True
        bpy.ops.gpencil.frame_duplicate()
        frame = layer.frames[0]
        to_delete = False
        for stroke in frame.strokes:
            stroke.select = True
            to_delete = True
        if to_delete:
            bpy.ops.gpencil.delete(type='STROKES')

        layer.frames[0].select = False
        layer.frames[-1].select = True

        context.scene.frame_set(layer.frames[-1].frame_number + 1)
        bpy.ops.gpencil.frame_duplicate()

        frame = layer.frames[-1]
        to_delete = False
        for stroke in frame.strokes:
            stroke.select = True
            to_delete = True
        if to_delete:
            bpy.ops.gpencil.delete(type='STROKES')

        bpy.ops.object.mode_set(mode='OBJECT')
                        
    obj = bpy.data.objects[og_name]
    context.view_layer.objects.active = obj
    layer = obj.data.layers[0]
    matchings = []
    next_ind = 0
    
    for _ in range(len(layer.frames) - 1):
        if len(layer.frames[next_ind].strokes) > 0 and len(layer.frames[next_ind + 1].strokes) > 0:
            matching = propose_matching(context, layer, next_ind)
            matchings = np.append(matchings, matching)
        next_ind = next_ind + 1
        if (next_ind >= len(layer.frames)):
            break

    visualize_matching(og_name, context, layer, matchings)

    context.area.type = 'PROPERTIES'


def visualize_matching(og_name, context, layer, matchings):
    context.area.type = 'VIEW_3D'
    OFFSET = 5

    STROKE_PAIRS = context.scene.pairings
    FRAME_CENTROIDS = context.scene.frame_centroids
    STROKE_CENTROIDS = context.scene.stroke_centroids

    for ind in range(len(matchings)):
        # export matching info into json
        matching = matchings[ind]

        # find object using og_name & ind
        name = og_name
        # if ind > 0:
        #     name = name + '.' + str(ind).zfill(3)
        obj = bpy.data.objects[name]
        context.view_layer.objects.active = obj
        obj.select_set(True)
        layer = obj.data.layers[0]
        layer.select = True

        world_frame_centroids = None
        offsets = None

        delete_context = {'area': context.area}

        # duplicate GP for frames
        for f_ind in range(len(matching.frames)):
            frame = int(matching.frames[f_ind])

            # check if visualization already exists
            if bpy.data.objects.get('vis.' + og_name + '.' + str(int(frame))) is not None:
                layer.select = False
                obj.select_set(False)
                delete = bpy.data.objects.get('vis.' + og_name + '.' + str(int(frame)))
                delete.select_set(True)
                context.view_layer.objects.active = delete
                context.area.type = 'VIEW_3D'
                bpy.ops.object.mode_set(mode='OBJECT')
                context.view_layer.objects.active = delete

                bpy.ops.object.delete()
                context.view_layer.objects.active = obj
                obj.select_set(True)

            # collect world coordinates of frame_centroid
            # convert from camera to world space

            if world_frame_centroids is None:
                world_frame_centroids = np.array([convert_cam_to_world(context, frame, matching.frame_centroids[f_ind])])
            else:
                world_frame_centroids = np.append(world_frame_centroids, [convert_cam_to_world(context, frame, matching.frame_centroids[f_ind])], axis=0)

            bpy.ops.object.mode_set(mode='OBJECT')

            bpy.ops.object.duplicate(linked=False)
            new_obj = bpy.data.objects[name + '.001']
            
            vis_name = 'vis.' + str(name) + "." + str(frame)
            new_obj.name = vis_name

            # bpy.ops.gpencil.layer_duplicate_object(object=vis_name, mode='ALL')
            
            # translate in space (multiply by ind for x,y offset)
            obj.select_set(False)  
            new_obj.select_set(True)
            context.view_layer.objects.active = new_obj
            layer = new_obj.data.layers[0]
            layer.select = True
            if offsets is None:
                offsets = np.array([Vector([0, OFFSET, frame])])
            else:
                offsets = np.append(offsets, [Vector([0, OFFSET, frame])], axis=0)
            new_obj.location = new_obj.location + Vector(offsets[-1])

            bpy.ops.object.mode_set(mode='EDIT_GPENCIL')

            # delete other frames, translate frame to 0
            for stroke_frame in layer.frames:
                if stroke_frame.frame_number != frame and stroke_frame.frame_number >= matching.frames[0] and stroke_frame.frame_number <= matching.frames[-1]:
                    stroke_frame.select = True
                    context.scene.frame_set(stroke_frame.frame_number)
                    bpy.ops.gpencil.delete(type='FRAME')

            context.area.ui_type = 'DOPESHEET'
            context.space_data.ui_mode = 'GPENCIL'

            bpy.ops.action.select_all(action='DESELECT')
            new_obj.select_set(True)
            context.view_layer.objects.active = new_obj
            layer = new_obj.data.layers[0]
            layer.select = True

            for stroke_frame in layer.frames:
                if stroke_frame.frame_number == frame:
                    stroke_frame.select = True
                    bpy.ops.transform.transform(mode='TIME_TRANSLATE', value=(-int(stroke_frame.frame_number), 0, 0, 0), orient_axis='Z', orient_type='VIEW', orient_matrix=((-1, -0, -0), (-0, -1, -0), (-0, -0, -1)), orient_matrix_type='VIEW', mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)

            context.area.type = 'VIEW_3D'

            layer.select = False
            context.view_layer.objects.active = obj
            obj.select_set(True)  
            new_obj.select_set(False)

        # construct empty GP object for pairing visualization. Save one reference two-point stroke (duplicate to turn into frame centroid marker).
        vis_name = 'vis.pair.' + str(name) + "." + str(matching.frames[0]) + "." + str(matching.frames[1])

        bpy.ops.object.mode_set(mode='OBJECT') 

        if bpy.data.objects.get(vis_name) is not None:
            delete = bpy.data.objects.get(vis_name)
            delete.select_set(True)
            obj.select_set(False)
            context.view_layer.objects.active = delete
            bpy.ops.object.delete(delete_context)
            context.view_layer.objects.active = obj
            obj.select_set(True) 

        bpy.ops.object.mode_set(mode='OBJECT') 
        
        bpy.ops.object.duplicate(linked=False)
        new_obj = bpy.data.objects[name + '.001']

        bpy.ops.object.mode_set(mode='EDIT_GPENCIL') 
                
        new_obj.name = vis_name

        context.view_layer.objects.active = new_obj
        obj.select_set(False)
        new_obj.select_set(True)

        gp_mat_white = bpy.data.materials.new("white")
        bpy.data.materials.create_gpencil_data(gp_mat_white)
        gp_mat_white.grease_pencil.color = (1, 1, 1, 1)
        new_obj.data.materials.append(gp_mat_white)
        white_name = search_for_color(new_obj, "white")

        gp_mat_gray = bpy.data.materials.new("gray")
        bpy.data.materials.create_gpencil_data(gp_mat_gray)
        gp_mat_gray.grease_pencil.color = (0.5, 0.5, 0.5, 0.5)
        new_obj.data.materials.append(gp_mat_gray)

        # delete all but first non-empty frame
        bpy.ops.object.mode_set(mode='EDIT_GPENCIL') 
        layer = new_obj.data.layers[0]
        for l_frame in layer.frames:
            context.scene.frame_set(l_frame.frame_number)
            if l_frame.frame_number != matching.frames[0] and l_frame.frame_number >= matching.frames[0] and l_frame.frame_number <= matching.frames[-1]:
                l_frame.select = True
                bpy.ops.gpencil.delete(type='FRAME')

            elif l_frame.frame_number == matching.frames[0]:
                context.area.ui_type = 'DOPESHEET'
                context.space_data.ui_mode = 'GPENCIL'
                bpy.ops.action.select_all(action='DESELECT')
                layer = new_obj.data.layers[0]
                layer.select = True
                l_frame.select = True
                bpy.ops.transform.transform(mode='TIME_TRANSLATE', value=(-int(l_frame.frame_number), 0, 0, 0), orient_axis='Z', orient_type='VIEW', orient_matrix=((-1, -0, -0), (-0, -1, -0), (-0, -0, -1)), orient_matrix_type='VIEW', mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)

                context.area.type = 'VIEW_3D'
                bpy.ops.object.mode_set(mode='EDIT_GPENCIL') 

                frame = l_frame
                # delete all but first stroke
                ref_ind = 0
                while len(l_frame.strokes[ref_ind].points) < 2:
                    ref_ind += 1

                for ind in range(len(l_frame.strokes)):
                    stroke = l_frame.strokes[ind]
                    if ind != ref_ind:
                        stroke.select = True
                    else:
                        stroke.select = False
                bpy.ops.gpencil.delete(type='STROKES')

                # delete all but two points in first stroke
                ref_stroke = l_frame.strokes[0]
                # ref_stroke.line_width = 100
                ref_stroke.select = True
                bpy.ops.gpencil.stroke_change_color(material=white_name)
            
                points_selected = False
                for ind in range(len(ref_stroke.points)):
                    point = ref_stroke.points[ind]
                    if ind > 1:
                        point.select = True
                        points_selected = True
                    else:
                        point.select = False
                if points_selected:
                    bpy.ops.gpencil.dissolve(type='POINTS')

                if FRAME_CENTROIDS:
                    # reposition two points to form first frame centroid
                    f_ind = 0
                    ref_stroke.points[0].co = Vector(world_frame_centroids[0]) + Vector([0, 0.1, 0]) + Vector(offsets[0])
                    ref_stroke.points[1].co = Vector(world_frame_centroids[0]) + Vector([0, -0.1, 0]) + Vector(offsets[0])

                    # duplicate, complete '*'
                    draw_stroke(l_frame, ref_stroke, Vector(world_frame_centroids[0]) + Vector([0.1, 0, 0]) + Vector(offsets[0]), Vector(world_frame_centroids[0]) + Vector([-0.1, 0, 0]) + Vector(offsets[0]))
                    draw_stroke(l_frame, ref_stroke, Vector(world_frame_centroids[0]) + Vector([0, 0, 0.1]) + Vector(offsets[0]), Vector(world_frame_centroids[0]) + Vector([0, 0, -0.1]) + Vector(offsets[0]))

                    # duplicate again, reposition to form second frame centroid
                    f_ind = 1
                    draw_stroke(l_frame, ref_stroke, Vector(world_frame_centroids[1]) + Vector([0.1, 0, 0]) + Vector(offsets[1]), Vector(world_frame_centroids[1]) + Vector([-0.1, 0, 0]) + Vector(offsets[1]))
                    draw_stroke(l_frame, ref_stroke, Vector(world_frame_centroids[1]) + Vector([0, 0, 0.1]) + Vector(offsets[1]), Vector(world_frame_centroids[1]) + Vector([0, 0, -0.1]) + Vector(offsets[1]))
                    draw_stroke(l_frame, ref_stroke, Vector(world_frame_centroids[1]) + Vector([0, 0.1, 0]) + Vector(offsets[1]), Vector(world_frame_centroids[1]) + Vector([0, -0.1, 0]) + Vector(offsets[1]))

        context.area.ui_type = 'DOPESHEET'
        context.space_data.ui_mode = 'GPENCIL'
        bpy.ops.action.select_all(action='DESELECT')

        context.area.type = 'VIEW_3D'
        bpy.ops.object.mode_set(mode='EDIT_GPENCIL')
            
        obj0 = bpy.data.objects['vis.' + str(name) + "." + str(int(matching.frames[0]))]
        obj1 = bpy.data.objects['vis.' + str(name) + "." + str(int(matching.frames[1]))]
        obj0.data.materials.append(gp_mat_white)
        obj1.data.materials.append(gp_mat_white)
        obj0.data.materials.append(gp_mat_gray)
        obj1.data.materials.append(gp_mat_gray)

        # construct color map based on maximum pairing energy
        # max_energy = matching.stroke_energies[-1]['energy']
        max_energy = np.percentile(matching.stroke_energies['energy'], 95)
        viridis = cm.get_cmap('viridis')
        # max_ind = len(matching.stroke_pairs)
        # viridis = cm.get_cmap('viridis')
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                break
        override = {'area': area}

        pair_ind = 1
        for pair0 in matching.stroke_pairs:
            if matching.stroke_energies.size > 1:
                for row in matching.stroke_energies:
                    if row[0] == pair0:
                        energy = row[2]
                        break
                energy = clamp(matching.stroke_energies['energy'], energy)
                color = viridis(abs((max_energy - energy) / max_energy))
            else:
                color = viridis(0)
            # color = viridis(abs((max_ind - pair_ind) / max_ind))
            col_name_og = ''
            for item in color:
                col_name_og = col_name_og + str(item) + ','
            col_name_og = col_name_og[:-1]

            gp_mat = bpy.data.materials.new(col_name_og)
            bpy.data.materials.create_gpencil_data(gp_mat)
            gp_mat.grease_pencil.color = color
            new_obj.data.materials.append(gp_mat)
            obj0.data.materials.append(gp_mat)
            obj1.data.materials.append(gp_mat)

            col_name = search_for_color(new_obj, col_name_og)

            stroke_centroid0 = matching.stroke_centroids[0][pair0] + matching.frame_centroids[0]
            # print(pair0, int(matching.stroke_pairs[pair0]), matching.stroke_centroids[0][pair0], matching.stroke_centroids[1][int(matching.stroke_pairs[pair0])])
            stroke_centroid1 = matching.stroke_centroids[1][int(matching.stroke_pairs[pair0])] + matching.frame_centroids[1]
            
            obj1.select_set(False)
            new_obj.select_set(True)
            context.view_layer.objects.active = new_obj
            for stroke in frame.strokes:
                stroke.select = False

            if not STROKE_PAIRS and not STROKE_CENTROIDS:
                draw_stroke(frame, ref_stroke, Vector(convert_cam_to_world(context, matching.frames[0], stroke_centroid0)) + Vector(offsets[0]), Vector(convert_cam_to_world(context, matching.frames[0], stroke_centroid0)) + Vector(offsets[0]))

            if STROKE_PAIRS:
                # duplicate stroke to draw connection
                draw_stroke(frame, ref_stroke, Vector(convert_cam_to_world(context, matching.frames[0], stroke_centroid0)) + Vector(offsets[0]), Vector(convert_cam_to_world(context, matching.frames[1], stroke_centroid1)) + Vector(offsets[1]))
                frame.strokes[-1].select = True
                bpy.ops.gpencil.stroke_change_color(override, material=col_name)
                frame.strokes[-1].select = False

            if STROKE_CENTROIDS:
                # connect stroke centroids with frame centroids
                draw_stroke(frame, ref_stroke, Vector(convert_cam_to_world(context, matching.frames[0], stroke_centroid0)) + Vector(offsets[0]), world_frame_centroids[0] + Vector(offsets[0]))
                frame.strokes[-1].select = True
                bpy.ops.gpencil.stroke_change_color(override, material=col_name)
                frame.strokes[-1].select = False
                
                draw_stroke(frame, ref_stroke, Vector(convert_cam_to_world(context, matching.frames[1], stroke_centroid1)) + Vector(offsets[1]), world_frame_centroids[1] + Vector(offsets[1]))
                frame.strokes[-1].select = True
                bpy.ops.gpencil.stroke_change_color(override, material=col_name)
                frame.strokes[-1].select = False

            bpy.ops.object.mode_set(mode='EDIT_GPENCIL')

            # recolor corresponding strokes
            new_obj.select_set(False)
            obj0.select_set(True)
            context.view_layer.objects.active = obj0
            col_name = search_for_color(obj0, col_name_og)
            ref_stroke.select = False
            for layer_frame in obj0.data.layers[0].frames:
                if layer_frame.frame_number == 0:
                    frame0 = layer_frame
                    break
            for ind in range(len(frame0.strokes)):
                if pair0 == ind:
                    frame0.strokes[ind].select = True
                else:
                    frame0.strokes[ind].select = False

            bpy.ops.gpencil.stroke_change_color(override, material=col_name)
            frame0.strokes[pair0].select = False

            for layer_frame in obj1.data.layers[0].frames:
                if layer_frame.frame_number == 0:
                    frame1 = layer_frame
                    break

            obj0.select_set(False)
            obj1.select_set(True)
            context.view_layer.objects.active = obj1
            col_name = search_for_color(obj1, col_name_og)
            for ind in range(len(frame1.strokes)):
                if int(matching.stroke_pairs[pair0]) == ind:
                    frame1.strokes[ind].select = True
                else:
                    frame1.strokes[ind].select = False
            bpy.ops.gpencil.stroke_change_color(override, material=col_name)
            frame1.strokes[int(matching.stroke_pairs[pair0])].select = False

            pair_ind += 1

        obj1.select_set(False)
        new_obj.select_set(True)
        context.view_layer.objects.active = new_obj
        ref_stroke.select = True
        if FRAME_CENTROIDS:
            bpy.ops.gpencil.stroke_change_color(override, material=white_name)
        else:
            new_obj.data.layers[0].select = True
            new_obj.data.layers[0].active_frame.select = True
            bpy.ops.object.mode_set(mode='EDIT_GPENCIL')
            bpy.ops.gpencil.delete(override, type='STROKES')
        context.scene.frame_set(matching.frames[0])

        # write matching frames to scene
        bpy.context.scene.collection["matching_frames"] = matching.frames

        bpy.ops.object.mode_set(mode='OBJECT')
        new_obj.select_set(False)


def clamp(arr, item):
    p95 = np.percentile(arr, 95)
    if item > p95:
        return p95
    
    return item


def clamp_90(arr, item):
    p90 = np.percentile(arr, 75)
    if item > p90:
        return p90
    
    return item


def search_for_color(obj, name):
    for material in obj.data.materials:
        index = material.name.find(name)
        if index >= 0:
            return material.name


def convert_cam_to_world(context, frame_number, vec):
    context.scene.frame_set(frame_number)
    cam_to_world = calc_cam_to_world(context, frame_number)
    return np.transpose((cam_to_world @ np.expand_dims(np.append(vec, 1), 1))[:3])[0]


def pd_df_to_array(df):
    array = []
    for ind in range(len(df[0])):
        row = []
        for item in df.loc[ind]:
            row.append(item)
        array.append(row)
    return array


def export_env(context):
    # translate 3D object coordinates into camera space
    camera = context.scene.camera
    obj = context.object
    verts = obj.data.vertices

    screen_verts = []

    M = euler_to_matrix(camera.rotation_euler)
    t = camera.location
    t = np.array([[t[0]], [t[1]], [t[2]]])
    cam_to_world = np.concatenate((M, t), axis=1)
    cam_to_world = np.concatenate(
        (cam_to_world, [[0, 0, 0, 1]]), axis=0)

    world_to_cam = np.linalg.inv(cam_to_world)

    z_coords = []
    for vert in verts:
        cam_vert = world_to_cam @ np.expand_dims(np.append(obj.matrix_world @ vert.co, 1), 1)
        z_coords.append(cam_vert[2])
        screen_verts.append(compute_screen_coordinates(cam_vert))

    z_coords = [z_coords]

    # min_x = screen_verts[0][0]
    # max_x = screen_verts[0][0]
    # min_y = screen_verts[0][1]
    # max_y = screen_verts[0][1]

    # for vert in screen_verts:
    #     if vert[0] < min_x:
    #         min_x = vert[0]
    #     if vert[0] > max_x:
    #         max_x = vert[0]
    #     if vert[1] < min_y:
    #         min_y = vert[1]
    #     if vert[1] > max_y:
    #         max_y = vert[1]

    # boundary_verts = [[min_x, min_y],[min_x, max_y],[max_x, max_y],[max_x, min_y]]
    
    # # calculate 2D boundary of object coordinates in camera space
    boundary_verts = []
    for vert in ConvexHull(screen_verts).vertices:
        boundary_verts.append(screen_verts[int(vert)])

    # export as metadata
    with open("/Users/ilenee/Documents/2020-2021/Thesis/3:9/metadata.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(boundary_verts)

    with open("/Users/ilenee/Documents/2020-2021/Thesis/3:9/z_"+obj.name+".csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(z_coords)


def export_gp(context):
    obj = context.object
    layer = obj.data.layers[0]

    stroke_lengths = []
    for stroke in layer.frames[1].strokes:
        stroke_lengths = np.append(stroke_lengths, len(stroke.points))
    row_length = int(np.max(stroke_lengths))

    for frame_ind in range(1, len(layer.frames) - 1):
        print(frame_ind)
        frame = layer.frames[frame_ind]
        world_to_cam = np.linalg.inv(calc_cam_to_world(context, frame.frame_number))
        z_data = []

        for ind in range(len(frame.strokes)):
            screen_verts = []
            z_row = []
            # output stroke points
            for point in frame.strokes[ind].points:
                cam_coord = world_to_cam @ np.expand_dims(np.append(obj.matrix_world @ point.co, 1), 1)
                z_row.append(cam_coord[2][0])
                screen_vert = compute_screen_coordinates(cam_coord)
                screen_verts.append((screen_vert[0], screen_vert[1]))

            zeros = np.zeros(int(row_length - len(z_row)))
            z_row = np.concatenate([z_row, zeros])
            
            z_data.append(z_row)
     
            with open("/Users/ilenee/Documents/2020-2021/Thesis/3_16/hair/frame" + str(frame.frame_number) + "stroke" + str(ind) + ".csv", "w") as f:
                writer = csv.writer(f)
                writer.writerows(screen_verts)

        with open("/Users/ilenee/Documents/2020-2021/Thesis/3_16/hair/z_frame" + str(frame.frame_number) + ".csv", "w") as f:
            writer = csv.writer(f)
            writer.writerows(z_data)


def calculate_centroid(points):
    centroid = Vector([0, 0, 0])
    for point in points:
        centroid += point.co
    return centroid/len(points)


def export_gp_centroids(context):
    obj = context.object
    layer = obj.data.layers[0]
    camera = context.scene.camera

    M = euler_to_matrix(camera.rotation_euler)
    t = camera.location
    t = np.array([[t[0]], [t[1]], [t[2]]])
    cam_to_world = np.concatenate((M, t), axis=1)
    cam_to_world = np.concatenate(
        (cam_to_world, [[0, 0, 0, 1]]), axis=0)

    world_to_cam = np.linalg.inv(cam_to_world)

    for frame in layer.frames:
        data = []
        for ind in range(len(frame.strokes)):
            # append stroke centroid
            stroke = frame.strokes[ind]
            centroid = calculate_centroid(stroke.points)
            data.append(compute_screen_coordinates(world_to_cam @ np.expand_dims(np.append(obj.matrix_world @ centroid, 1), 1)))
                    
        with open("/Users/ilenee/Documents/2020-2021/Thesis/3:9/" + str(frame.frame_number) + ".csv", "w") as f:
            writer = csv.writer(f)
            writer.writerows(data)


def calc_cam_to_world(context, frame_num):
    context.scene.frame_set(frame_num)
    camera = context.scene.camera

    M = euler_to_matrix(camera.rotation_euler)
    t = camera.location
    # print(camera.location)
    t = np.array([[t[0]], [t[1]], [t[2]]])
    cam_to_world = np.concatenate((M, t), axis=1)
    cam_to_world = np.concatenate(
        (cam_to_world, [[0, 0, 0, 1]]), axis=0)
    return cam_to_world


def import_def(context):
    # calculate unit vectors in x, y
    bpy.ops.object.mode_set(mode='EDIT_GPENCIL')
    obj = context.object
    layer = obj.data.layers[0]

    for frame_ind in range(1, len(layer.frames) - 1):
        frame = layer.frames[frame_ind]
        cam_to_world = calc_cam_to_world(context, frame.frame_number)
        z_data = pd.read_csv("/Users/ilenee/Documents/2020-2021/Thesis/3_16/hair/z_frame"+str(frame.frame_number)+".csv", delimiter = ",", header = None) 

        for stroke_ind in range(len(frame.strokes) - 1):
            stroke = frame.strokes[stroke_ind]

            screen_data = pd.read_csv("/Users/ilenee/Documents/2020-2021/Thesis/3_16/hair/output_frame"+str(frame.frame_number)+"stroke"+str(stroke_ind)+".csv", delimiter = ",", header = None)
            cam_data = []
            for ind, row in screen_data.iterrows():
                if ind < len(stroke.points):
                    cam_data.append(screen_to_cam(row, z_data[ind][stroke_ind]))

            print(frame_ind, stroke_ind, len(cam_data), len(stroke.points))
            
            for point_ind in range(len(stroke.points)):
                point = stroke.points[point_ind]
                world_coords = cam_to_world @ np.expand_dims(np.append(cam_data[point_ind], 1), 1)
                world_coords = world_coords[:-1]
                point.co = world_coords


class ImportGP(bpy.types.Operator):
    """ImportGP"""
    bl_idname = "object.import_gp"
    bl_label = "ImportGP"

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        import_def(context)
        return {'FINISHED'}
        
        
class ExportGPCent(bpy.types.Operator):
    """ExportGP"""
    bl_idname = "object.export_gp_centroids"
    bl_label = "ExportGP"

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        export_gp_centroids(context)
        return {'FINISHED'}


class ExportGPPt(bpy.types.Operator):
    """ExportGP"""
    bl_idname = "object.export_gp_points"
    bl_label = "ExportGP"

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        export_gp(context)
        return {'FINISHED'}


class ExportEnv(bpy.types.Operator):
    """ExportEnv"""
    bl_idname = "object.export_env"
    bl_label = "ExportEnv"

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        export_env(context)
        return {'FINISHED'}


def draw_stroke(l_frame, ref_stroke, pos0, pos1):
    ref_stroke.select = True
    bpy.ops.gpencil.duplicate()
    stroke = l_frame.strokes[-1]
    stroke.points[0].co = pos0
    stroke.points[1].co = pos1


def object_interpolate(context):
    COPIES = int(bpy.context.scene.QueryProps.copies)
    COPY_SPACING = int(bpy.context.scene.QueryProps.copy_spacing)

    # SETUP
    context.area.type = 'VIEW_3D'
    bpy.ops.object.mode_set(mode='OBJECT')
    obj = context.object
    og_name = obj.name
    context.view_layer.objects.active = obj
    layer = obj.data.layers[0]
    bpy.ops.gpencil.stroke_sample(length=RESAMPLE_LENGTH)

    # DUPLICATE AND DISTORT
    for iter in range(1, COPIES):
        bpy.ops.object.duplicate()

        obj = context.object
        layer = obj.data.layers[0]
        for frame in layer.frames:
            for stroke in frame.strokes:
                offset = np.array([(random.random() - 0.5)/RAND_FACTOR, (random.random() - 0.5)/RAND_FACTOR, 0])
                for point in stroke.points:
                    point.co = np.add(point.co, offset)

    # obj = bpy.data.objects[og_name]
    # layer = obj.data.layers[0]
    # print("**()()(()",obj.name)
    
    bpy.ops.object.mode_set(mode='EDIT_GPENCIL')

    for iter in range(0, COPIES):
        if iter < COPIES - 1:
            obj = bpy.data.objects[og_name + '.' + str(COPIES - iter - 1).zfill(3)]
        else:
            obj = bpy.data.objects[og_name]
        layer = obj.data.layers[0]
        context.view_layer.objects.active = obj
        print("**()()(()",obj.name)
        next_ind = 0
        for _ in range(len(layer.frames) - 1):
            if len(layer.frames[next_ind].strokes) > 0 and len(layer.frames[next_ind + 1].strokes) > 0:
                # load stroke pairs
                stroke_pairs = {}
                for key in bpy.context.scene.collection["matching"]:
                    stroke_pairs[int(key)] = int(bpy.context.scene.collection["matching"][key])

                layer_frames = np.array([layer.frames[next_ind], layer.frames[next_ind + 1]])
                
                # load delete0, delete1
                ignore = bpy.context.scene.collection.get('ignore', None)

                delete0 = array.array('i')
                delete1 = array.array('i')
                if ignore is not None:
                    if str(layer.frames[next_ind].frame_number) in ignore:
                        user_specified0 = ignore[str(layer.frames[next_ind].frame_number)][1:].split(" ")
                        for ind in user_specified0:
                            delete0 = np.append(delete0, int(ind))
                    if str(layer.frames[next_ind + 1].frame_number) in ignore:
                        user_specified1 = ignore[str(layer.frames[next_ind + 1].frame_number)][1:].split(" ")
                        for ind in user_specified1:
                            delete1 = np.append(delete1, int(ind))
                # print(len(delete0), len(delete1))
                        
                if len(delete0) == 0:
                    if len(delete1) > 0:
                        stroke_centroids, frames, frame_centroids = calc_stroke_centroids(context, layer_frames, None, delete1)
                    else:
                        stroke_centroids, frames, frame_centroids = calc_stroke_centroids(context, layer_frames)
                else:
                    if len(delete1) == 0:
                        stroke_centroids, frames, frame_centroids = calc_stroke_centroids(context, layer_frames, delete0, None)
                    else:
                        stroke_centroids, frames, frame_centroids = calc_stroke_centroids(context, layer_frames, delete0, delete1)

                matching = Matching(stroke_pairs, None, stroke_centroids, frames, frame_centroids, delete0, delete1)
                # print("before interpolate frames")
                next_ind = interpolate_frames(context, layer, next_ind, matching)

                # print("end loop")
            else:
                # skip to next frame if this frame is empty
                next_ind = next_ind + 1
            if (next_ind >= len(layer.frames)):
                break
        # if iter < COPIES - 1:
        #     obj = bpy.data.objects[og_name + '.' + str(COPIES - iter - 1).zfill(3)]
        #     layer = obj.data.layers[0]
        #     context.view_layer.objects.active = obj
        # else:
        #     obj = bpy.data.objects[og_name]
        #     layer = obj.data.layers[0]
        #     context.view_layer.objects.active = obj

    # ("leave loop")

    # OFFSET DUPLICATES

    obj = bpy.data.objects[og_name]
    context.view_layer.objects.active = obj
    layer = obj.data.layers[0]

    context.area.ui_type = 'DOPESHEET'
    context.space_data.ui_mode = 'GPENCIL'
    layer.lock = True

    for iter in range(1, COPIES):
        obj = bpy.data.objects[og_name + '.' + str(iter).zfill(3)]
        layer = obj.data.layers[0]
        obj.select_set(True)

        bpy.ops.action.select_all(action='SELECT')
        bpy.ops.transform.transform(mode='TIME_TRANSLATE', value=(COPY_SPACING, 0, 0, 0), orient_type='VIEW', orient_matrix=((-1, -0, -0), (-0, -1, -0), (-0, -0, -1)), orient_matrix_type='VIEW', mirror=True)

        obj.select_set(False)

        layer.lock = True

    bpy.ops.action.select_all(action='DESELECT')

    context.area.type = 'PROPERTIES'


class Interpolate(bpy.types.Operator):
    """Interpolate"""
    bl_idname = "object.interpolate"
    bl_label = "Interpolate"

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        object_interpolate(context)
        return {'FINISHED'}


class ModalOperator(bpy.types.Operator):
    """Move an object with the mouse, example"""
    bl_idname = "object.modal_operator"
    bl_label = "Simple Modal Operator"

    def modal(self, context, event):
        if event.type == 'LEFTMOUSE' and len(context.object.name) > 4 and context.object.name[:4] == 'vis.' and not context.object.name[4] == 'p':
            obj = context.object
            curr_frame = obj.data.layers[0].active_frame
            prefix_end = obj.name.rindex(".")
            curr_frame_num = int(obj.name[prefix_end+1:])
            prefix = obj.name[0:prefix_end+1]
            
            if context.scene.frame_current == curr_frame_num:
                context.scene.camera.data.type = 'ORTHO'
                # context.area.type = 'VIEW_3D'

                for area in context.screen.areas:
                    if area.type == "VIEW_3D":
                        break

                for region in area.regions:
                    if region.type == "WINDOW":
                        break

                space = area.spaces[0]

                override = context.copy()
                override['area'] = area
                override['region'] = region
                override['space_data'] = space
                override['scene'].cursor.location = obj.location

                bpy.ops.view3d.view_axis(override, 'EXEC_DEFAULT', type='TOP')
                bpy.ops.view3d.view_center_cursor(override)

                override = {'area': area}

                frames = []

                frames = np.append(frames, (bpy.context.scene.collection["matching_frames"][0]))
                frames = np.append(frames, (bpy.context.scene.collection["matching_frames"][1]))

                if curr_frame_num == int(frames[0]):
                    other_frame = int(frames[1])
                else:
                    other_frame = int(frames[0])
                
                other_obj = bpy.data.objects[prefix + str(other_frame)]

                curr_stroke = -1

                bpy.ops.object.mode_set(mode='EDIT_GPENCIL')

                for stroke_ind in range(len(curr_frame.strokes)):
                    if curr_frame.strokes[stroke_ind].select == True:
                        curr_stroke = stroke_ind

                og_width = bpy.context.scene.collection.get('og_width', None)
                if og_width is None:
                    og_width = curr_frame.strokes[0].line_width
                    bpy.context.scene.collection["og_width"] = og_width

                if curr_stroke >= 0:
                    for stroke_ind in range(len(curr_frame.strokes)):
                        if not stroke_ind == curr_stroke:
                            curr_frame.strokes[stroke_ind].line_width = og_width
                        else:
                            curr_frame.strokes[stroke_ind].line_width = og_width * 3

                    # load stroke_pairs, select relevant
                    for key in bpy.context.scene.collection["matching"]:
                        if frames[0] == curr_frame_num and int(key) == curr_stroke:
                            match_ind = int(bpy.context.scene.collection["matching"][key])
                        elif not frames[0] == curr_frame_num and int(bpy.context.scene.collection["matching"][key]) == curr_stroke:
                            match_ind = int(key)

                    # load energies, select relevant (pertaining to curr_stroke)
                    energy_rows = bpy.context.scene.collection["matching_energies"].split(',')
                    energies = None
                    for row in energy_rows:
                        data = row.split(' ')
                        data[0] = int(data[0])
                        data[1] = int(data[1])
                        data[2] = float(data[2])
                        if energies is None:
                            energies = np.array((data[0], data[1], data[2]), dtype=[('first', 'i'), ('second', 'i'), ('energy', 'f')])
                        else:
                            energies = np.append(energies, np.array((data[0], data[1], data[2]), dtype=[('first', 'i'), ('second', 'i'), ('energy', 'f')]))
                    if frames[0] == curr_frame_num:
                        condition = energies['first'] == curr_stroke
                        other_col = 1
                    else:
                        condition = energies['second'] == curr_stroke
                        other_col = 0
                    extracted_energies = np.extract(condition, energies)

                    # select other_obj
                    obj.select_set(False)
                    other_obj.select_set(True)
                    context.view_layer.objects.active = other_obj
                    frame = other_obj.data.layers[0].active_frame

                    # iterate through, recolor
                    if extracted_energies.size > 1:
                        extracted_energies_sorted = np.sort(extracted_energies, order='energy')
                        bpy.context.scene.collection["range"] = str(extracted_energies_sorted[0]['energy']) + ' ' + str(extracted_energies_sorted[-1]['energy'])
                    else:
                        extracted_energies_sorted = extracted_energies
                        bpy.context.scene.collection["range"] = str(extracted_energies_sorted['energy']) + ' ' + str(extracted_energies_sorted['energy'])
                    max_energy = np.percentile(extracted_energies_sorted['energy'], 95)
                    viridis = cm.get_cmap('viridis')

                    for row in extracted_energies:
                        for stroke_ind in range(len(frame.strokes)):
                            if stroke_ind == row[other_col]:
                                frame.strokes[stroke_ind].select = True
                            else:
                                frame.strokes[stroke_ind].select = False
                            energy = row[2]
                            energy = clamp(extracted_energies_sorted['energy'], energy)

                            # recolor according to energy
                            color = list(viridis(abs((max_energy - energy) / max_energy)))
                            color[-1] = 0.3
                            col_name = ''
                            for item in color:
                                col_name = col_name + str(item) + ','
                            col_name = col_name[:-1]

                            gp_mat = bpy.data.materials.new(col_name)
                            bpy.data.materials.create_gpencil_data(gp_mat)
                            gp_mat.grease_pencil.color = color
                            other_obj.data.materials.append(gp_mat)

                            col_name = search_for_color(other_obj, col_name)
                            bpy.ops.gpencil.stroke_change_color(override, material=col_name)

                            # set line_width
                            if stroke_ind == match_ind:
                                frame.strokes[stroke_ind].line_width = og_width * 3
                            else:
                                frame.strokes[stroke_ind].line_width = og_width

                    # restore original selection state
                    other_obj.select_set(False)
                    obj.select_set(True)
                    context.view_layer.objects.active = obj
                    frame = obj.data.layers[0].active_frame

                    for stroke_ind in range(len(frame.strokes)):
                        if stroke_ind == curr_stroke:
                            frame.strokes[stroke_ind].select = True
                        else:
                            frame.strokes[stroke_ind].select = False
        elif event.type == 'ESC':  # finish
            return {'FINISHED'}

        return {'PASS_THROUGH'}
        

    def invoke(self, context, event):
        if context.object:
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "No active object, could not finish")
            return {'CANCELLED'}


class Match(bpy.types.Operator):
    """Match"""
    bl_idname = "object.match"
    bl_label = "Match"

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        object_match(context)
        return {'FINISHED'}


class Examine(bpy.types.Operator):
    """Examine"""
    bl_idname = "object.examine"
    bl_label = "Examine"

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        bpy.ops.object.modal_operator('INVOKE_DEFAULT')
        return {'FINISHED'}


def ignore(context):
    obj = context.object
    layer = obj.data.layers[0]
    strokes = layer.active_frame.strokes
    for ind in range(len(strokes)):
        if strokes[ind].select is True:
            ignore = bpy.context.scene.collection.get('ignore', None)
            if ignore is None:
                bpy.context.scene.collection["ignore"] = {}
                bpy.context.scene.collection["ignore"][str(layer.active_frame.frame_number)] = " " + str(ind)
            elif str(layer.active_frame.frame_number) not in ignore:
                bpy.context.scene.collection["ignore"][str(layer.active_frame.frame_number)] = " " + str(ind)
            elif " " + str(ind) not in bpy.context.scene.collection["ignore"][str(layer.active_frame.frame_number)]:
                bpy.context.scene.collection["ignore"][str(layer.active_frame.frame_number)] += " " + str(ind)


class Ignore(bpy.types.Operator):
    """Ignore"""
    bl_idname = "object.ignore"
    bl_label = "Ignore"

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        ignore(context)
        return {'FINISHED'}


def pair0(context):
    obj = context.object
    curr_frame = obj.data.layers[0].active_frame
    prefix_end = obj.name.rindex(".")
    curr_frame_num = int(obj.name[prefix_end+1:])
    prefix = obj.name[0:prefix_end+1]

    strokes = curr_frame.strokes
    selected = None
    for ind in range(len(strokes)):
        if strokes[ind].select is True:
            selected = ind
            break

    frames = []

    frames = np.append(frames, (bpy.context.scene.collection["matching_frames"][0]))
    frames = np.append(frames, (bpy.context.scene.collection["matching_frames"][1]))

    if frames[0] == curr_frame_num:
        bpy.context.scene.collection["pair_cache"] = "0 " + str(selected)
    else:
        bpy.context.scene.collection["pair_cache"] = "1 " + str(selected)

    if curr_frame_num == int(frames[0]):
        other_frame = int(frames[1])
    else:
        other_frame = int(frames[0])

    context.scene.frame_set(other_frame)
                
    other_obj = bpy.data.objects[prefix + str(other_frame)]
    other_obj.select_set(True)
    obj.select_set(False)
    context.view_layer.objects.active = other_obj
    bpy.ops.object.mode_set(mode='EDIT_GPENCIL')


def pair1(context):
    obj = context.object
    curr_frame = obj.data.layers[0].active_frame
    strokes = curr_frame.strokes
    selected = None
    for ind in range(len(strokes)):
        if strokes[ind].select is True:
            selected = ind
            break

    pair_cache = bpy.context.scene.collection.get('pair_cache', None)
    if pair_cache is not None:
        pair_cache_arr = pair_cache.split(" ")

        pair = bpy.context.scene.collection.get('pair', None)
        if pair is None:
            if int(pair_cache_arr[0]) == 0:
                bpy.context.scene.collection["pair"] = str(pair_cache_arr[1]) + " " + str(selected) + ","
            else:
                bpy.context.scene.collection["pair"] = str(selected) + " " + str(pair_cache_arr[1]) + ","
        else:
            if int(pair_cache_arr[0]) == 0 and str(pair_cache_arr[1]) + " " + str(selected) not in pair:
                bpy.context.scene.collection["pair"] += str(pair_cache_arr[1]) + " " + str(selected) + ","
            elif str(selected) + " " + str(pair_cache_arr[1]) not in pair:  
                bpy.context.scene.collection["pair"] += str(selected) + " " + str(pair_cache_arr[1]) + ","


def pair(context):
    obj = context.object
    layer = obj.data.layers[0]
    strokes0 = layer.frames[0].strokes
    strokes1 = layer.frames[1].strokes
    if len(strokes0) == 0:
        strokes0 = layer.frames[1].strokes
        strokes1 = layer.frames[2].strokes
    
    selected0 = None
    for ind in range(len(strokes0)):
        if strokes0[ind].select is True:
            selected0 = ind
            break

    selected1 = None
    for ind in range(len(strokes1)):
        if strokes1[ind].select is True:
            selected1 = ind
            break

    if selected0 is None or selected1 is None:
        return

    pair = bpy.context.scene.collection.get('pair', None)
    if pair is None:
        bpy.context.scene.collection["pair"] = str(selected0) + " " + str(selected1) + ","
    elif str(selected0) + " " + str(selected1) + "," not in pair:
        bpy.context.scene.collection["pair"] += str(selected0) + " " + str(selected1) + ","


class LoadImage(bpy.types.Operator):
    bl_idname      = "object.loadimage"
    bl_label       = "LoadImage"
    bl_description = "Load images"
    
    @classmethod
    def poll(cls, context):
        return True
    
    def execute(self, context):
        bpy.ops.image.open(filepath='/Users/ilenee/Documents/viridis.png')
        bpy.data.images['viridis.png'].pack()
        image = bpy.data.images['viridis.png']
        tex = bpy.data.textures.new('viridis', type='IMAGE')
        tex.image = image

        return {'FINISHED'}


class Pair(bpy.types.Operator):
    """Pair"""
    bl_idname = "object.pair"
    bl_label = "Pair"

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        pair(context)
        return {'FINISHED'}


class Pair0(bpy.types.Operator):
    """Pair0"""
    bl_idname = "object.pair0"
    bl_label = "Pair0"

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        pair0(context)
        return {'FINISHED'}


class Pair1(bpy.types.Operator):
    """Pair1"""
    bl_idname = "object.pair1"
    bl_label = "Pair1"

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        pair1(context)
        return {'FINISHED'}


class TestPanel(bpy.types.Panel):
    """Creates a Panel in the Object properties window"""
    bl_label = "Interpolate Panel"
    bl_idname = "OBJECT_PT_interpolate"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "object"

    def draw(self, context):
        props = bpy.context.scene.QueryProps
        layout = self.layout

        obj = context.object

        row = layout.row()
        row.label(text="Calculate energies using: " + obj.name)

        row = layout.row()
        row.prop(context.scene, "position")

        row = layout.row()
        row.prop(context.scene, "shape_context")

        row = layout.row()
        row.prop(context.scene, "dot_product")

        row = layout.row()
        row.label(text="Visualize matching: " + obj.name)

        row = layout.row()
        row.prop(context.scene, "pairings")

        row = layout.row()
        row.prop(context.scene, "frame_centroids")

        row = layout.row()
        row.prop(context.scene, "stroke_centroids")

        row = layout.row()
        row.prop(context.scene, "animate_outliers_bool")

        row = layout.row()
        row.label(text="Active object is: " + obj.name)

        row = layout.row()
        row.operator("object.pair" , text = "Pair strokes")

        row = layout.row()
        row.operator("object.pair0" , text = "Pair stroke 1")

        row = layout.row()
        row.operator("object.pair1" , text = "Pair stroke 2")

        row = layout.row()
        row.operator("object.ignore" , text = "Select strokes to ignore")

        for frame in bpy.context.scene.collection["ignore"]:
            row = layout.row()
            row.label(text="Strokes to ignore in frame " + frame + ":" + bpy.context.scene.collection["ignore"][frame])

        col = layout.column(align=True)
        rowsub = col.row(align=True)

        rowsub.label(text="Cutoff threshold")

        rowsub = col.row(align=True)
        rowsub.prop(props, "query", text="")
        
        row = layout.row()

        en_range = bpy.context.scene.collection.get('range', None)

        if en_range is not None:
            en_range = en_range.split(" ")
            row = layout.row()
            row.label(text="Stroke pairing energies")

            row = layout.row()
            row.template_preview(bpy.data.textures["viridis"])

            row = layout.row()
            split = row.split(factor=0.5)
            split.label(text=en_range[0])
            for _ in range(15):
                split = row.split()
            split.label(text=en_range[1])

        row = layout.row() 

        row = layout.row()
        row.operator("object.match" , text = "Create Matching")

        row = layout.row()
        row.operator("object.examine" , text = "Examine stroke")

        col = layout.column(align=True)
        rowsub = col.row(align=True)

        rowsub.label(text="Regenerate copies")

        rowsub = col.row(align=True)
        rowsub.prop(props, "copies", text="")

        col = layout.column(align=True)
        rowsub = col.row(align=True)

        rowsub.label(text="Copy spacing")

        rowsub = col.row(align=True)
        rowsub.prop(props, "copy_spacing", text="")

        row = layout.row()
        row.operator("object.interpolate" , text = "Interpolate")

        row = layout.row()
        row.operator("object.export_env" , text = "Export Env")

        row = layout.row()
        row.operator("object.export_gp_centroids" , text = "Export GP Centroids")

        row = layout.row()
        row.operator("object.export_gp_points" , text = "Export GP Points")

        row = layout.row()
        row.operator("object.import_gp" , text = "Import GP")

        row = layout.row()


class QueryProps(bpy.types.PropertyGroup):

    query: bpy.props.StringProperty(default="1.5")
    copies: bpy.props.StringProperty(default="1")
    copy_spacing: bpy.props.StringProperty(default="5")


def register():
    bpy.context.scene.collection["ignore"] = {}
    bpy.context.scene.collection["pair"] = None
    bpy.types.Scene.position = bpy.props.BoolProperty(
        name = "Relative position",
        default = True
        )
    bpy.types.Scene.shape_context = bpy.props.BoolProperty(
        name = "Shape context",
        default = True
        )
    bpy.types.Scene.dot_product = bpy.props.BoolProperty(
        name = "Dot product",
        default = True
        )
    bpy.types.Scene.pairings = bpy.props.BoolProperty(
        name = "Stroke pairs",
        default = True
        )
    bpy.types.Scene.frame_centroids = bpy.props.BoolProperty(
        name = "Frame centroids",
        default = True
        )
    bpy.types.Scene.stroke_centroids = bpy.props.BoolProperty(
        name = "Stroke centroids",
        default = True
        )
    bpy.types.Scene.animate_outliers_bool = bpy.props.BoolProperty(
        name = "Animate unmatched strokes",
        default = True
        )
    bpy.utils.register_class(LoadImage)
    bpy.utils.register_class(ImportGP)
    bpy.utils.register_class(QueryProps)
    bpy.types.Scene.QueryProps = bpy.props.PointerProperty(type=QueryProps)
    bpy.ops.object.loadimage()
    bpy.utils.register_class(TestPanel)
    bpy.utils.register_class(Interpolate)
    bpy.utils.register_class(Match)
    bpy.utils.register_class(Examine)
    bpy.utils.register_class(Ignore)
    bpy.utils.register_class(Pair)
    bpy.utils.register_class(Pair0)
    bpy.utils.register_class(Pair1)
    bpy.utils.register_class(ModalOperator)
    bpy.utils.register_class(ExportEnv)
    bpy.utils.register_class(ExportGPCent)
    bpy.utils.register_class(ExportGPPt)


def unregister():
    del bpy.types.Scene.position
    del bpy.types.Scene.shape_context
    del bpy.types.Scene.dot_product
    del bpy.types.Scene.pairings
    del bpy.types.Scene.frame_centroids
    del bpy.types.Scene.stroke_centroids
    del bpy.types.Scene.animate_outliers_bool
    del(bpy.types.Scene.QueryProps)
    bpy.utils.unregister_class(TestPanel)
    bpy.utils.unregister_class(ImportGP)
    bpy.utils.unregister_class(QueryProps)
    bpy.utils.unregister_class(LoadImage)
    bpy.utils.unregister_class(Interpolate)
    bpy.utils.unregister_class(Match)
    bpy.utils.unregister_class(Examine)
    bpy.utils.unregister_class(Ignore)
    bpy.utils.unregister_class(Pair)
    bpy.utils.unregister_class(Pair0)
    bpy.utils.unregister_class(Pair1)
    bpy.utils.unregister_class(ModalOperator)
    bpy.utils.unregister_class(ExportEnv)
    bpy.utils.unregister_class(ExportGPCent)
    bpy.utils.unregister_class(ExportGPPt)


if __name__ == "__main__":
    register()