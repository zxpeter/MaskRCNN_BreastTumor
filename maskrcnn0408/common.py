# -*- coding: utf-8 -*-
# File: common.py

import numpy as np
import cv2
from tensorpack import *
import copy 
from tensorpack.dataflow import RNGDataFlow
from tensorpack.dataflow.imgaug import transform


class DataFromListOfDict(RNGDataFlow):
    def __init__(self, lst, keys, shuffle=False):
        self._lst = lst
        self._keys = keys
        self._shuffle = shuffle
        self._size = len(lst)

    def __len__(self):
        return self._size

    def __iter__(self):
        if self._shuffle:
            self.rng.shuffle(self._lst)
        for dic in self._lst:
            dp = [dic[k] for k in self._keys]
            yield dp


class ElasticDeform:
    def __init__(self):
        pass
    
    def reset_state(self):
        self.rng = get_rng(self)
        
    def _augment(self, img, param):
        assert img.ndim in [2, 3], img.ndim
        DU, DV = param
        shape = img.shape
        
        X, Y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(Y+DV, (-1, 1)), np.reshape(X+DU, (-1, 1))
        
        flow = copy.deepcopy(img)
        from scipy.ndimage.interpolation import map_coordinates
        for k in range(3):
            tmp = map_coordinates(img[...,k], indices, order=1)
            flow[...,k] = tmp.reshape(shape[0:2])
        flow = flow.reshape(shape)
        return flow

    def _get_augment_params(self, img):
        """
        get the augmentor parameters
        """
        size = self.rng.choice(range(8,30)) #8
        ampl = self.rng.choice(range(8,30)) #8
        du = self.rng.uniform(-ampl, ampl, size=(size, size))
        dv = self.rng.uniform(-ampl, ampl, size=(size, size)) 
        
        # Dont distort at the boundaries
        du[ 0,:] = 0; du[-1,:] = 0; du[:, 0] = 0; du[:,-1] = 0;
        dv[ 0,:] = 0; dv[-1,:] = 0; dv[:, 0] = 0; dv[:,-1] = 0;
        shape = img.shape
        DU = cv2.resize(du, (shape[1], shape[0])) 
        DV = cv2.resize(dv, (shape[1], shape[0])) 

        return DU, DV
    def _augment_return_params(self, d):
        """
        Augment the image and return both image and params
        """
        prms = self._get_augment_params(d)
        return (self._augment(d, prms), prms)

    def _augment_coords(self, coords, param):
        DU, DV = param
        coords = np.asarray(coords, dtype='int32')
        coords, DU, DV = coords.tolist(), DU.tolist(), DV.tolist()

        print(coords)
        new_coords = copy.deepcopy(coords)
        for i in range(len(coords)):
            new_coords[i][0] = coords[i][0] + DU[coords[i][1]][coords[i][0]]
            new_coords[i][1] = coords[i][1] + DV[coords[i][1]][coords[i][1]]
        return np.array(new_coords)


class CustomResize(transform.TransformAugmentorBase):
    """
    Try resizing the shortest edge to a certain number
    while avoiding the longest edge to exceed max_size.
    """

    def __init__(self, short_edge_length, max_size, interp=cv2.INTER_LINEAR):
        """
        Args:
            short_edge_length ([int, int]): a [min, max] interval from which to sample the
                shortest edge length.
            max_size (int): maximum allowed longest edge length.
        """
        super(CustomResize, self).__init__()
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        self._init(locals())

    def _get_augment_params(self, img):
        h, w = img.shape[:2]
        size = self.rng.randint(
            self.short_edge_length[0], self.short_edge_length[1] + 1)
        scale = size * 1.0 / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > self.max_size:
            scale = self.max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return transform.ResizeTransform(h, w, newh, neww, self.interp)


def box_to_point8(boxes):
    """
    Args:
        boxes: nx4

    Returns:
        (nx4)x2
    """
    b = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]]
    b = b.reshape((-1, 2))
    return b


def point8_to_box(points):
    """
    Args:
        points: (nx4)x2
    Returns:
        nx4 boxes (x1y1x2y2)
    """
    p = points.reshape((-1, 4, 2))
    minxy = p.min(axis=1)   # nx2
    maxxy = p.max(axis=1)   # nx2
    return np.concatenate((minxy, maxxy), axis=1)


def segmentation_to_mask(polys, height, width):
    """
    Convert polygons to binary masks.

    Args:
        polys: a list of nx2 float array. Each array contains many (x, y) coordinates.

    Returns:
        a binary matrix of (height, width)
    """
    polys = [p.flatten().tolist() for p in polys]
    assert len(polys) > 0, "Polygons are empty!"

    import pycocotools.mask as cocomask
    rles = cocomask.frPyObjects(polys, height, width)
    rle = cocomask.merge(rles)
    return cocomask.decode(rle)


def clip_boxes(boxes, shape):
    """
    Args:
        boxes: (...)x4, float
        shape: h, w
    """
    orig_shape = boxes.shape
    boxes = boxes.reshape([-1, 4])
    h, w = shape
    boxes[:, [0, 1]] = np.maximum(boxes[:, [0, 1]], 0)
    boxes[:, 2] = np.minimum(boxes[:, 2], w)
    boxes[:, 3] = np.minimum(boxes[:, 3], h)
    return boxes.reshape(orig_shape)


def filter_boxes_inside_shape(boxes, shape):
    """
    Args:
        boxes: (nx4), float
        shape: (h, w)

    Returns:
        indices: (k, )
        selection: (kx4)
    """
    assert boxes.ndim == 2, boxes.shape
    assert len(shape) == 2, shape
    h, w = shape
    indices = np.where(
        (boxes[:, 0] >= 0) &
        (boxes[:, 1] >= 0) &
        (boxes[:, 2] <= w) &
        (boxes[:, 3] <= h))[0]
    return indices, boxes[indices, :]


try:
    import pycocotools.mask as cocomask

    # Much faster than utils/np_box_ops
    def np_iou(A, B):
        def to_xywh(box):
            box = box.copy()
            box[:, 2] -= box[:, 0]
            box[:, 3] -= box[:, 1]
            return box

        ret = cocomask.iou(
            to_xywh(A), to_xywh(B),
            np.zeros((len(B),), dtype=np.bool))
        # can accelerate even more, if using float32
        return ret.astype('float32')

except ImportError:
    from utils.np_box_ops import iou as np_iou  # noqa
