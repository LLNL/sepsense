#!/usr/bin/env python3

import torch
import PIL.ImageFilter
from PIL import Image, ImageDraw
import numpy as np
import scipy.misc
from imsim import sensor_model_degrade_dg as sd    
from pprint import pprint
import traceback
import skimage.transform


class TargetCrop(object):
    def __init__(self, r, modality, pad_mode):
        if modality == 'rgb':
            self.hr = r*2
        elif modality == 'ms3':
            self.hr = r//2
        self.modality = modality
        self.pad_mode = pad_mode

    def __call__(self, data):
        dgim, bbox, path = data
        # Get coordinates of bbox
        th, tw, tc = dgim.im.shape
        x0, y0, w, h = bbox
        x1, y1 = x0+w, y0+h
        # Get coordinates of bbox center
        xc = (x0+x1)//2
        yc = (y0+y1)//2
        # Get new image bounds
        xn0 = xc - self.hr
        xn1 = xc + self.hr
        yn0 = yc - self.hr
        yn1 = yc + self.hr
        # Return a new bbox
        bbox = (x0-xn0, y0-yn0, w, h)
        # Clip bounds and determine padding constants
        xp0, yp0, xp1, yp1 = 0, 0, 0, 0
        if xn0 < 0:
            xp0 = abs(xn0)
            xn0 = 0
        if yn0 < 0:
            yp0 = abs(yn0)
            yn0 = 0
        if xn1 > tw:
            xp1 = xn1 - tw
            xn1 = tw
        if yn1 > th:
            yp1 = yn1 - th
            yn1 = th
        # Crop the image
        dgim.im = dgim.im[yn0:yn1, xn0:xn1]
        # Pad the image
        dgim.im = np.pad(dgim.im, ((yp0, yp1), (xp0, xp1), (0, 0)), 
            mode=self.pad_mode)
        # Save off crop information
        dgim.meta['padding'] = (xp0, dgim.im.shape[1]-xp1, yp0, dgim.im.shape[0]-yp1)

        return dgim, bbox, path

class CenterCrop(object):
    def __init__(self, r, modality, pad_mode):
        if modality == 'rgb':
            self.hr = r*2
        elif modality == 'ms3':
            self.hr = r//2
        self.modality = modality
        self.pad_mode = pad_mode

    def __call__(self, data):
        dgim, bbox, path = data
        # Get coordinates of bbox
        th, tw, tc = dgim.im.shape
        x0, y0, w, h = bbox
        x1, y1 = x0+w, y0+h
        # Get coordinates of bbox center
        xc = (x0+x1)//2
        yc = (y0+y1)//2
        # Center crop limits
        if True:
            hr = min(tw - xc, xc, th - yc, yc, self.hr)
        # Get new image bounds
        xn0 = xc - hr
        xn1 = xc + hr
        yn0 = yc - hr
        yn1 = yc + hr
        # Return a new bbox
        bbox = (x0-xn0, y0-yn0, w, h)
        # Clip bounds and determine padding constants
        xp0, yp0, xp1, yp1 = 0, 0, 0, 0
        if xn0 < 0:
            xp0 = abs(xn0)
            xn0 = 0
        if yn0 < 0:
            yp0 = abs(yn0)
            yn0 = 0
        if xn1 > tw:
            xp1 = xn1 - tw
            xn1 = tw
        if yn1 > th:
            yp1 = yn1 - th
            yn1 = th
        # Crop the image
        dgim.im = dgim.im[yn0:yn1, xn0:xn1]
        dgim.im = skimage.transform.resize(dgim.im, (self.hr, self.hr), order=1)

        return dgim, bbox, path

class TargetCrop2:
    def __call__(self, data):
        dgim, bbox, path = data
        # Get crop information
        xp0, xp1, yp0, yp1 = dgim.meta['padding']
        dgim.im = dgim.im[yp0:yp1, xp0:xp1]
        # Modify bbox
        x0, y0, w, h = bbox
        bbox = 0, 0, dgim.im.shape[1], dgim.im.shape[0]

        return dgim, bbox, path

class TargetExtract(object):
    def __init__(self, modality, pc=25):
        self.modality = modality
        self.pc = pc

    def __call__(self, data):
        dgim, bbox, path = data
        th, tw, tc = dgim.im.shape
        # Get coordinates of bbox
        x0, y0, w, h = bbox
        x1, y1 = x0+w, y0+h
        # Add padding context
        if self.pc > 0:
            x0, y0 = x0-self.pc, y0-self.pc
            x1, y1 = x1+self.pc, y1+self.pc
            dgim.meta['pc'] = self.pc
        # Fix values outside of bounds
        if self.pc > 0:
            if x0 < 0 or y0 < 0 or x1 > tw or y1 > th:
                xp0 = abs(x0) if x0 < 0 else 0
                yp0 = abs(y0) if y0 < 0 else 0
                xp1 = x1-tw if x1 > tw else 0
                yp1 = y1-th if y1 > th else 0
                # Crop the target to a square
                dgim.im = dgim.im[max(0, y0):min(th, y1), max(0, x0):min(tw, x1)]
                dgim.im = np.pad(dgim.im, ((yp0, yp1), (xp0, xp1), (0, 0)), mode='edge')
            else:
                # Crop the target to a square
                dgim.im = dgim.im[y0:y1, x0:x1]
        else:
            # Crop the target to a square
            dgim.im = dgim.im[y0:y1, x0:x1]
        # Create new bbox
        bbox = 0, 0, w, h
        
        return dgim, bbox, path

def resize_image(img, size, interp_type, interp_range):
    if interp_range == 'default':
        img = scipy.misc.imresize(img, size, interp=interp_type)
    elif interp_range == 'noscale':
        # Determine interpolation order
        order = 0 if interp_type == 'nearest' else 1
        # Squash image range to [0, 1]
        min_val = img.min()
        img = img - min_val
        max_val = img.max()
        img = img / max_val
        # Resize image with 'order' interpolation, needs image in [0, 1]
        img = skimage.transform.resize(img, size, order=order)
        # Rescale image back to original range
        img = (img * max_val) + min_val

    return img

class TargetResize(object):
    def __init__(self, r, modality, interp_type, interp_range):
        self.r = r
        self.modality = modality
        self.interp_type = interp_type
        self.interp_range = interp_range

    def __call__(self, data):
        img, bbox, path = data
        _, _, w, h = bbox
        # Crop the image
        if type(img) != np.ndarray:
            img = img.im
        img = img[:h, :w]
        # Resize image
        img = resize_image(img, (self.r, self.r), self.interp_type, self.interp_range)
        
        return img, bbox, path


class GaussianBlur(object):
    def __init__(self, radius):
        self.filter = PIL.ImageFilter.GaussianBlur(radius=radius)

    def __call__(self, data):
        img, bbox, path = data
        blurred = img.filter(self.filter)
        return blurred, bbox, path

class Denormalize(object):
    def __init__(self, mean=None, std=None):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, tsr):
        arr = tsr.numpy()
        norm = (np.moveaxis(arr, 0, -1) * self.std) + self.mean 
        norm = norm * 255.0
        img = Image.fromarray(norm.astype(np.uint8))
        b, g, r = img.split()
        img = Image.merge("RGB", (r, g, b))
        return img


class DrawBBox(object):
    def __init__(self, thickness):
        self.thickness = thickness

    def __call__(self, data):
        img, bbox, path = data
        # Get bbox coordinates
        x0, y0, w, h = bbox
        x1, y1 = x0+w, y0+h
        # Draw concentric boxes for each thickness level
        d = ImageDraw.Draw(img)
        for i in range(self.thickness):
            d.rectangle([x0-i, y0-i, x1+i, y1+i], outline='black')
        return img


class DeTuple(object):
    def __call__(self, data):
        if type(data) == tuple:
            return data[0]
        else:
            return data

class ToTensor(object):
    def __call__(self, data):
        return torch.FloatTensor(data).permute(2, 0, 1)

class ToReflectance(object):
    def __call__(self, data):
        dgim, bbox, path = data
        #orig_min, orig_max = dgim.im.min(), dgim.im.max()
        #print(orig_min, orig_max)
        dgim = sd.convert_to_radiance(dgim)
        dgim.im = sd.convert_to_toa_reflectance(dgim.im, dgim.meta)
        dgim.meta.update({'units': 2})
        return dgim, bbox, path

class SensorError(Exception):
    def __init__(self, message, data, traceback):

        # Call the base class constructor with the parameters it needs
        super().__init__(message)

        # Now for your custom code...
        self.data = data
        self.traceback = traceback


class Satellite(object):

    def __init__(self, sensor_params, modality, interp_type, interp_range, out_units):
        # Create image system
        self.sensor_params = sensor_params
        self.modality = modality
        self.interp_type = interp_type
        self.interp_range = interp_range
        self.sat = sd.imageSystem(**sensor_params)
        if out_units == 'radiance':
            self.out_units = 1
        elif out_units == 'reflectance':
            self.out_units = 2
        else:
            raise Exception('Invalid out_units: {}'.format(out_units))

    def __call__(self, data):
        dgim, bbox, img_path = data
        h, w, c = dgim.im.shape
        # For approximating radiance!!!!
        dgim = sd.convert_to_radiance(dgim)
        dgim.makeSquare()

        try:
            new_dgim = sd.degradeIm(dgim, self.sat, displayImages=0, im_out_units=self.out_units)
        except ValueError:
            raise SensorError('Sensor failure occurred with the bundled params.', self.sensor_params, traceback.format_exc())

        # If out units are in radiance, we need to normalize
        # XXX: scipy.misc.imresize scales to [0, 255]. This may be a problem...
        if self.out_units == 1:
            pass

        # Adjust expected size
        new_dgim.adjustBoundingBox()
        # Crop off zero-padding
        new_dgim.restoreDimensions()

        # Resize image to original size
        try:
            # Resize image
            new_dgim.im = resize_image(new_dgim.im, (h, w), self.interp_type, self.interp_range)
        except IndexError:
            print('Resize fail. shape={}, w={}, h={}'.format(new_dgim.im.shape, w, h))
            print('Required GSD: {}'.format(self.sat.requiredGSD))
            print('Actual GSD: {}'.format(dgim.meta.get('gsd')))
            raise

        # Crop off context padding
        new_dgim.restoreContext()

        return new_dgim, bbox, img_path


class HOG(object):
    #def __init__(self, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False, multichannel=True):
    def __init__(self, orientations=5, pixels_per_cell=(24, 24), cells_per_block=(2, 2), visualize=False, multichannel=True):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.visualize = visualize
        self.multichannel = multichannel

    def __call__(self, img):
        feat = hog(img, orientations=self.orientations, pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block, visualize=self.visualize, multichannel=self.multichannel)

        feat_tsr = torch.FloatTensor(feat)

        return feat_tsr

class Normalize(object):
    def __init__(self, feat_mean, feat_std):
        self.feat_mean = feat_mean
        self.feat_std = feat_std

    def __call__(self, feat_tsr):
        return (feat_tsr - self.feat_mean) / self.feat_std

class MinMax(object):
    def __init__(self, per_channel=False):
        self.per_channel = per_channel

    def __call__(self, tsr):
        if self.per_channel:
            tsr = tsr - tsr.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
            tsr = tsr - tsr.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        else:
            tsr = tsr - tsr.min()
            tsr = tsr - tsr.max()
        return tsr

class NormalizeImage(object):
    def __init__(self, mean, std):
        self.mean = torch.FloatTensor(mean).view(3, 1, 1)
        self.std = torch.FloatTensor(std).view(3, 1, 1)

    def __call__(self, tsr):
        return (tsr - self.mean) / self.std


if __name__=='__main__':
    """
    import os
    from pipeline import VizPipeline
    import pilutils
    from params import parse_params

    template_path = './params.yaml'
    profile_param_path = './profileviz.yaml'
    profile_params = parse_params(template_path, profile_param_path)

    category = 'oil_or_gas_facility'
    inst_idx = 5

    focal_length = 0.5
    pipeline = VizPipeline(profile_params, 'train', 0, param_name='focal_length', param_val=focal_length)
    img = pipeline.get(category, inst_idx)
    """
    import json
    img_path = '/data/fmow/full/train/office_building/office_building_712/office_building_712_3_rgb.tif'
    json_path = '/data/fmow/full/train/office_building/office_building_712/office_building_712_3_rgb.json'
    with open(json_path, 'r') as fp:
        md_dict = json.load(fp)
    bbox = md_dict['bounding_boxes'][0]['box']
    img = Image.open(img_path)
    print(bbox)
    print(img.size)
    
    # Extract target from image
    extract = TargetExtract()
    img, bbox, path = extract((img, bbox, img_path)) 
    print(bbox)
    print(img.size)

    # Run target through imsim
    sensor_params = dict((x, y) for x, y in [('bit_depth', 13.0),
             ('election_well_depth', 40300.0),
             ('p_diameter', 0.09),
             ('QE', np.array([0.22, 0.22, 0.16])),
             ('opticTransmission', np.array([0.95, 0.95, 0.95])),
             ('Tint', 0.00025),
             ('wvl', np.array([4.5e-07, 5.5e-07, 6.5e-07])),
             ('read_noise', 12.5),
             ('delta_wvl', np.array([0.1, 0.1, 0.1])),
             ('focal_length', 4.0),
             ('alt', 500000.0),
             ('pixel_pitch', 6e-06),
             ('add_poissonNoise', 1.0),
             ('s_diameter', 0.009)])
    satellite = Satellite(sensor_params)
    img, bbox, path = satellite((img, bbox, path)) 
