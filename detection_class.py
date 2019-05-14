#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 09:17:59 2019

@author: zhejun
"""

from termcolor import colored
import time
from collections import namedtuple
from skimage.measure import regionprops, label as sklabel
from skimage.morphology import disk as skdisk
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import helpy
import torch
from multiprocessing import Pool, cpu_count
import os
from orientation_model import ConvNet
from orientation_model import ClassNet

def unwrap_self_f(arg, **kwarg):
    return position_detection.detection(*arg, **kwarg)


def prep_image(im_file, width = 2):
    im = plt.imread(im_file).astype(float)
    s = width * im.std()
    m = im.mean()
    im -= m - s
    im /= 2*s
    np.clip(im, 0, 1, out = im)
    return im


def gdisk(width, inner = 0, outer = None):
    outer = outer or inner + 4 * width
    circ = skdisk(outer).astype(int)
    incirc = circ.nonzero()

    x = np.arange(-outer, outer+1, dtype=float)
    x, y = np.meshgrid(x, x)
    r = np.hypot(x, y) - inner
    np.clip(r, 0, None, r)

    g = np.exp(-0.5*(r/width)**2)
    g -= g[incirc].mean()
    g /= g[incirc].std()
    g *= circ   
    return g


def label_particles_convolve(im, kern, thresh=0.45, **extra_args):
    kernel = np.sign(kern)*gdisk(abs(kern)/4, abs(kern))
    convolved = ndimage.convolve(im, kernel)
    convolved -= convolved.min()
    convolved /= convolved.max()

    threshed = convolved > thresh

    labels = sklabel(threshed, connectivity=1)

    return labels, convolved, threshed

class position_detection:
    
    def __init__(self, files, output, plot = False, kern=-2.5, ecc=0.9, min_area=40, max_area=180, convex = 0.15, image_size = 32, \
                 thresh=0.46, threads=0, mean = 0.581848, std = 0.22367333, save_result = True, target = 0, batch = 100,\
                 or_weight_path ='/home/zhejun/Result/Phase_Separation/class_model.pth',\
                 class_weight_path='/home/zhejun/Result/Phase_Separation/particle_cat.pth'):
        self.plot = plot
        self.save_result = save_result
        self.batch = batch
        self.target = target
        self.image_size = image_size
        self.init_file(files)
        self.init_outdir(output)
        self.convex = convex
        self.meta = helpy.load_meta(self.prefix)
        self.kern_area = np.pi*kern**2
        self.size = {'center': {'max_ecc': ecc,
                            'min_area': min_area or int(self.kern_area//2),
                            'max_area': max_area or int(self.kern_area*2 + 1),
                            'kern': float(kern),
                            'thresh': thresh}}
        self.meta.update({k: v for k, v in self.size['center'].iteritems()})
        self.init_boundary()
        self.print_freq = len(self.filenames)//100 + 1
        self.init_cpu(threads)
        self.Segment = namedtuple('Segment', 'x y label ecc area o'.split())
        self.mean = mean
        self.std = std
        self.or_weight_path = or_weight_path
        self.class_weight_path = class_weight_path
        self.init_model()
    
    def init_file(self, files):
        self.total = len(files)
        if self.total > 1:
            self.filenames = sorted(files)            
            self.first = files[0]
            self.batch_number = [(start, start+self.batch) for start in range(0, self.total, self.batch)]
            """
            i = sys.argv.index(self.first)
            argv = filter(lambda s: s not in filenames, sys.argv)
            
            argv.insert(i, self.filepattern)
            argv[0] = os.path.basename(argv[0])
            self.argv = ' '.join(argv)
            """
        else:
            self.filenames = [files]
            self.first = self.filenames[0]
            self.batch = [(0, self.batch)]
            
        
        
    def init_outdir(self, output):
        self.suffix = '_POSITIONS'
        self.outdir = os.path.abspath(os.path.dirname(output))
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        self.prefix = output
        self.output = self.prefix + self.suffix
        self.imdir = self.prefix + '_detection'
        if not os.path.isdir(self.imdir):
            os.makedirs(self.imdir)
        return
    
    def init_cpu(self, threads):
        self.threads = cpu_count() or threads
        if False:
            print("Multiprocessing with {} threads".format(self.threads))
            p = Pool(self.threads)
            self.mapper = p.map
        else:
            self.mapper = map
        return
            
    def init_boundary(self):
        self.boundary = self.meta.get('boundary')
        if self.boundary is None or self.boundary == [0.0]*3:
            self.boundary = helpy.circle_click(self.first)
        self.meta.update(boundary = self.boundary)
        helpy.save_meta(self.prefix, self.meta)
        self.x0, self.y0, self.R0 = self.boundary
        return
    
    def init_model(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #self.device = 'cpu'
        # load category network
        self.net_cat = ClassNet()
        self.net_cat.load_state_dict(torch.load(self.class_weight_path))
        self.net_cat.eval()
        self.net_cat.to(self.device)
        # load orientation network
        self.net = ConvNet()        
        self.net.load_state_dict(torch.load(self.or_weight_path))
        self.net.eval()        
        self.net.to(self.device)
        
    def filter_segments(self, labels, max_ecc, min_area, max_area, circ=None, intensity=None, keep = False, **extra_args):
        """
        if keep bad segments,  return two results pts and mask whether pts is good or not.
        """
        pts = []
        pts_mask = list()
        centroid = 'Centroid' if intensity is None else 'WeightedCentroid'
        rpropargs = labels, intensity
        for rprop in regionprops(*rpropargs, coordinates='xy'):
            area = rprop['area']
            good = (min_area <= area <= max_area)
            if not (good or keep):
                continue
            ecc = rprop['eccentricity']
            good = (good &(ecc <= max_ecc))
            if not (good or keep):
                continue
            convex = rprop['convex_area']
            ratio = (convex - area)/float(area)
            good &= (ratio < self.convex)
            if not (good or keep):
                continue
            x, y = rprop[centroid]
            pts.append(self.Segment(x, y, rprop.label, ecc, area, 0))
            if keep:
                pts_mask.append(good)
        if keep:
            return pts, np.asarray(pts_mask)
        return pts

    def find_particles(self, im, **kwargs):
        labels, convolved, threshed = label_particles_convolve(im, **kwargs)
        intensity = im if kwargs['kern'] > 0 else 1 - im
        keep = self.plot
        pts = self.filter_segments(labels, intensity= intensity, keep = keep, **kwargs)
        return pts, labels, convolved       

    def extract_image(self, xys, image, x0=0, y0 =0, R0 =0, image_size = 32):
        center = image_size //2
        pos = xys - [center, center]
        pos = pos.astype(int)
        result = list()
        mask = (np.hypot(*(xys - [self.x0, self.y0]).T) < self.R0 - 5)
        for x, y in pos[mask]:
            result.append(image[x:x+image_size,  y: y+image_size])
        return (np.array(result) - self.mean)/self.std, mask 
    
    def net_orientation(self, input_images):
        total_image = len(input_images)
        input_images = np.expand_dims(input_images, axis= 1)
        batches = [(start, start + 100) for start in range(0, total_image, 100)]
        cats, orients = list(), list()
        for start, end in batches:
            images = input_images[start:end]
            inputs = torch.Tensor(images)
            inputs = inputs.to(self.device)
            mask = self.net_cat(inputs)
            cat = mask.argmax(dim=1)
            cat = cat.tolist()
            cats += cat
            mask = torch.ByteTensor(cat)            

            orient = self.net(inputs[mask])        
            orient = orient.argmax(dim=1)
            orient = orient.tolist()
            orients += orient            
        return np.array(cats, dtype=bool), np.array(orients)
    
    def get_positions(self, tuple_input):
        filename = tuple_input[1]
        self.snapshot_num = 0 
        filebase = os.path.splitext(os.path.basename(filename))[0]
        imbase = os.path.join(self.imdir, filebase)
        self.imprefix = imbase
        #prep image
        image = prep_image(filename)
        out = self.find_particles(image, **self.size['center'])
        segments = out[0]
        if self.plot:
            #self.plot_positions(*out, **self.size['center'])
            segments = np.array(segments[0], dtype=np.float64)[segments[1]]
        nfound = len(segments)
        if nfound:
            centers = np.hstack([np.full((nfound, 1), self.id_number, 'f8'), segments])
        else:  # empty line of length 6 = id + len(Segment)
            centers = np.empty((0, 7))
        xys = centers[:, 1:3]
        self.id_number += 1
        start_time = time.time()
        input_images, mask = self.extract_image(xys, image)
        print(time.time() - start_time)
        return input_images, centers[mask] 
    
    def plot_check(self, image, centers, savename):
        fig, ax = plt.subplots(figsize = (8,8))
        ax.imshow(image, cmap='gray')
        for row in centers:
            ax.arrow(row[2], row[1], 20*np.sin(row[-1]), 20*np.cos(row[-1]), color = 'red')
        fig.savefig(savename, dpi = 300)
        return

    
    def detection(self):
        self.points = np.empty((0,7))
        self.id_number = 0
        for start, end in self.batch_number:
            print(start)
            points = np.empty((0, 7))
            inputs = np.empty((0, self.image_size, self.image_size))
            batch_file = self.filenames[start: end]
            ret = self.mapper(self.get_positions, enumerate(batch_file))
            for input_image, center in ret:
                inputs = np.concatenate((inputs, input_image))
                points = np.vstack((points, center))
            print(inputs.shape)
            print(points.shape)
            cat, orient = self.net_orientation(inputs)
            points = points[cat]
            points[:,-1] = np.deg2rad(orient)
            self.points = np.vstack((self.points, points))
            
        if self.plot:
            self.count = 0
            plot_number = min(self.total, 8)
            plot_index = np.random.choice(self.total, plot_number, replace=False)
            plot_image = [self.filenames[idx] for idx in plot_index]
            for (file_name, plot_index) in zip(plot_image, plot_index):
                image = prep_image(file_name)
                mask = self.points[:,0] == plot_index
                centers = self.points[mask]
                savename = file_name.split('.')[0]
                savename += '.pdf'
                savename = os.path.join(self.imdir, savename)
                self.plot_check(image, centers, savename)            
        
        
        if self.save_result:
            savenotice = "Saving {} positions to {}{{{},.npz}}".format
            hfmt = ('Kern {kern:.2f}, Min area {min_area:d}, '
                    'Max area {max_area:d}, Max eccen {max_ecc:.2f}\n'
                    'Frame    X           Y             Label  Eccen        Area      Orientation')
            txtfmt = ['%6d', '%7.3f', '%7.3f', '%4d', '%1.3f', '%5d', '%1.3f']
            ext = '.txt'+'.gz'
            print(savenotice('center', self.output, ext))
            np.savetxt(self.output+ext, self.points, header=hfmt.format(**self.size['center']), delimiter='     ', fmt=txtfmt)
            helpy.txt_to_npz(self.output+ext, verbose=False, compress=True)