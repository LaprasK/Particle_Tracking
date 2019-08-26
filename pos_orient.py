#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 12:57:19 2019

@author: zhejun
"""

from argparse import ArgumentParser
from collections import namedtuple

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    arg = parser.add_argument
    arg('files', metavar='FILE', nargs='+', help='Images to process')
    arg('-o', '--output', help='Output filename prefix.')
    arg('-p', '--plot', default=False, help="Plot detection results max 10 frames")
    arg('-v', '--verbose', action='count', help="Control verbosity")
    arg('-N', '--threads', default = 0, type=int, help='Number of worker threads for '
        'parallel processing. N=0 uses all available cores')
    arg('-b', '--batch', default = 100, type = int, help = "batch size for processing")
    arg('--thresh', type=float, default=0.45, help='Binary threshold '
        'for defining segments, in units of standard deviation')
    arg('-k', '--kern', type=float, default=-2.5, 
        help='Kernel size for convolution')
    arg('--min', default = 40, type=int, help='Minimum area')
    arg('--max', default = 180, type=int, help='Maximum area')
    arg('--ecc', default=.8, type=float, help='Maximum eccentricity')
    arg('--convex', default=.2, type=float, help='Convex area ratio')
    arg('--ncen', type = int, required = True, help = 'Experimental particle number')
    arg('--image', default = 32, type = float, help = 'image size to crop')
    arg('--mean', default = 0.56655243, type = float, help = 'mean value to initilize NN images')
    arg('--std', default = 0.22493147, type = float, help = 'std to initialize NN images')
    arg('--save', default = True, type = bool, help = 'whether save result')
    arg('--opath', default = '/home/zhejun/Result/Phase_Separation/big_model.pth',\
        type = str, help ='path to load orientation network path')
    arg('--cpath', default = '/home/zhejun/Result/Phase_Separation/particle_cat.pth',\
        type = str, help = 'path to load category network path')
    args = parser.parse_args()
    


    
def prep_image(im_file, width = 2):
    """
    Given image filename, normalize it to 0 to 1
    """
    im = plt.imread(im_file).astype(float)
    s = width * im.std()
    m = im.mean()
    im -= m - s
    im /= 2*s
    np.clip(im, 0, 1, out = im)
    return im


def gdisk(width, inner = 0, outer = None):
    """
    Given the width, build a Gaussian Kernel for image convolution
    """
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
    """
    Given image and kenerl, use the kernel to convolve with the image, 
    if value larger than threshold, make it 1, otherwise 0.
    use 1 and 0 to build the center segments
    """
    kernel = np.sign(kern)*gdisk(abs(kern)/4, abs(kern))
    convolved = ndimage.convolve(im, kernel)
    convolved -= convolved.min()
    convolved /= convolved.max()

    threshed = convolved > thresh

    labels = sklabel(threshed, connectivity=1)

    return labels, convolved, threshed


Segment = namedtuple('Segment', 'x y label ecc area o'.split())

    
def filter_segments(labels, max_ecc, min_area, max_area, convex=0.2, circ=None, intensity=None, keep = False, **extra_args):
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
        convex_area = rprop['convex_area']
        ratio = (convex_area - area)/float(area)
        good &= (ratio < convex)
        if not (good or keep):
            continue
        x, y = rprop[centroid]
        pts.append(Segment(x, y, rprop.label, ecc, area, 0))
        if keep:
            pts_mask.append(good)
    if keep:
        return pts, np.asarray(pts_mask)
    return pts


def find_particles(im, keep = False, **kwargs):
    labels, convolved, threshed = label_particles_convolve(im, **kwargs)
    intensity = im if kwargs['kern'] > 0 else 1 - im
    pts = filter_segments(labels, intensity= intensity, keep = keep, **kwargs)
    return pts, labels, convolved


def net_orientation(input_images, net_cat = None, net = None, device = 'cuda:0'):
    total_image = len(input_images)
    input_images = np.expand_dims(input_images, axis= 1)
    if total_image % 100 > 3:
        my_batch = 100
    elif total_image % 101 > 3:
        my_batch = 101
    else:
        my_batch = 102
    #print(total_image % my_batch)
    batches = [(start, start + my_batch) for start in range(0, total_image, my_batch)]
    cats, orients = list(), list()
    for start, end in batches:
        images = input_images[start:end]
        inputs = torch.Tensor(images)
        inputs = inputs.to(device)
        mask = net_cat(inputs)
        cat = mask.argmax(dim=1)
        cat = cat.tolist()
        cats += cat
        mask = torch.ByteTensor(cat)            

        orient = net(inputs[mask])        
        orient = orient.argmax(dim=1)
        orient = orient.tolist()
        orients += orient            
    return np.array(cats, dtype=bool), np.array(orients)

def plot_points(pts, img, name='', s=10, c='r', cmap=None,
                vmin=None, vmax=None, cbar=False, save = False):
    #global snapshot_num, imprefix
    fig, ax = plt.subplots(figsize=(8, 8))
    PPI = 84.638  # if figsize (8, 8)
    dpi = 4*PPI
    axim = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax,
                     interpolation='nearest')
    if cbar:
        fig.tight_layout()
        cb_height = 4
        cax = fig.add_axes(np.array([10, 99-cb_height, 80, cb_height])/100.0)
        fig.colorbar(axim, cax=cax, orientation='horizontal')
    xl, yl = ax.get_xlim(), ax.get_ylim()
    s = abs(s)
    xys = pts[:, 1:3][:,::-1]
    helpy.draw_circles(xys, s, ax, 
                       lw=max(s/10, .5), color=c, fill=False, zorder=2)
    if s > 3:
        ax.scatter(pts[2], pts[1], s, c, '+')
    ax.set_xlim(xl)
    ax.set_ylim(yl)
    ax.set_xticks([])
    ax.set_yticks([])
    
    savename = '{}_{}.png'.format(imprefix, name)
    fig.savefig(savename, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    return

def plot_positions(segments, labels, convolved=None, centers = None, **kwargs):
    Segment_dtype = np.dtype({'names': 'x y label ecc area o'.split(),
                              'formats': [float, float, int, float, float, float]})
    pts = np.asarray(segments[0], dtype=Segment_dtype)
    pts_by_label = np.zeros(labels.max()+1, dtype=Segment_dtype)
    pts_by_label[0] = (np.nan, np.nan, 0, np.nan, np.nan, np.nan)
    pts_by_label[pts['label']] = pts

    plot_points(centers, convolved, name='CONVOLVED',
                s=kwargs['kern'], c='r', cmap='viridis')

    labels_mask = np.where(labels, labels, np.nan)
    plot_points(centers, labels_mask, name='SEGMENTS',
                s=kwargs['kern'], c='k')#, cmap='prism_r')

    ecc_map = labels_mask*0
    ecc_map.flat = pts_by_label[labels.flat]['ecc']
    plot_points(centers, ecc_map, name='ECCEN',
                s=kwargs['kern'], c='k', cmap='Paired',
                vmin=0, vmax=1, cbar=True)

    area_map = labels_mask*0
    area_map.flat = pts_by_label[labels.flat]['area']
    plot_points(centers, area_map, name='AREA',
                s=kwargs['kern'], c='k', cmap='Paired',
                vmin=0, vmax=1.2*kwargs['max_area'], cbar=True)
    return

def plot_check(image, centers, savename):
    fig, ax = plt.subplots(figsize = (8,8))
    ax.imshow(image, cmap='gray')
    for row in centers:
        ax.arrow(row[2], row[1], 20*np.sin(row[-1]), 20*np.cos(row[-1]), color = 'red')
    fig.savefig(savename, dpi = 300)
    return

from termcolor import colored
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

if __name__ == '__main__':
    #build batch for files
    total = len(args.files)
    if total > 1:
        filenames = sorted(args.files)            
        first = filenames[0]
        batch_number = [(start, start+args.batch) for start in range(0, total, args.batch)]
    else:
        filenames = args.files
        first = filenames[0]
        batch_number = [(0, args.batch)]
    id_number = 0
    # initialize output directory
    suffix = '_POSITIONS'
    outdir = os.path.abspath(os.path.dirname(args.output))
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    prefix = args.output
    output = prefix + suffix
    imdir = prefix + '_detection'
    if not os.path.isdir(imdir):
        os.makedirs(imdir)
    #load meta and save basic information to meta
    meta = helpy.load_meta(prefix)
    kern_area = np.pi*args.kern**2
    size = {'center': {'max_ecc': args.ecc,
                       'min_area': args.min or int(kern_area//2),
                       'max_area': args.max or int(kern_area*2 + 1),
                       'kern': float(args.kern),
                       'thresh': args.thresh,
                       'convex': args.convex}}
    meta.update({k: v for k, v in size['center'].iteritems()})   
    #initialize boundary
    boundary = meta.get('boundary')
    if boundary is None or boundary == [0.0]*3:
        boundary = helpy.circle_click(first)
    meta.update(boundary = boundary)
    helpy.save_meta(prefix, meta)
    x0, y0, R0 = boundary
    #initialize model for machine leanring
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # load category network
    net_cat = ClassNet()
    net_cat.load_state_dict(torch.load(args.cpath))
    net_cat.eval()
    net_cat.to(device)
    # load orientation network
    net = ConvNet()        
    net.load_state_dict(torch.load(args.opath))
    net.eval()        
    net.to(device)
    
    def extract_image(xys, image, image_size = 32):
        """
        Given center positions of particles, extract 32*32 image for all particle
        xys: position of center of particle
        image: original image
        """
        center = image_size //2
        pos = xys - [center, center]
        pos = np.round(pos).astype(int)
        result = list()
        #require particle start point larger than 0 and smaller 1024 - 32 
        #m1 = (pos[:,0] > 0) & (pos[:,1] > 0) & (pos[:,0] < 1024 - image_size) & (pos[:,1] < 1024 - image_size)
        m2 = (np.hypot(*(xys - [x0, y0]).T) < R0)
        mask = m2 #m1 & m2
        for x, y in pos[mask]:
            if x + image_size >= 1024:
                diff = x + image_size - 1024
                pads = np.pad(image[x:1024, y:y+image_size], ((0,diff),(0,0)), 'edge')
            elif y + image_size >= 1024:
                diff = y + image_size - 1024
                pads = np.pad(image[x:x+image_size, y:1024], ((0,0),(0,diff)), 'edge')
            elif x < 0:
                diff = -x
                pads = np.pad(image[0:x+image_size, y:y+image_size],((diff,0),(0,0)),'edge')
            elif y < 0: 
                diff = -y
                pads = np.pad(image[x:x+image_size,0:y+image_size],((0,0),(diff,0)),'edge')
            else:
                pads = image[x: x+image_size, y:y+image_size]
                diff = 0
            result.append(pads)
            #print(diff)
            #mask.append(True)
        return (np.array(result) - args.mean)/args.std, np.array(mask) 
    
    
    def get_positions(tuple_input):
        n = tuple_input[0]
        filename = tuple_input[1]
        #prep image
        image = prep_image(filename)
        out = find_particles(image, **size['center'])
        segments = out[0]
        nfound = len(segments)
        if nfound:
            centers = np.hstack([np.full((nfound, 1), n, 'f8'), segments])
        else:  # empty line of length 6 = id + len(Segment)
            centers = np.empty((0, 7))
        xys = centers[:, 1:3]
        #x_mask = (xys[:,0] >= args.image//2) & (xys[:,0] <= 1024 - args.image//2)
        #xys = xys[x_mask]
        #y_mask = (xys[:,1] >= args.image//2) & (xys[:,1] <= 1024 - args.image//2)
        #xys = xys[y_mask]
        input_images, mask = extract_image(xys, image)
        
        return input_images, centers[mask]

    #initialize CPU for multiprocessing
    threads = cpu_count() or args.threads
    if threads > 1:
        print("Multiprocessing with {} threads".format(threads))
        p = Pool(threads)
        mapper = p.map
    else:
        mapper = map
        
    ###########################################################################
    # Start getting positions
    ###########################################################################
    ret_points = np.empty((0,7))
    for start, end in batch_number:
        print("Start Detecting Particles From {} ...".format(args.files[start]))
        points = np.empty((0, 7))
        inputs = np.empty((0, args.image, args.image))
        batch_file = filenames[start: end]
        ret = mapper(get_positions,enumerate(batch_file,start))
        for input_image, center in ret:
            #if input_image.shape[0] != args.image or input_image.shape[1] != args.image:
                #continue
            inputs = np.concatenate((inputs, input_image))
            points = np.vstack((points, center))
        cat, orient = net_orientation(inputs, net_cat = net_cat, net= net, device= device)
        points = points[cat]    
        points[:,-1] = np.deg2rad(orient)
        ret_points = np.vstack((ret_points, points))
        if total <= 100:
            for image_index in range(start, min(total, end)):
                finded = np.sum(points[:,0] == image_index)
                print("Find {} particles in {}".format(colored(finded, 'blue' if finded >= args.ncen else 'red'),\
                      args.files[image_index]))
        else:   
            finded = np.sum(points[:,0] == start)
            print("Find {} particles in {}".format(colored(finded, 'blue' if finded >= args.ncen else 'red'),\
                  args.files[start]))
        
    if args.plot:
        count = 0
        plot_number = min(total, 8)
        plot_index = np.random.choice(total, plot_number, replace=False)
        plot_image = [filenames[idx] for idx in plot_index]
        for (file_name, plot_index) in zip(plot_image, plot_index):
            image = prep_image(file_name)            
            mask = ret_points[:,0] == plot_index
            centers = ret_points[mask]
            pts, labels, convolved = find_particles(image, keep = True, **size['center'])
            filebase = os.path.splitext(os.path.basename(file_name))[0]
            imbase = os.path.join(imdir, filebase)
            imprefix = imbase
            plot_positions(pts, labels, convolved, centers,**size['center'])
            savename = file_name.split('.')[0]
            savename += '.png'
            savename = os.path.join(imdir, savename)
            plot_check(image, centers, savename)            
       
    
    if args.save:
        from shutil import copy
        copy(first, prefix+'_'+os.path.basename(first))
        savenotice = "Saving {} positions to {}{{{},.npz}}".format
        hfmt = ('Kern {kern:.2f}, Min area {min_area:d}, '
                'Max area {max_area:d}, Max eccen {max_ecc:.2f}\n'
                'Frame    X           Y             Label  Eccen        Area      Orientation')
        txtfmt = ['%6d', '%7.3f', '%7.3f', '%4d', '%1.3f', '%5d', '%1.3f']
        ext = '.txt'+'.gz'
        print(savenotice('center', output, ext))
        np.savetxt(output+ext, ret_points, header=hfmt.format(**size['center']), delimiter='     ', fmt=txtfmt)
        helpy.txt_to_npz(output+ext, verbose=False, compress=True)
    
