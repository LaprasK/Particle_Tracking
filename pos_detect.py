#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 16:13:23 2019

@author: zhejun
"""

from argparse import ArgumentParser
from detection_class import position_detection


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    arg = parser.add_argument
    arg('files', metavar='FILE', nargs='+', help='Images to process')
    arg('-o', '--output', help='Output filename prefix.')
    arg('-p', '--plot', default=False, help="Plot detection results max 10 frames")
    arg('-v', '--verbose', action='count', help="Control verbosity")
    arg('-N', '--threads', default = 0, type=int, help='Number of worker threads for '
        'parallel processing. N=0 uses all available cores')
    arg('--thresh', type=float, default=0.45, help='Binary threshold '
        'for defining segments, in units of standard deviation')
    arg('-k', '--kern', type=float, default=-2.5, 
        help='Kernel size for convolution')
    arg('--min', default = 40, type=int, help='Minimum area')
    arg('--max', default = 180, type=int, help='Maximum area')
    arg('--ecc', default=.8, type=float, help='Maximum eccentricity')
    arg('--ncen', type = int, required = True, help = 'Experimental particle number')
    arg('--mean', default = 0.581848, type = float, help = 'mean value to initilize NN images')
    arg('--std', default = 0.22367333, type = float, help = 'std to initialize NN images')
    arg('--opath', default = '/home/zhejun/Result/Phase_Separation/class_model.pth',\
        type = str, help ='path to load orientation network path')
    arg('--cpath', default = '/home/zhejun/Result/Phase_Separation/particle_cat.pth',\
        type = str, help = 'path to load category network path')
    args = parser.parse_args()
    
    detection = position_detection(files = args.files, output = args.output, \
                                   plot = args.plot, kern = args.kern, ecc = args.ecc,\
                                   min_area = args.min, max_area = args.max, thresh = args.thresh,\
                                   threads = args.threads, mean = args.mean, std=args.std,\
                                   target=args.ncen, or_weight_path=args.opath, \
                                   class_weight_path=args.cpath)
    detection.detection()