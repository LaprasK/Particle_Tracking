# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import helpy
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.transform import resize




class build_training_data:
    def __init__(self, image_file, frame_data, training_dict = None, number_test = None,\
                 training_shape = 32, upscale_size = 4, start_particle = 0, \
                 image_size = 1000, save_name = "2inch_square_training_data.npy"):
        self.image_file = image_file
        self.frame_data = frame_data
        self.training_shape = training_shape
        self.centralize = training_shape // 2
        self.upscale_size = upscale_size
        if number_test:
            self.number_test = number_test
        else:
            self.number_test = len(frame_data)
        self.start_particle = start_particle
        self.image_size = image_size
        if training_dict is None:
            self.training_dict = {
                "training_class": list(),
                "training_regress": list(),
                "training_x": list()
            }
        elif isinstance(training_dict, str):
            self.training_dict = np.load(training_dict).item()
            self.save_name = training_dict
        else:
            self.training_dict = training_dict
            
    def prep_image(self, width = 2):
        im = plt.imread(self.image_file).astype(float)
        s = width*im.std()
        m = im.mean()
        im -= m - s
        im /= 2 * s
        np.clip(im,0,1,out=im)
        return im
    
    def necessary_data(self):
        pos = self.frame_data['xy'] - np.array([self.centralize, self.centralize])
        pos_start = pos.astype(int)
        pos_cen = (self.frame_data['xy'] - pos_start)*self.upscale_size
        pos_arrow = pos_cen + self.upscale_size * 15.0 * \
                    np.array([np.cos(self.frame_data['o']), np.sin(self.frame_data['o'])]).T
        pos_arrow = pos_arrow.astype(np.float32)
        pos_cen = pos_cen.astype(np.float32)
        return pos_start, pos_cen, pos_arrow
    
    def interactive_mode(self):
        cv2.namedWindow("Build", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Build", self.image_size, self.image_size)
        pi2 = np.pi * 2
        img = self.prep_image()
        pos_start, pos_cen, pos_arrow = self.necessary_data()
        for frame, (x_start,y_start),arr_cen, arr_pos in zip(self.frame_data['o'][self.start_particle:self.number_test],\
                                                             pos_start[self.start_particle:self.number_test],\
                                                             pos_cen[self.start_particle:self.number_test],\
                                                             pos_arrow[self.start_particle:self.number_test]):
            temp_img = img[x_start:x_start+self.training_shape, y_start:y_start + self.training_shape]
            resize_img = resize(temp_img, (self.training_shape*self.upscale_size, self.training_shape*self.upscale_size),\
                               mode = 'reflect', anti_aliasing = True)
            if np.isnan(arr_pos[0]):
                cv2.imshow("Build", resize_img)
            else:
                arrow_plot = cv2.arrowedLine(resize_img, (arr_cen[1],arr_cen[0]),(arr_pos[1], arr_pos[0]),(255,0,0),2)
                cv2.imshow('Build', resize_img)
            print("detect or is: {}".format(frame))
            press = cv2.waitKey(0)
            # everything is good, store current information into training sets
            if press == 49:
                self.training_dict["training_x"].append(temp_img)
                self.training_dict["training_regress"].append(frame)
                self.training_dict["training_class"].append(1)
            # enter plot mode
            elif press == 50: 
                points_store = []
                button_down = False
                cv2.imshow("Build", resize_img)
                def orientation_click(event, x, y, flags, param):
                    global button_down
                    if event == cv2.EVENT_LBUTTONUP and button_down:
                        button_down = False
                        points_store.append((x,y))
                        cv2.arrowedLine(resize_img, points_store[0], points_store[1], (255,0,0),1)
                        cv2.imshow("Build", resize_img)
                        
                    elif event == cv2.EVENT_MOUSEMOVE and button_down:
                        button_down = True
                        image = resize_img.copy()
                        cv2.arrowedLine(image, points_store[0], (x,y), (255,0,0),1)
                        cv2.imshow("Build", image)

                    elif event == cv2.EVENT_LBUTTONDOWN and len(points_store) < 2:
                        button_down = True
                        points_store.insert(0, (x,y))
                cv2.setMouseCallback('Build',orientation_click, points_store)
                enter = cv2.waitKey(0)
                orientation = np.arctan2(points_store[1][0]-points_store[0][0], points_store[1][1]-points_store[0][1])
                orientation = (orientation + pi2)%pi2
                print("Click orientation is: {}".format(orientation))
                if enter == 49:
                    self.training_dict["training_x"].append(temp_img)
                    self.training_dict["training_regress"].append(orientation)
                    self.training_dict["training_class"].append(1)
            # wrong detected particles goes to mode 3, no orientation detected give it class 1
            elif press == 51:
                self.training_dict["training_x"].append(temp_img)
                self.training_dict["training_regress"].append(np.nan)
                self.training_dict["training_class"].append(0)
            elif press == 27:
                break
        cv2.destroyAllWindows()
        for i in range (1,5):
            cv2.waitKey(1)
        return
    
    
    def get_training_dict(self):
        self.interactive_mode()
        np.save(self.save_name, self.training_dict)