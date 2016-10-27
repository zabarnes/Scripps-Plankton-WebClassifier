# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 14:14:45 2014

Make the annular selector / averager available for import 

@author: Orenstein
"""

import numpy as np
from math import pi

class AnnulusProcessing:
    
    def __init__(self, data, num_rings, num_wedge, rad_in = 15):
        self.data = data
        self.num_rings = num_rings
        self.num_wedge = num_wedge
        self.rad_in = rad_in # inner radius for wedges. default = 15
        
    def make_annular_mean(self):
        """
        Generate concentric rings of equal area.
    
        Data == FT of ROI (should be padded to appropriate size when transformed)
        num_rings = number of annular regions needed
    
        out == the mean of pixel values in each ring
        """

        ny, nx = self.data.shape[:2]
    
        if ny != nx:
            print "Array must be square"
        elif ny % 2 != 0 or nx % 2 != 0:
            print "All array dimensions must be even"
        else:
            org_x, org_y = nx // 2, ny // 2
    
        # calculate radius    
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        x -= org_x
        y -= org_y
        r = np.sqrt(x**2 + y**2)
    
        rad_int = (nx - org_x) // self.num_rings
        out = np.zeros(self.num_rings)
        rad_in = 0
    
        for i in range(0, self.num_rings):
            rad_out = rad_in + rad_int
            
            ii, jj = np.logical_and(rad_in <= r, r < rad_out).nonzero()
            
            seg = self.data[ii, jj]
            
            out[i] = np.mean(np.abs(seg)) # abs cause data is FT 
            
            rad_in = rad_out
            
        out = out / np.sum(np.abs(self.data))
                
        return out
        
    def make_wedge(self):
        
        ny, nx = self.data.shape[:2]
        if ny != nx:
            print "Array must be square"
        elif ny % 2 != 0 or nx % 2 != 0:
            print "All array dimensions must be even"
        else:
            org_x, org_y = nx // 2, ny // 2
    
        # calculate radius    
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        x -= org_x
        y -= org_y
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        
        rad_out = nx - org_x
        
        ii, jj = np.logical_and(self.rad_in <= r, r < rad_out).nonzero()
        
        mask = np.zeros([ny, nx])
        mask[ii, jj] = 1
        
        ang = 0 
        inc = pi/self.num_wedge
        out = np.zeros(self.num_wedge)
        for i in range(0, self.num_wedge):
            p, q = np.logical_and(ang <= theta, theta < ang+inc).nonzero()
            wedge = np.zeros([ny, nx])
            wedge[p,q] = 1
            wedge = wedge*mask
            temp = self.data*wedge
            ind = np.nonzero(temp)
            ind = np.asarray(ind)
            out[i] = np.mean(np.abs(temp[ind[0],ind[1]]))
            ang += inc
        
        out = out / np.sum(np.abs(self.data))
        
        return out
        
        
        
        