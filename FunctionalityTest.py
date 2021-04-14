#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 17:39:00 2021

@author: alireza
"""

import segmentationtodicomrt as rt
import os

path='/Users/Alireza/Desktop/Datasets/covid4237/'
files = os.listdir(path+'dicom')
name=files[10]
print(name)

rt.Image2Mask(path,name)

rt.BorderPixels2NumpyArray(path,name,1)
rt.BorderPixels2NumpyArray(path,name,2)

rt.TextMaskImage2Numpy(path)

rt.DicomRT(path,name,1)
rt.DicomRT(path,name,2)