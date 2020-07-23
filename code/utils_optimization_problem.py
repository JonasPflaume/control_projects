# -*- coding: utf-8 -*-

import numpy as np

def augmented_system(Ar,Br,Cr):
    # creating the augmented system for unbiased tracking task
    
    temp1a = np.zeros((Br.shape[1],Ar.shape[0]))
    temp2a = np.eye(Br.shape[1])
    temp3a = np.concatenate((Ar,Br),1)
    temp4a = np.concatenate((temp1a,temp2a),1)
    Ab = np.concatenate((temp3a,temp4a),0) # order 10 system here is 12*12
    
    temp1b = np.eye(Br.shape[1])
    Bb = np.concatenate((Br,temp1b),0) # new b matrix is 12*2
    
    temp1c = np.zeros((Cr.shape[0],Ab.shape[0]-Cr.shape[1]))
    Cb = np.concatenate((Cr,temp1c),1)
    
    return Ab,Bb,Cb
