# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:27:17 2020

@author: Li Jiayun
"""
import state_space_system

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from control.matlab import *
import control 
from scipy.linalg import fractional_matrix_power


plt.rcParams['figure.figsize'] = [10, 8]
plt.rcParams.update({'font.size': 18})


q = 2   # Number of inputs
p = 3   # Number of outputs
r = 10  # Reduced model order

def ERA(YY,m,n,nin,nout,r):
    Dr = np.zeros((nout,nin))
    Y = np.zeros((nout,nin,YY.shape[2]-1))
    for i in range(nout):
        for j in range(nin):
            Dr[i,j] = YY[i,j,0]
            Y[i,j,:] = YY[i,j,1:]
            
    assert len(Y[:,1,1]) == nout
    assert len(Y[1,:,1]) == nin
    assert len(Y[1,1,:]) >= m+n
    
    H = np.zeros((nout*m,nin*n))
    H2 = np.zeros((nout*m,nin*n))
    
    for i in range(m):
        for j in range(n):
            for Q in range(nout):
                for P in range(nin):
                    H[nout*i+Q,nin*j+P] = Y[Q,P,i+j]
                    H2[nout*i+Q,nin*j+P] = Y[Q,P,i+j+1] # hanlel matrix
                    
    U,S,VT = np.linalg.svd(H,full_matrices=0) #singular value
    V = VT.T
    Sigma = np.diag(S[:r])
    Ur = U[:,:r]
    Vr = V[:,:r]
    Ar = fractional_matrix_power(Sigma,-0.5) @ Ur.T @ H2 @ Vr @ fractional_matrix_power(Sigma,-0.5)
    Br = fractional_matrix_power(Sigma,-0.5) @ Ur.T @ H[:,:nin]
    Cr = H[:nout,:] @ Vr @ fractional_matrix_power(Sigma,-0.5)
    HSVs = S
    
    return Ar,Br,Cr,Dr,HSVs # state space 



if __name__ == '__main__':
    time = np.linspace(0, 20, num=500)
    sys = state_space_system.sys()# import original system
    sys.reset()
    
    u = np.zeros((time.shape[0]-1, 2))
    u[0,0] = 1#/(time[1]-time[0])
    u1 = u
    
    u = np.zeros((time.shape[0]-1, 2))
    u[0,1] = 1#/(time[1]-time[0])
    u2 = u
    
    y_1 = sys.simulate(time, u1)
    
    y_2 = sys.simulate(time, u2)
    
    yFull = np.zeros((time.shape[0],p,q)) #r*5+2
    
    
    #tspan = np.arange(0,(r*5+2),1)

    yFull[:,:,0] = y_1[:,:] #0:r*5+2
    yFull[:,:,1] = y_2[:,:]

    YY = np.transpose(yFull,axes=(1,2,0)) # reorder to size p x q x m

    ## Compute ERA from impulse response
    mco = int(np.floor((yFull.shape[0]-1)/2)) # m_c = m_o = (m-1)/2
    Ar,Br,Cr,Dr,HSVs = ERA(YY,mco,mco,q,p,r)
    sysERA = ss(Ar,Br,Cr,Dr,1)


    y = np.zeros((time.shape[0],p,q))

    for qi in range(q):
        y[:,:,qi],t = impulse(sysERA,np.arange(time.shape[0]),input=qi)
   

    #Plot the individual output from each input channel
    '''
    fig,axs = plt.subplots(3,2)
    axs = axs.reshape(-1)
    
    axs[0].step(t,y[:,0,0],linewidth=1.2)
    
    axs[0].set_ylabel('y1')
    axs[0].set_title('u1')
    
    axs[1].step(t,y[:,0,1],linewidth=1.2)
    
    axs[1].set_title('u2')
    
    axs[2].step(t,y[:,1,0],linewidth=1.2)
    
    axs[2].set_ylabel('y2')
    
    axs[3].step(t,y[:,1,1],linewidth=1.2,label='ERA, r={}'.format(r))
    
    axs[4].step(t,y[:,2,0],linewidth=1.2)
    axs[4].set_ylabel('y3')
    axs[5].step(t,y[:,2,1],linewidth=1.2)
    axs[3].legend(prop={'size': 12})
    
    for ax in axs:
        ax.set_xlim(0,500)
        
        plt.show()
        '''
        
        
    ###################################################  
    
    fig,axs = plt.subplots(2,2)
    axs = axs.reshape(-1)
    
    axs[0].plot(time,y[:,:,0],linewidth=2)
    axs[0].set_ylabel('Impul_res_from_ERA')
    axs[0].set_title('u1')
    axs[1].plot(time,y[:,:,1],linewidth=2)
    axs[1].set_title('u2')
    
    axs[2].plot(time,y_1,linewidth=2)
    axs[2].set_ylabel('Impul_res_from_Model')
    axs[3].plot(time,y_2,linewidth=2)
    plt.savefig('impulse_response',dpi=250)
    
    ##################################################
    fig,axs = plt.subplots(2,1)
    axs = axs.reshape(-1)
    # original
    sys.reset()
    u = np.ones((time.shape[0]-1, 2))
    
    y = sys.simulate(time, u)

    # ERA sys
    x0 = np.zeros((r,1))
    y_t = []
    
    y0 = np.array([0,0,0]).reshape(3,1)
    #u = (time[1]-time[0])*u
    
    for n in range(len(time)-1):
        
        y_t.append(y0)
        x_n = Ar@x0+(Br@u[n,:]).reshape(r,1)
        x0 = x_n
        y0 = Cr@x0+(Dr@u[n,:]).reshape(Cr.shape[0],1)
        
    y_t.append(y0)
        
    y_t = np.array(y_t).reshape(time.shape[0],3)
    y_t = y_t
        
    axs[0].plot(time,y,linewidth=2)
    axs[0].set_ylabel('Original_sys')
    axs[0].set_title('Step response')
    axs[1].plot(time,y_t,linewidth=2)
    axs[1].set_ylabel('ERA')
    plt.savefig('Step_response',dpi=250)
    