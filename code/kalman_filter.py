# -*- coding: utf-8 -*-

import state_space_system
import numpy as np
import matplotlib.pyplot as plt
from ERA import *


def predict(x,u,P,A,Q,B):
    
    xpred = A@x + (B@u).reshape(A.shape[0],1)
    Ppred = A@P@A.T + Q
    return xpred,Ppred

def K_G(P,C,R):
    # kalman gain calculating
    
    K = P@C.T@np.linalg.inv(C@P@C.T+R)
    return K

def update(xpred,K,z,C,P,R,A):
    #update the x and P matrix
    
    z = z.reshape(C.shape[0],1)
    xnew = xpred + K@(z-(C@xpred).reshape(C.shape[0],1))
    temp = np.eye(A.shape[0]) - K@C
    Pnew = temp@P@temp.T + K@R@K.T
    
    return xnew,Pnew

if __name__ == '__main__':
    
    # identifield system
    q = 2   # Number of inputs
    p = 3   # Number of outputs
    r = 10  # Reduced model order
    
    time = np.linspace(0, 20, num=500)
    # import real system
    sys = state_space_system.sys()
    sys.reset()
    u = np.zeros((time.shape[0]-1, 2))
    u[0,0] = 1
    u1 = u
    
    u = np.zeros((time.shape[0]-1, 2))
    u[0,1] = 1
    u2 = u
    
    y_1 = sys.simulate(time, u1)
    # Concatenate:
    y_1 = np.array(y_1)
        
    y_2 = sys.simulate(time, u2)
    
    # Concatenate:
    y_2 = np.array(y_2)
        
    yFull = np.zeros((r*5+2,p,q))
    
    yFull[:,:,0] = y_1[0:r*5+2,:]
    yFull[:,:,1] = y_2[0:r*5+2,:]
    
    YY = np.transpose(yFull,axes=(1,2,0)) # reorder to size p x q x m 
    
    # Compute reduced order model from impulse response
    mco = int(np.floor((yFull.shape[0]-1)/2)) # m_c = m_o = (m-1)/2
    A,B,C,D,HSVs = ERA(YY,mco,mco,q,p,r)
    
    ##TEST KALMAN FILTER##
    
    u = np.ones((time.shape[0]-1, 2))
    
    covariance = 1
    
    P = np.eye(A.shape[0],A.shape[1])
    
    Q = np.ones((A.shape[0],A.shape[1]))*covariance
    
    R = np.eye(C.shape[0],C.shape[0])*covariance
    
    x = np.zeros((A.shape[0],1))
    
    sys.reset()
    y = [sys.make_measurement()]
    
    X = []
    
    for k in range(time.shape[0]-1):
    # Simulate for one Horizon:
        X.append(x)
        
        sys.simulate(time[k:k+2], u[[k]])
        
        y.append(sys.make_measurement())
        
        xpred, Ppred = predict(x, u[k], P, A, Q, B)
        
        K = K_G(P,C,R)
        
        x,P = update(xpred,K,y[k+1],C,P,R,A) 
        
        Y = np.array(y)
        
    X = np.array(X)
    y = np.array(y)
    
    # generate the kalman filter evaluated trajectory
    y_test = []
    for n in range(len(time)-1):
        y_test.append(C@X[n]+(D@u[n]).reshape(C.shape[0],1))
        
    y_test = np.array(y_test)
    y_test = y_test.reshape(499,3)
    
    x0 = np.zeros((A.shape[0],1))
    
    # trajectory from the ERA system under the same input signal
    y_ERA = []
    for n in range(len(time)-1):
        x_n = A@x0 + (B@u[n]).reshape(A.shape[0],1)
        y_ERA.append(C@x0+(D@u[n]).reshape(C.shape[0],1))
        x0 = x_n
        
    y_ERA = np.array(y_ERA)
    y_ERA = y_ERA.reshape(499,3)
    
    # visulization of the results
    
    fig, ax = plt.subplots(3,1)
    ax = ax.reshape(-1)
    
    ax[0].plot(time[:-1], y_test[:,0],'r',linewidth=1.5)
    ax[0].plot(time[:-1],y[:-1,0],'b:',linewidth=1.5)
    ax[0].plot(time[:-1],y_ERA[:,0],'g-.',linewidth=2)
    ax[0].set_ylabel('channel_1')
    ax[0].set_title('red:Kalman   blue_dot:original_sys   Green_dot_dash:ERA')
    
    ax[1].plot(time[:-1], y_test[:,1],'r',linewidth=1.5)
    ax[1].plot(time[:-1],y[:-1,1],'b:',linewidth=1.5)
    ax[1].plot(time[:-1],y_ERA[:,1],'g-.',linewidth=2)
    ax[1].set_ylabel('channel_2')
    
    ax[2].plot(time[:-1], y_test[:,2],'r',linewidth=1.5)
    ax[2].plot(time[:-1],y[:-1,2],'b:',linewidth=1.5)
    ax[2].plot(time[:-1],y_ERA[:,2],'g-.',linewidth=2)
    ax[2].set_ylabel('channel_3')
    ax[2].set_xlabel('time/s')
    
    plt.show()
    fig.savefig('kalman_filter_result',dpi=250)

    