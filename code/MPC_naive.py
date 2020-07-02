# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 14:37:22 2020

@author: Li Jiayun
"""

import state_space_system
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
import numpy as np
import slycot
from scipy import integrate
from scipy.linalg import schur
from ERA import *
from casadi import *
from kalman_filter import *

mpl.rcParams['font.size'] = 16 # set the matplotlib parameter

q = 2   # Number of inputs
p = 3   # Number of outputs
r = 10  # Reduced model order

time = np.linspace(0, 20, num=500)
sys = state_space_system.sys()# import real system
sys.reset()
u = np.zeros((time.shape[0], 2))
u[0,0] = 1
u1 = u

u = np.zeros((time.shape[0], 2))
u[0,1] = 1
u2 = u

y_1 = [sys.make_measurement()]
for k in range(time.shape[0]-1):
    # Simulate for one timestep:
    sys.simulate(time[k:k+2], u1[[k]])
    # Append new measurement
    y_1.append(sys.make_measurement())

# Concatenate:
y_1 = np.array(y_1)
    
y_2 = [sys.make_measurement()]
for k in range(time.shape[0]-1):
    # Simulate for one timestep:
    sys.simulate(time[k:k+2], u2[[k]])
    # Append new measurement
    y_2.append(sys.make_measurement())

# Concatenate:
y_2 = np.array(y_2)
    
yFull = np.zeros((r*5+2,p,q))

yFull[:,:,0] = y_1[0:r*5+2,:]
yFull[:,:,1] = y_2[0:r*5+2,:]

YY = np.transpose(yFull,axes=(1,2,0)) # reorder to size p x q x m 

# Compute reduced order model from impulse response
mco = int(np.floor((yFull.shape[0]-1)/2)) # m_c = m_o = (m-1)/2
Ar,Br,Cr,Dr,HSVs = ERA(YY,mco,mco,q,p,r)

########################################################Identification finished

nx = Ar.shape[1] #set the number of state and the control inputs
nu = Br.shape[1]
ny = Cr.shape[0]

lam, v = np.linalg.eig(Ar)
# step 1 (investigate the stability of the identified system)

fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlabel('Real component')
ax.set_ylabel('Imag. component')
ax.add_artist(plt.Circle((0, 0), 1,edgecolor='k', fill=False))
ax.plot(np.real(lam),np.imag(lam),'o', markersize=10)
ax.axhline(0,color='k')
ax.axvline(0,color='k')
ax.set_ylim(-1,1)
ax.set_xlim(-1,1)

#  the system is stable but not asymptotically stable.

# step 2 (create a symbolic expression of state space)
x = SX.sym("x",nx,1)
u = SX.sym("u",nu,1)

x_next = Ar@x + Br@u

system = Function("sys",[x,u],[x_next]) # Create the CasADi function

# step 3 (simulate a linear discrete-time system)

N_sim = 50 # Define the total steps to simulate

x_0 = np.random.randint(3,size=(nx,1)) # Define the initial condition

u_k = np.array([[0]*nu]) # Define the input (for the moment consider u = 0)

res_x = [x_0]

for i in range(N_sim):
    x_next = system(x_0,u_k)
    res_x.append(x_next)
    x_0 = x_next

# Make an array from the list of arrays:
res_x = np.concatenate(res_x,axis=1)

fig, ax = plt.subplots(figsize=(10,6))

# plot the states
lines = ax.plot(res_x.T)

# Set labels
ax.set_ylabel('states')
ax.set_xlabel('time')
# we could see if we seted the N_sim = 500 the system will calm down itself, but the process will be very slow.

# step 4 (MPC initialization and casadi objective function) 
# parameters here need to be tuned

Q = 20                    # make it larger to get a more aggresive controller
Q = Q*np.diag(np.ones(ny)) # the state cost mutiplyer by using the output channel: ny=3

R = 10                     # set it to 1, trying to reduce the influence from the control energy term
R = np.diag(R*np.ones(nu)) # the cost of control

N = 20                     # horizon

y_Ref = SX.sym("y_ref",ny,1)   # desired trajectory

# state cost
stage_cost = (y_Ref-Cr@x -Dr@u).T@Q@(y_Ref-Cr@x-Dr@u)+ u.T@R@u
stage_cost_fcn = Function('stage_cost',[x,u,y_Ref],[stage_cost])

# terminal cost: we set the S matrix the same as Q
terminal_cost = (y_Ref-Cr@x).T@Q@(y_Ref-Cr@x)
terminal_cost_fcn = Function('terminal_cost',[x,y_Ref],[terminal_cost])

# state constraints  # need to be modified
lb_x = -3*np.ones((nx,1))
ub_x = 3*np.ones((nx,1))
# input constraints
lb_u = -10*np.ones((nu,1))
ub_u = 10*np.ones((nu,1))


# step 5 create optimization problem
X = SX.sym("X",(N+1)*nx,1)
U = SX.sym("U",N*nu,1)

J = 0
lb_X = [] # lower bound for X.
ub_X = [] # upper bound for X
lb_U = [] # lower bound for U
ub_U = [] # upper bound for U
g = []    # constraint expression g
lb_g = []  # lower bound for constraint expression g
ub_g = []  # upper bound for constraint expression g

#################################
y_ref_set = 1*np.ones((N,ny))  ## the reference trajectory N horizon (step response)
################################# If we want to test another trajectory we must integrate this into the MPC main loop

for k in range(N):
    # 01 
    x_k = X[k*nx:(k+1)*nx,:]
    x_k_next = X[(k+1)*nx:(k+2)*nx,:]
    u_k = U[k*nu:(k+1)*nu,:]
    # 01
    
    # 02 
    # objective
    J += stage_cost_fcn(x_k,u_k,y_ref_set[k])
    # 02
    
    # 03 
    # equality constraints (system equation)
    x_k_next_calc = system(x_k,u_k)
    # 03

    # 04
    g.append(x_k_next - x_k_next_calc)
    lb_g.append(np.zeros((nx,1)))
    ub_g.append(np.zeros((nx,1)))
    # 04

    # 05
    lb_X.append(lb_x)
    ub_X.append(ub_x)
    lb_U.append(lb_u)
    ub_U.append(ub_u)
    # 05


x_terminal = X[N*nx:(N+1)*nx,:]
J += terminal_cost_fcn(x_terminal,y_ref_set[-1])
lb_X.append(lb_x)
ub_X.append(ub_x)

# Step 6 Create Casadi solver
lbx = vertcat(*lb_X, *lb_U)
ubx = vertcat(*ub_X, *ub_U)
x = vertcat(X,U)
g = vertcat(*g)
lbg = vertcat(*lb_g)
ubg = vertcat(*ub_g)

prob = {'f':J,'x':x,'g':g}
solver = nlpsol('solver','ipopt',prob)

# Step 7 run the solver
x_0 = np.ones((nx,1))

lbx[:nx]=x_0
ubx[:nx]=x_0

res = solver(lbx=lbx,ubx=ubx,lbg=lbg,ubg=ubg)

X = res['x'][:(N+1)*nx].full().reshape(N+1, nx)
U = res['x'][(N+1)*nx:].full().reshape(N, nu)
Y = np.zeros((ny,1))

for n in range(N):
    temp = X[n+1].reshape(nx,1)
    Y = np.hstack((Y,Cr@temp))
    

# visulization of the result
fig, ax = plt.subplots(2,1, figsize=(10,8), sharex=True)
ax[0].plot(Y.T)
ax[1].plot(U)
ax[0].set_ylabel('Output')
ax[1].set_ylabel('control input')
ax[1].set_xlabel('time')

# Highlight the selected initial state (the lines should start here!)
ax[0].plot(0,x_0.T, 'o', color='black')

fig.align_ylabels()
fig.tight_layout()



# step 8 MPC main loop
x_0 = np.zeros((nx,1))
res_x = [x_0]
res_u = []

N_sim = 500 # total simulation step
sys.reset()
y_res = [sys.make_measurement()]
# kalman filter parameters
P_k = np.eye(Ar.shape[0],Ar.shape[1])
Q_k = np.ones((Ar.shape[0],Ar.shape[1]))*0.5
R_k = np.eye(Cr.shape[0],Cr.shape[0])*0.5

X = []

for i in range(N_sim-1):
    # fix initial condition of the state:
    lbx[:nx]=x_0
    ubx[:nx]=x_0
  
    # solve optimization problem
    res = solver(lbx=lbx,ubx=ubx,lbg=lbg,ubg=ubg)
    u_k = res['x'][(N+1)*nx:(N+1)*nx+nu,:]
    res_u.append(u_k)

    # simulate the system
    sys.simulate(time[i:i+2], (np.array(u_k)).reshape(1,2))
    # Append new measurement
    y_res.append(sys.make_measurement())
    
    xpred, Ppred = predict(x_0, (np.array(u_k)).reshape(2,1), P_k, Ar, Q_k, Br) # we need a full state estimator!
    K = K_G(P_k,Cr,R_k)
    x_next,P = update(xpred,K,y_res[i+1],Cr,P_k,R_k,Ar)
    
    res_x.append(x_next)
    x_0 = x_next
    
# Make an array from the list of arrays:
res_x = np.concatenate(res_x,axis=1)
res_u = np.concatenate(res_u, axis=1)
res_y = np.array(y_res)

time = np.linspace(0, 20, num=500)

fig, ax = plt.subplots(3,1, figsize=(10,8), sharex=True)
ax[0].plot(time,res_y)
ax[0].set_title('value 1 tracking on original system')
ax[1].set_title('control signal u')
ax[1].plot(time[:-1],res_u.T)
ax[0].set_ylabel('output')
ax[1].set_ylabel('control input')
ax[0].set_xlabel('time')
ax[1].set_xlabel('time')
ax[2].plot(time[:-1],res_x[:,:-1].T)


fig.align_ylabels()
fig.tight_layout()

plt.savefig('Step_response',dpi=250)

np.savetxt('res_u.txt',res_u)

# no matter how i have tried tuning the Q and R matrices, it's impossible to
# get a ideal contol effect. System ocillated and biased. Just as the prediction in mit-term screencast.
# conclusion: a incremental implementation is needed! or any other way to get through the trouble.