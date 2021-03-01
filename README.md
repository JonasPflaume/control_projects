## 1.model predictive control - project ##

3 Masses,2 motors system control project given by Department of Telecommunication Systems
Internet of Things for Smart Buildings tu-berlin.

model identification:  eigensystem realization algorithm  
state estimation:      kalman filter  
controller design:      model predictive control, optimization problem solved by casadi  
**env:	code need to be run under anaconda. It's a long time after finishing this project, so that uni research team provided state space system might not be able to load by pyarmor any more(quite frustrating:/).**


## 2.(Feb 23. 2021)start building a adaptive dynamic programming controller to double inverted pendulum.  

~~- TODO: Build a double inverted pendulum  ~~
~~- TODO: Implement collocation utils  ~~
~~- TODO: Implement nonlinear MPC  ~~
- TODO: Add obstacle avoidance
- TODO: Implement ADP controller  

ADP method followed **Optimal Control of Unknown Nonlinear Discrete‐Time Systems Using the Iterative Globalized Dual Heuristic Programming Algorithm Derong Liu et. al**  
Nonlinear MPC mothod will also be implemented. Possibly by collocation.
