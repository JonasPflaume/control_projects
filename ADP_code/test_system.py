import state_space_system
import matplotlib.pyplot as plt
import numpy as np

# Import the system:
print('------------------------------------------------')
print('Import unknown state-space system.')
sys = state_space_system.sys()

"""
sys.set_random -> Set random intial condition.

Use this method whenever you want to start from a random (excited)
initial state.
"""
sys.set_random()



"""
sys.simulate -> Simulate the system for N-1 timesteps and return the output as a vector.

Inputs :
- time : time vector with shape (N,), where N>=2 (at least start and stop) (in seconds)
- u    : input matrix with shape (N-1, 2)

Returns:
- y    : output vector with shape (N,3)

Two different applications are possible:
1) Pass a full input sequence at once (e.g. for system identification) -> Get sequence of measurements.

2) Use recursively (e.g. for MPC application)
get measurement -> estimate state -> compute input -> **simulate** -> next measurement


First we demonstrate application 1):
"""
# Time vector from t=0 to t=20 with 500 steps
time = np.linspace(0, 20, num=500)
# Control input all zero. We need one less control input than number of timesteps.
u = np.zeros((time.shape[0]-1, 2))


print('------------------------------------------------')
print('Simulate system (input all zero).')
# Simulate the system (for multiple steps at once!)
y = sys.simulate(time, u)

# Plot the result.
fig, ax = plt.subplots(1, 2)
ax[0].plot(time, y)
ax[0].set_xlabel('time in [s]')
ax[0].set_ylabel('Outputs')
ax[0].set_title('No control input.')


"""
Now we demonstrate application 2):

We loop over the control inputs and apply them one after the other
in successive calls of sys.simulate.
"""


# Reset state:
print('------------------------------------------------')
print('Reset state.')

"""
sys.reset -> All states at zero (equilibrium).
"""
sys.reset()

print('------------------------------------------------')
print('Simulate system (random input).')

# Contol input all zero:
u = np.random.rand(time.shape[0]-1, 2)-0.5
# Initial measurement
y = [sys.make_measurement()]
for k in range(time.shape[0]-1):
    # Simulate for one timestep:
    sys.simulate(time[k:k+2], u[[k]])
    # Append new measurement
    y.append(sys.make_measurement())

# Concatenate:
y = np.array(y)

# Plot the results:
ax[1].plot(time, y)
ax[1].set_xlabel('time in [s]')
ax[1].set_ylabel('Outputs')
ax[1].set_title('Random control sequence.')
plt.show()
