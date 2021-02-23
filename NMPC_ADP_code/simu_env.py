import numpy as np

class DoubleInvertedPendulum():
    def __init__(self):
        # basic states
        self.pos = 0.
        self.ang1 = 0.
        self.ang2 = 0.
        self.pos_dot = 0.
        self.ang1_dot = 0.
        self.ang2_dot = 0.
        self.states = [self.pos, self.ang1, self.ang2, self.pos_dot, self.ang1_dot, self.ang2_dot]
        # settings
        self.L1 = 0.5
        self.L2 = 0.5
        self.cart_mass = 0.6
        self.rod1_mass = 0.2
        self.rod2_mass = 0.2
        # control variable limit
        self.force_list = [-4,4] 
        # inertia
        self.Jrod_1 = self.rod1_mass * self.L1 ** 2 / 3
        self.Jrod_2 = self.rod2_mass * self.L2 ** 2 / 3
        self.g = 9.8 # gravity

        self.f = 0. #current force

    @property
    def curr_states(self):
        return self.states

    @curr_states.setter
    def curr_states(self, new_states):
        #TODO add constraints
        self.states = new_states

    def get_dynamics(self):
        '''
        [C M] @ dot{q} + g = Qg
        get the dynamics matrices
        '''
        h1 = self.cart_mass + self.rod1_mass + self.rod2_mass
        h2 = self.rod1_mass * self.L1/2 + self.rod2_mass * self.L1
        h3 = self.rod2_mass * self.L2/2
        h4 = self.rod1_mass * (self.L1/2) ** 2 + self.rod2_mass * self.L1 ** 2 + self.Jrod_1
        h5 = self.rod2_mass * self.L2/2 * self.L1
        h6 = self.rod2_mass * (self.L2/2) ** 2 + self.Jrod_2
        h7 = self.rod1_mass * self.L1/2 * self.g + self.rod2_mass * self.L1 * self.g
        h8 = self.rod2_mass * self.L2/2 *self.g

        M_q = [[h1, h2*np.cos(self.states[1]), h3*np.cos(self.states[2])], \
              [h2*np.cos(self.states[1]), h4, h5*np.cos(self.states[1]-self.states[2])],\
              [h3*np.cos(self.states[2]), h5*np.cos(self.states[1]-self.states[2]), h6]]

        C_q_q_dot = [[0, -h2*self.states[4]*np.sin(self.states[1]), -h3*self.states[5]*np.sin(self.states[2])],\
                     [0, 0, h5*self.states[5]*np.sin(self.states[1]-self.states[2])],\
                     [0, -h5*self.states[4]*np.sin(self.states[1]-self.states[2]), 0]]

        q = self.states
        g_q = [0, -h7*np.sin(self.states[1]), -h8*np.sin(self.states[2])]
        Q_q = [self.f, 0., 0.]

        M_q, C_q_q_dot, q, g_q, Q_q = np.array(M_q) , np.array(C_q_q_dot), np.array(q).reshape(len(q), 1), \
                                        np.array(g_q).reshape(len(g_q), 1), np.array(Q_q).reshape(len(Q_q), 1)

        CM = np.concatenate((C_q_q_dot, M_q), axis=1)
        return CM, g_q, Q_q



if __name__ == '__main__':
    import time
    env = DoubleInvertedPendulum()
    for _ in range(100):
        test = np.random.randn(6)
        env.curr_states = test
        CM, g_q, Q_q = env.get_dynamics()
        print(CM)
