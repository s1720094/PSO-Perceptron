import numpy as np
import itertools

class Particle: # all the material that is relavant at the level of the individual particles

    def __init__(self, dim, minx, maxx, point, label, points):
        self.label = label
        self.position = np.empty
        self.points = points
        for dimension in point:
            self.position(dimension + np.random.uniform(low=minx, high=maxx))

        self.velocity = np.random.uniform(low=-0.1, high=0.1, size=dim)
        self.best_particle_pos = self.position
        self.dim = dim

        self.fitness = fitness(points)
        self.best_particle_fitness = self.fitness   # we couldd start with very large number here,
                                                    #but the actual value is better in case we are lucky

    def fitness(self):
        distance = 0
        for i in range(self.dim):
            distance += (self.position[i] - self.points[self.label][i])**2
        return (distance)**(1/self.dim)

    def setPos(self, pos):
        self.position = pos
        self.fitness = fitness()
        if self.fitness<self.best_particle_fitness:     # to update the personal best both
                                                        # position (for velocity update) and
                                                        # fitness (the new standard) are needed
                                                        # global best is update on swarm leven
            self.best_particle_fitness = self.fitness
            self.best_particle_pos = pos

    def updateVel(self, inertia, a1, a2, best_self_pos, best_swarm_pos):
                # Here we use the canonical version
                # V <- inertia*V + a1r1 (peronal_best - current_pos) + a2r2 (global_best - current_pos)
        cur_vel = self.velocity
        r1 = np.random.uniform(low=0, high=1, size = self.dim)
        r2 = np.random.uniform(low=0, high=1, size = self.dim)
        a1r1 = np.multiply(a1, r1)
        a2r2 = np.multiply(a2, r2)
        best_self_dif = np.subtract(best_self_pos, self.position)
        best_swarm_dif = np.subtract(best_swarm_pos, self.position)
                    # the next line is the main equation, namely the velocity update,
                    # the velocities are added to the positions at swarm level
        return inertia*cur_vel + np.multiply(a1r1, best_self_dif) + np.multiply(a2r2, best_swarm_dif)
