import numpy as np
import itertools
from Particle import Particle



class PSO: # all the material that is relavant at swarm leveel

    def __init__(self, w, a1, a2, dim, population_size, time_steps, search_range, points):

        # Here we use values that are (somewhat) known to be good
        # There are no "best" parameters (No Free Lunch), so try using different ones
        # There are several papers online which discuss various different tunings of a1 and a2
        # for different types of problems
        self.w = w # Inertia
        self.a1 = a2 # Attraction to personal best
        self.a2 = a2 # Attraction to global best
        self.dim = dim
        self.swarm = []

        for label in points:
            temp_swarm = [Particle(dim,-search_range,search_range, points[label], label, points) for i in range(int(population_size/2))]
            self.swarm += temp_swarm

        self.X = [p.position for p in self.swarm]
        self.Y = [p.label for p in self.swarm]

        print(self.X)
        print(self.Y)

        self.time_steps = time_steps

        # Initialising global best, you can wait until the end of the first time step
        # but creating a random initial best and fitness which is very high will mean you
        # do not have to write an if statement for the one off case

        self.best_swarm_pos = np.random.uniform(low=-500, high=500, size=dim)
        self.best_swarm_fitness = 1e100

    def run(self):
        for t in range(self.time_steps):
            for p in range(len(self.swarm)):
                particle = self.swarm[p]

                new_position = particle.position + particle.updateVel(self.w, self.a1, self.a2, particle.best_particle_pos, self.best_swarm_pos)

                if new_position@new_position > 1.0e+18: # The search will be terminated if the distance
                                                        # of any particle from center is too large
                    print('Time:', t,'Best Pos:',self.best_swarm_pos,'Best Fit:',self.best_swarm_fitness)
                    raise SystemExit('Most likely divergent: Decrease parameter values')

                self.swarm[p].setPos(new_position)


                new_fitness = particle.forward_pass(X, Y, p.position)

                if new_fitness < self.best_swarm_fitness:   # to update the global best both
                                                            # position (for velocity update) and
                                                            # fitness (the new group norm) are needed
                    self.best_swarm_fitness = new_fitness
                    self.best_swarm_pos = new_position
                    self.best_label = particle.label

            if t % 10 == 0: #we print only two components even it search space is high-dimensional
                #print("Time: %6d,  Best Fitness: %14.6f,  Best Pos: %9.4f,%9.4f" % (t,self.best_swarm_fitness,self.best_swarm_pos[0],self.best_swarm_pos[1]), end =" ")
                # perceptron = 0
                # for particle in self.swarm:
                #     perceptron += abs()
                # perceptron = (1/dim) * perceptron

                print("Time: %6d,  Best Fitness: %14.6f,  Best Pos: %9.4f,%9.4f"% (t,self.best_swarm_fitness,self.best_swarm_pos[0],self.best_swarm_pos[1]), end =" ")
                if self.dim>2:
                    print('...')
                else:
                    print('')


x = {1:[2,2], -1:[-2,-2]}
p = PSO(dim=2, w=0.7, a1=2.02, a2=2.02, population_size=10, time_steps=1001, search_range=10, points=x).run()

# W = s.get_best_solution()
# Y_pred = predict(X, W)
# accuracy = get_accuracy(Y, Y_pred)
# print("Accuracy: %.3f"% accuracy)
