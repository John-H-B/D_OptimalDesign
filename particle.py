import numpy as np

class Particle():
    def __init__(self,
                 k, max_dose, min_dose, fitness_function,
                 alpha=1.0, inertia=0.8, personal_importance=1.0, global_importance=1.0):
        self.k = k
        self.alpha = alpha
        self.inertia = inertia
        self.personal_importance = personal_importance
        self.global_importance = global_importance

        self.max_dose = max_dose
        self.min_dose = min_dose

        self.X = np.random.rand(self.k)
        self.X *= (self.max_dose - self.min_dose)
        self.X += (self.min_dose)

        self.W = np.random.rand(self.k-1)
        self.W /= (self.k - 1)  # to ensure that the sum will equal 1

        self.VX = np.random.randn(self.k)
        self.VW = np.random.randn(self.k - 1)

        self.fitness = fitness_function(self.X, self.W)
        self.X_best, self.W_best = self.X, self.W
        self.fitness_best = self.fitness


    def move(self):
        self.X = self.X + self.alpha*self.VX
        self.X = np.clip(self.X, self.min_dose, self.max_dose)

        self.W = self.W + self.alpha * self.VW
        self.W = np.clip(self.W, 0.0, 1.0)
        _sum = np.sum(self.W)
        if _sum > 1: # Ensures bounds are obeyed
            self.W = .99 * self.W/_sum


    def assess(self, fitness_function):
        self.fitness = fitness_function(self.X, self.W)
        if self.fitness > self.fitness_best:
            self.fitness_best = self.fitness
            self.X_best, self.W_best = self.X, self.W

    def update_velocity(self, X_gbest, W_gbest):
        assert np.shape(X_gbest) == np.shape(self.X)
        assert np.shape(W_gbest) == np.shape(self.W)
        r1, r2 = np.random.rand(2)
        self.VX = self.inertia*self.VX\
                  + r1 * self.personal_importance * (self.X_best - self.X)\
                  + r2 * self.global_importance * (X_gbest - self.X)
        self.VW = self.inertia * self.VW \
                  + r1 * self.personal_importance * (self.W_best - self.W) \
                  + r2 * self.global_importance * (W_gbest - self.W)

class Swarm():
    def __init__(self,
                 num_particles,k, max_dose, min_dose, fitness_function,
                 alpha=1.0, inertia=0.8, personal_importance=1.0, global_importance=1.0):
        self.fitness_function = fitness_function
        self.particles = []
        for _ in range(num_particles):
            self.particles.append(Particle(k, max_dose, min_dose, fitness_function,
                                           alpha, inertia,
                                           personal_importance=1.0, global_importance=1.0))
        self.global_fitness_best = -np.inf
        for p in self.particles:
            if p.fitness > self.global_fitness_best:
                self.global_fitness_best = p.fitness
                self.X_best = p.X
                self.W_best = p.W

    def one_step(self, i, verbose):
        for p in self.particles:
            p.move()
            p.assess(self.fitness_function)
            p.update_velocity(self.X_best, self.W_best)
            if p.fitness > self.global_fitness_best:
                if verbose:
                 print(f'New BEST at iteration {i}: {p.fitness}')
                self.global_fitness_best = p.fitness
                self.X_best = p.X
                self.W_best = p.W

    def optimise(self, iterations, verbose = False):
        for i in range(iterations):
            self.one_step(i, verbose)
