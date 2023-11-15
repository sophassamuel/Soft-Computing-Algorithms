import numpy as np

class Particle:
    def __init__(self, x0):
        self.position = np.array(x0)
        self.velocity = np.zeros_like(self.position)
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')

    def update_position(self):
        self.position += self.velocity

    def update_velocity(self, global_best_position, w, c1, c2):
        r1, r2 = np.random.rand(len(self.position)), np.random.rand(len(self.position))
        self.velocity = w * self.velocity + c1 * r1 * (self.best_position - self.position) + c2 * r2 * (global_best_position - self.position)

    def evaluate_fitness(self, objective_function):
        self.fitness = objective_function(self.position)
        if self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position = self.position.copy()

def particle_swarm_optimization(objective_function, bounds, num_particles, max_iterations, w, c1, c2):
    particles = [Particle([np.random.uniform(b[0], b[1]) for b in bounds]) for _ in range(num_particles)]
    global_best_position = min(particles, key=lambda p: p.best_fitness).best_position.copy()
    global_best_fitness = min(particles, key=lambda p: p.best_fitness).best_fitness

    for _ in range(max_iterations):
        for particle in particles:
            particle.update_velocity(global_best_position, w, c1, c2)
            particle.update_position()
            particle.evaluate_fitness(objective_function)

            if particle.best_fitness < global_best_fitness:
                global_best_fitness = particle.best_fitness
                global_best_position = particle.best_position.copy()

    return global_best_position, global_best_fitness

# Example usage
def objective_function(x):
    return np.sum(x**2)

bounds = [(-5, 5), (-5, 5), (-5, 5)]  # Variable bounds
num_particles, max_iterations, w, c1, c2 = 20, 100, 0.5, 1.0, 1.0

best_position, best_fitness = particle_swarm_optimization(objective_function, bounds, num_particles, max_iterations, w, c1, c2)

print("Best position:", best_position)
print("Best fitness:", best_fitness)
