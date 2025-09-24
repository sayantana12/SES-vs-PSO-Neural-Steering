
# Multi-Robot Path Planning using Swarm Evolutionary Strategy (SES) and Particle Swarm Optimization (PSO)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import trange
import time
import pandas as pd

class MultiRobotPathPlanner:
    def __init__(self, num_robots=8, workspace_size=40, max_velocity=5.0, max_acceleration=0.05,
                 dt=2.0, influence_distance=8.0, population_size=30, generations=100, use_pso=False):

        self.num_robots = num_robots
        self.workspace_size = workspace_size
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.dt = dt
        self.influence_distance = influence_distance
        self.population_size = population_size
        self.generations = generations
        self.use_pso = use_pso

        self.robot_deviation_angles = np.zeros(self.num_robots)
        self.ksi_att = 0.5
        self.ksi_rep = 0.3

        np.random.seed(42)
        self.robot_positions = np.random.uniform(5, workspace_size-5, (num_robots, 2))
        self.robot_goals = np.random.uniform(5, workspace_size-5, (num_robots, 2))
        self.robot_velocities = np.zeros((num_robots, 2))

        self.input_size = 3
        self.hidden_size = 10
        self.output_size = 1

        self.travelling_times = np.zeros(num_robots)
        self.cooperation_counts = np.zeros(num_robots)
        self.deviation_logs = []
        self.att_history = []
        self.paths = [[] for _ in range(self.num_robots)]

        self.initialize_ann_structure()
        self.generate_training_data()

        if self.use_pso:
            self.train_ann_with_pso()
        else:
            self.train_ann_with_ses()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def tanh(self, x):
        return np.tanh(np.clip(x, -500, 500))

    def angle_diff(self, a, b):
        return np.arctan2(np.sin(a - b), np.cos(a - b))

    def initialize_ann_structure(self):
        self.w1_size = self.input_size * self.hidden_size
        self.b1_size = self.hidden_size
        self.w2_size = self.hidden_size * self.output_size
        self.b2_size = self.output_size
        self.vector_size = self.w1_size + self.b1_size + self.w2_size + self.b2_size

    def decode_ann(self, particle):
        idx = 0
        W1 = particle[idx:idx + self.w1_size].reshape(self.input_size, self.hidden_size)
        idx += self.w1_size
        b1 = particle[idx:idx + self.b1_size]
        idx += self.b1_size
        W2 = particle[idx:idx + self.w2_size].reshape(self.hidden_size, self.output_size)
        idx += self.w2_size
        b2 = particle[idx:idx + self.b2_size]
        return W1, b1, W2, b2

    def ann_forward(self, X, W1, b1, W2, b2):
        hidden = self.sigmoid(np.dot(X, W1) + b1)
        output = self.tanh(np.dot(hidden, W2) + b2) * np.pi
        return output

    def generate_training_data(self):
        num_samples = 2000
        self.X_train = np.zeros((num_samples, 3))
        self.y_train = np.zeros(num_samples)

        for i in range(num_samples):
            goal_angle = np.random.uniform(-np.pi, np.pi)
            crit_angle = np.random.uniform(-np.pi, np.pi)
            crit_dist = np.random.uniform(1.0, self.influence_distance + 2.0)
            F_att = self.ksi_att * crit_dist
            F_rep = self.ksi_rep * ((1/crit_dist - 1/self.influence_distance) / (crit_dist**2)) if crit_dist < self.influence_distance else 0
            deviation = np.arctan2(F_rep * np.sin(crit_angle) + F_att * np.sin(goal_angle),
                                   F_rep * np.cos(crit_angle) + F_att * np.cos(goal_angle))
            self.X_train[i] = [goal_angle, crit_angle, crit_dist]
            self.y_train[i] = deviation

    def diversity_penalty(self, population):
        if len(population) <= 1:
            return 0.0
        centroid = np.mean(population, axis=0)
        diversity = np.mean(np.linalg.norm(population - centroid, axis=1))
        return 1.0 / (diversity + 1e-6)

    def train_ann_with_ses(self):
        population = np.random.uniform(-1, 1, (self.population_size, self.vector_size))
        fitness_history = []

        for generation in trange(self.generations, desc="Training with SES"):
            fitness_scores = []

            for i in range(self.population_size):
                W1, b1, W2, b2 = self.decode_ann(population[i])
                y_pred = self.ann_forward(self.X_train, W1, b1, W2, b2).flatten()
                mse = np.mean(self.angle_diff(self.y_train, y_pred)**2)
                fitness_scores.append(mse)

            fitness_scores = np.array(fitness_scores)
            penalty = self.diversity_penalty(population)
            total_fitness = fitness_scores + 0.1 * penalty

            elite_idx = np.argmin(total_fitness)
            elite = population[elite_idx]
            new_population = [elite.copy()]

            while len(new_population) < self.population_size:
                mutation = np.random.normal(0, 0.1, size=self.vector_size)
                child = elite + mutation
                new_population.append(np.clip(child, -5, 5))

            population = np.array(new_population)
            fitness_history.append(np.min(fitness_scores))

        self.best_weights = self.decode_ann(population[np.argmin(fitness_scores)])
        self.fitness_history = fitness_history
        self.plot_training_results(fitness_history)

    def train_ann_with_pso(self):
        population = np.random.uniform(-1, 1, (self.population_size, self.vector_size))
        velocities = np.zeros_like(population)
        p_best = population.copy()
        p_best_scores = np.full(self.population_size, np.inf)
        g_best = None
        g_best_score = np.inf
        fitness_history = []

        w, c1, c2 = 0.729, 1.49445, 1.49445

        for gen in trange(self.generations, desc="Training with PSO"):
            for i in range(self.population_size):
                W1, b1, W2, b2 = self.decode_ann(population[i])
                y_pred = self.ann_forward(self.X_train, W1, b1, W2, b2).flatten()
                mse = np.mean(self.angle_diff(self.y_train, y_pred) ** 2)

                if mse < p_best_scores[i]:
                    p_best[i] = population[i]
                    p_best_scores[i] = mse

                if mse < g_best_score:
                    g_best = population[i]
                    g_best_score = mse

            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.vector_size), np.random.rand(self.vector_size)
                velocities[i] = (w * velocities[i] + c1 * r1 * (p_best[i] - population[i])
                                 + c2 * r2 * (g_best - population[i]))
                population[i] += velocities[i]
                population[i] = np.clip(population[i], -5, 5)

            fitness_history.append(g_best_score)

        self.best_weights = self.decode_ann(g_best)
        self.fitness_history = fitness_history
        self.plot_training_results(fitness_history)

    def plot_training_results(self, fitness_history):
        y_pred = self.ann_forward(self.X_train, *self.best_weights).flatten()
        errors = self.angle_diff(self.y_train, y_pred)

        plt.figure()
        plt.hist(errors, bins=50, alpha=0.7, color='skyblue')
        plt.title("Prediction Error Histogram")
        plt.xlabel("Angle Error (rad)")
        plt.ylabel("Frequency")
        plt.grid()
        plt.show()

        plt.figure()
        plt.plot(fitness_history, label='Best MSE')
        plt.title(f"Fitness Progress during {'PSO' if self.use_pso else 'SES'} Training")
        plt.xlabel("Generation")
        plt.ylabel("MSE")
        plt.grid()
        plt.legend()
        plt.show()

    def get_training_loss(self):
        return self.fitness_history if hasattr(self, 'fitness_history') else []

if __name__ == "__main__":
    planner_ses = MultiRobotPathPlanner(use_pso=False)
    planner_pso = MultiRobotPathPlanner(use_pso=True)

    plt.plot(planner_ses.get_training_loss(), label='SES')
    plt.plot(planner_pso.get_training_loss(), label='PSO')
    plt.title("Training Loss: SES vs PSO")
    plt.xlabel("Generation")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.show()
