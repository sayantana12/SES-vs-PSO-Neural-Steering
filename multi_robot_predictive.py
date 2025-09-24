
# Multi-Robot Path Planning with SES/PSO and Predictive Obstacle + Robot Avoidance (Proactive ANN Steering)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import trange

class MultiRobotPathPlanner:
    def __init__(self, num_robots=6, workspace_size=40, max_velocity=5.0, max_acceleration=0.05,
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

        self.ksi_att = 0.5
        self.ksi_rep = 0.3

        np.random.seed(42)
        self.robot_positions = np.random.uniform(5, workspace_size-5, (num_robots, 2))
        self.robot_goals = np.random.uniform(5, workspace_size-5, (num_robots, 2))
        self.robot_velocities = np.zeros((num_robots, 2))
        self.paths = [[] for _ in range(num_robots)]

        self.obstacles = [
            {'type': 'circle', 'center': (20, 20), 'radius': 4},
            {'type': 'rect', 'bottom_left': (10, 5), 'width': 4, 'height': 8}
        ]

        self.input_size = 3
        self.hidden_size = 10
        self.output_size = 1
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

    def is_collision(self, pos):
        for obs in self.obstacles:
            if obs['type'] == 'circle':
                cx, cy = obs['center']
                if np.linalg.norm(np.array(pos) - np.array([cx, cy])) <= obs['radius'] + 1.0:
                    return True
            elif obs['type'] == 'rect':
                x, y = pos
                x0, y0 = obs['bottom_left']
                if x0 <= x <= x0 + obs['width'] and y0 <= y <= y0 + obs['height']:
                    return True
        return False

    def generate_training_data(self):
        num_samples = 2000
        self.X_train = np.zeros((num_samples, 3))
        self.y_train = np.zeros(num_samples)

        for i in range(num_samples):
            goal_angle = np.random.uniform(-np.pi, np.pi)
            threat_angle = np.random.uniform(-np.pi, np.pi)
            threat_dist = np.random.uniform(1.0, self.influence_distance + 3.0)
            F_att = self.ksi_att * threat_dist
            if threat_dist < self.influence_distance:
                F_rep = self.ksi_rep * ((1/threat_dist - 1/self.influence_distance) / (threat_dist**2))
            else:
                F_rep = 0
            deviation = np.arctan2(F_rep * np.sin(threat_angle) + F_att * np.sin(goal_angle),
                                   F_rep * np.cos(threat_angle) + F_att * np.cos(goal_angle))
            self.X_train[i] = [goal_angle, threat_angle, threat_dist]
            self.y_train[i] = deviation

    def diversity_penalty(self, population):
        centroid = np.mean(population, axis=0)
        return 1.0 / (np.mean(np.linalg.norm(population - centroid, axis=1)) + 1e-6)

    def train_ann_with_ses(self):
        pop = np.random.uniform(-1, 1, (self.population_size, self.vector_size))
        fitness_history = []

        for gen in trange(self.generations, desc="SES"):
            scores = []
            for i in range(self.population_size):
                W1, b1, W2, b2 = self.decode_ann(pop[i])
                pred = self.ann_forward(self.X_train, W1, b1, W2, b2).flatten()
                mse = np.mean(self.angle_diff(self.y_train, pred)**2)
                scores.append(mse)
            scores = np.array(scores)
            elite_idx = np.argmin(scores)
            elite = pop[elite_idx]
            new_pop = [elite.copy()]
            while len(new_pop) < self.population_size:
                mutation = np.random.normal(0, 0.1, size=self.vector_size)
                new_pop.append(np.clip(elite + mutation, -5, 5))
            pop = np.array(new_pop)
            fitness_history.append(np.min(scores))
        self.best_weights = self.decode_ann(pop[np.argmin(scores)])
        self.plot_training_results(fitness_history)
        self.animate_predictive_movement()

    def train_ann_with_pso(self):
        pop = np.random.uniform(-1, 1, (self.population_size, self.vector_size))
        v = np.zeros_like(pop)
        p_best = pop.copy()
        p_scores = np.full(self.population_size, np.inf)
        g_best = None
        g_score = np.inf
        fitness_history = []

        for gen in trange(self.generations, desc="PSO"):
            for i in range(self.population_size):
                W1, b1, W2, b2 = self.decode_ann(pop[i])
                pred = self.ann_forward(self.X_train, W1, b1, W2, b2).flatten()
                mse = np.mean(self.angle_diff(self.y_train, pred)**2)
                if mse < p_scores[i]:
                    p_best[i] = pop[i]
                    p_scores[i] = mse
                if mse < g_score:
                    g_best = pop[i]
                    g_score = mse
            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.vector_size), np.random.rand(self.vector_size)
                v[i] = 0.729 * v[i] + 1.49 * r1 * (p_best[i] - pop[i]) + 1.49 * r2 * (g_best - pop[i])
                pop[i] = np.clip(pop[i] + v[i], -5, 5)
            fitness_history.append(g_score)
        self.best_weights = self.decode_ann(g_best)
        self.plot_training_results(fitness_history)
        self.animate_predictive_movement()

    def plot_training_results(self, fitness_history):
        plt.plot(fitness_history)
        plt.title("Training Loss")
        plt.xlabel("Generation")
        plt.ylabel("MSE")
        plt.grid(True)
        plt.show()

    def animate_predictive_movement(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, self.workspace_size)
        ax.set_ylim(0, self.workspace_size)
        for obs in self.obstacles:
            if obs['type'] == 'circle':
                ax.add_patch(plt.Circle(obs['center'], obs['radius'], color='red', alpha=0.3))
            elif obs['type'] == 'rect':
                x0, y0 = obs['bottom_left']
                ax.add_patch(plt.Rectangle((x0, y0), obs['width'], obs['height'], color='orange', alpha=0.3))
        bots = [ax.plot([], [], 'o')[0] for _ in range(self.num_robots)]
        paths = [ax.plot([], [], '--')[0] for _ in range(self.num_robots)]
        goals = [ax.plot(self.robot_goals[i,0], self.robot_goals[i,1], 'x')[0] for i in range(self.num_robots)]
        reached = [False]*self.num_robots

        def update(frame):
            W1, b1, W2, b2 = self.best_weights
            for i in range(self.num_robots):
                if reached[i]: continue
                pos = self.robot_positions[i]
                goal_vec = self.robot_goals[i] - pos
                dist_to_goal = np.linalg.norm(goal_vec)
                if dist_to_goal < 1.0:
                    reached[i] = True
                    continue
                goal_dir = goal_vec / (dist_to_goal + 1e-6)
                angle_to_goal = np.arctan2(goal_dir[1], goal_dir[0])
                closest_obj = None
                closest_dist = float("inf")
                closest_angle = 0

                # check for other robots
                for j in range(self.num_robots):
                    if i == j: continue
                    dist = np.linalg.norm(pos - self.robot_positions[j])
                    if dist < closest_dist and dist < self.influence_distance:
                        closest_obj = 'robot'
                        closest_dist = dist
                        delta = self.robot_positions[j] - pos
                        closest_angle = np.arctan2(delta[1], delta[0])

                # check for obstacles
                for obs in self.obstacles:
                    if obs['type'] == 'circle':
                        center = np.array(obs['center'])
                        d = np.linalg.norm(pos - center) - obs['radius']
                        if d < closest_dist:
                            closest_obj = 'obstacle'
                            closest_dist = d
                            delta = center - pos
                            closest_angle = np.arctan2(delta[1], delta[0])
                    elif obs['type'] == 'rect':
                        rect_center = np.array(obs['bottom_left']) + np.array([obs['width']/2, obs['height']/2])
                        d = np.linalg.norm(pos - rect_center)
                        if d < closest_dist:
                            closest_obj = 'obstacle'
                            closest_dist = d
                            delta = rect_center - pos
                            closest_angle = np.arctan2(delta[1], delta[0])

                input_vec = np.array([[angle_to_goal, closest_angle, closest_dist]])
                steer = self.ann_forward(input_vec, W1, b1, W2, b2)[0,0]
                rot = np.array([[np.cos(steer), -np.sin(steer)],
                                [np.sin(steer),  np.cos(steer)]])
                move_dir = np.dot(rot, goal_dir)
                new_pos = pos + move_dir * self.max_velocity * self.dt
                if not self.is_collision(new_pos):
                    self.robot_positions[i] = new_pos
                self.paths[i].append(self.robot_positions[i].copy())
                bots[i].set_data(self.robot_positions[i][0], self.robot_positions[i][1])
                xs, ys = zip(*self.paths[i])
                paths[i].set_data(xs, ys)
            return bots + paths + goals

        ani = FuncAnimation(fig, update, frames=150, interval=300, blit=True, repeat=False)
        plt.show()

if __name__ == "__main__":
    planner = MultiRobotPathPlanner(use_pso=False)
