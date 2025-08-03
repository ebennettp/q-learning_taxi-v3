import gymnasium as gym
import numpy as np

from config import Hyperparameters


def train(
    env: gym.Env, params: Hyperparameters, episodes: int, max_episodes_steps: int
) -> np.ndarray[tuple]:
    """Train a Q-learning agent. Q-table is built from the environment's observation
    and action spaces. Note that this function does not close the environment.

    Args:
        env (gym.Env): Pre-made Gym environment
        params (Hyperparameters): Hyperparameters class containing all the values.
            See `config.py` for more information.
        episodes (int): Number of episodes to train the agent.
        max_episodes_steps (int): Max number of steps before episode is terminated.

    Returns:
        np.ndarray[tuple]: Q-table with shape (observation_space, action_space).
    """

    q_table = np.zeros((env.observation_space.n, env.action_space.n))  # type: ignore
    epsilon = params.epsilon

    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        steps = 0

        while not done:
            if np.random.rand() < params.epsilon:  # Explore
                action = env.action_space.sample()
            else:  # Exploit
                action = np.argmax(q_table[obs])

            next_observation, reward, terminated, truncated, info = env.step(action)

            max_next_reward = q_table[next_observation][
                np.argmax(q_table[next_observation])
            ]

            # Update Q-table (Bellman equation)
            q_table[obs][action] += params.lr * (
                reward + params.gamma * max_next_reward - q_table[obs][action]
            )

            obs = next_observation
            done = terminated or truncated

            steps += 1
            if steps > max_episodes_steps:
                break

        epsilon = max(params.min_epsilon, epsilon - params.epsilon_decay)

    return q_table


def evaluate(
    env: gym.Env, q_table: np.ndarray[tuple], episodes: int, max_episode_steps: int
) -> float:
    """Calculate the average reward over a number of episodes. Note that this function
    does not close the environment.

    Args:
        env (gym.Env):
        q_table (np.ndarray[tuple]):
        episodes (int):
        max_episode_steps (int):

    Returns:
        float: Average episode reward.
    """
    total_reward = 0

    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done:
            action = np.argmax(q_table[obs])

            next_observation, reward, done, truncated, info = env.step(action)
            episode_reward += reward  # pyright: ignore[reportOperatorIssue]
            obs = next_observation

            steps += 1
            if steps > max_episode_steps:
                break

        total_reward += episode_reward

    return total_reward / episodes
