import json
import traceback

import gymnasium as gym
import numpy as np
import optuna

from actions import evaluate
from config import Config


def main(env: gym.Env, config: Config) -> optuna.Study:
    def objective(trial: optuna.Trial):
        q_table = np.zeros((env.observation_space.n, env.action_space.n))  # type: ignore
        q_table.shape

        lr = trial.suggest_float("lr", config.lr.min, config.lr.max, log=True)
        gamma = trial.suggest_float("gamma", config.gamma.min, config.gamma.max)
        epsilon = trial.suggest_float(
            "epsilon", config.epsilon.start.min, config.epsilon.start.max
        )
        min_epsilon = trial.suggest_float(
            "min_epsilon", config.epsilon.min.min, config.epsilon.min.max
        )
        epsilon_decay = trial.suggest_float(
            "epsilon_decay",
            config.epsilon.decay.min,
            config.epsilon.decay.max,
            log=True,
        )

        for episode in range(config.train.episodes):
            episode_reward = 0
            obs, info = env.reset()
            done = False
            steps = 0

            while not done:
                if np.random.rand() < epsilon:  # Explore
                    action = env.action_space.sample()
                else:  # Exploit
                    action = np.argmax(q_table[obs])

                next_observation, reward, done, truncated, info = env.step(action)
                episode_reward += reward  # pyright: ignore[reportOperatorIssue]

                max_next_reward = q_table[next_observation][
                    np.argmax(q_table[next_observation])
                ]

                # Update Q-table (Bellman equation)
                q_table[obs][action] += lr * (
                    reward + gamma * max_next_reward - q_table[obs][action]
                )

                obs = next_observation

                steps += 1
                if steps > config.train.max_episode_steps:
                    break

            epsilon = max(min_epsilon, epsilon - epsilon_decay)

            trial.report(episode_reward, step=episode)
            if trial.should_prune():
                raise optuna.TrialPruned()

        eval_reward = evaluate(
            env, q_table, config.train.episodes, config.train.max_episode_steps
        )

        return eval_reward / config.eval.episodes

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=config.optimization.trials)

    return study


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = Config.model_validate_json(f.read())

    study = None
    env = gym.make("Taxi-v3")

    try:
        study = main(env, config)
    except Exception as e:
        traceback.print_exc()
        print(f"Err: {e}")
    finally:
        env.close()

    if study:
        with open(config.optimization.params_file, "w") as f:
            json.dump(study.best_params, f, indent=4)
