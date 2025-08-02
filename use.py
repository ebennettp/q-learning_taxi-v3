import argparse

import gymnasium as gym
import numpy as np

from actions import evaluate, train
from config import Config, Hyperparameters

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.json",
        help="Path to the configuration file.",
    )
    parser.add_argument("-t", "--train-only", action="store_true")
    parser.add_argument("-e", "--evaluate-only", action="store_true")
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default="q_table.npy",
        help="Path to save the Q-table",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Do not save the Q-table"
    )

    args = parser.parse_args()

    print(f"{args = }")

    if args.train_only and args.evaluate_only:
        print("Only one of --train-only or --evaluate-only can be specified at once.")
        exit()

    with open(args.config, "r") as f:
        config = Config.model_validate_json(f.read())

    env = gym.make("Taxi-v3")
    q_table = None

    if not args.evaluate_only:
        with open(config.optimization.params_file, "r") as f:
            params = Hyperparameters.model_validate_json(f.read())

        print(f"Training parameters:\n{params.model_dump_json(indent=4)}")

        q_table = train(
            env, params, config.train.episodes, config.train.max_episode_steps
        )

        if not args.no_save:
            print("saving")
            q_table.dump(args.file)

    if not args.train_only:
        if args.evaluate_only:
            q_table = np.load(args.file, allow_pickle=True)

        if not q_table:
            print("Not able to load Q-Table")
            exit()

        mean_eval_reward = evaluate(
            env, q_table, config.eval.episodes, config.eval.max_episode_steps
        )

        print(f"Mean evaluation reward: {mean_eval_reward}")

    env.close()
