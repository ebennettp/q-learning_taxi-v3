import gymnasium as gym

from actions import evaluate, train
from config import Config, Hyperparameters

if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = Config.model_validate_json(f.read())

    with open(config.optimization.params_file, "r") as f:
        params = Hyperparameters.model_validate_json(f.read())

    print(params.model_dump_json(indent=4))

    env = gym.make("Taxi-v3")

    try:
        q_table = train(
            env, params, config.train.episodes, config.train.max_episode_steps
        )

        mean_eval_reward = evaluate(
            env, q_table, config.eval.episodes, config.eval.max_episode_steps
        )

        print(f"Mean evaluation reward: {mean_eval_reward}")
        q_table.dump("q_table.npy")
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Err: {e}")
    finally:
        env.close()
