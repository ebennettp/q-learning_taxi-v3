from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings  # , SettingsConfigDict


class Limits(BaseModel):
    min: float
    max: float


class EpsilonLimit(BaseModel):
    start: Limits
    min: Limits
    decay: Limits


class Stage(BaseModel):
    episodes: int
    max_episode_steps: int


class Optimization(BaseModel):
    trials: int = Field(default=100)
    params_file: str = Field(default="params.json")


class Hyperparameters(BaseModel):
    lr: float = Field(ge=0, le=1)
    gamma: float = Field(ge=0, le=1)
    epsilon: float = Field(ge=0, le=1)
    min_epsilon: float = Field(ge=0, le=1)
    epsilon_decay: float


class Config(BaseSettings):
    # model_config = SettingsConfigDict(
    #     env_file=".env", env_nested_delimiter="__", extra="ignore"
    # )

    train: Stage = Stage(episodes=10000, max_episode_steps=150)
    eval: Stage = Stage(episodes=100, max_episode_steps=50)

    optimization: Optimization = Optimization()

    lr: Limits = Field(default=...)
    gamma: Limits = Field(default=...)
    epsilon: EpsilonLimit = Field(default=...)


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = Config.model_validate_json(f.read())

    print(config.model_dump_json(indent=4))
