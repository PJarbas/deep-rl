import ray
from ray import tune
from configure import CONFIG


class TrainAgent:
    
    def __init__(self, env_name, algorithm, episode_reward_mean):
        self.env_name = env_name
        self.algorithm = algorithm
        self.config = CONFIG(self.env_name)
        self.stop = {"episode_reward_mean": episode_reward_mean}
    
    def init_ray(self):
        
        ray.shutdown()

        ray.init(
            num_cpus=3,
            include_dashboard=False,
            ignore_reinit_error=True,
            log_to_driver=False,
        )
    
    def run(self):       
        
        self.init_ray()
        
        # execute training
        analysis = tune.run(
            self.algorithm,
            config=self.config.get_config(),
            local_dir=self.config.local_dir,
            stop=self.stop,
            checkpoint_at_end=True,
            reuse_actors=True
        )
        
        return analysis


if __name__ == "__main__":
    # "LunarLander-v2", "CartPole-v0", "MountainCar-v0"
    ENV_NAME = "LunarLander-v2"
    EPISODE_REWARD_MEAN = 195
    ALGORITHM = "PPO"
    
    train = TrainAgent(ENV_NAME, algorithm=ALGORITHM ,episode_reward_mean=EPISODE_REWARD_MEAN)
    train.run()
    