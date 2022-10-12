import ray
from ray import tune
from configure import CONFIG


class TrainAgent:
    
    def __init__(self, env_name):
        self.env_name = env_name
        self.config = CONFIG(self.env_name)
        self.stop = {"episode_reward_mean": 195}
    
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
            "PPO",
            config=self.config.get_config(),
            local_dir=self.config.local_dir,
            stop=self.stop,
            checkpoint_at_end=True,
            reuse_actors=True
        )
        
        return analysis


if __name__ == "__main__":
    ENV_NAME = "CartPole-v0"
    train = TrainAgent(ENV_NAME)
    train.run()
    