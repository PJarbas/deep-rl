import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from configure import CONFIG


class TestAgent:
    def __init__(self, env_name):
        self.env_name = env_name
        self.config = CONFIG(self.env_name)
        
    def restore_trainer_from_checkpoint(self):

        analysis = tune.ExperimentAnalysis(experiment_checkpoint_path=f"{self.config.local_dir}/PPO")
        
        # restore a trainer from the last checkpoint
        trial = analysis.get_best_logdir("episode_reward_mean", "max")

        checkpoint = analysis.get_best_checkpoint(trial, "training_iteration", "max",)

        trainer = PPOTrainer(config=self.config.get_config())
        trainer.restore(checkpoint)
        
        return trainer
    
    def make_inference(self):
        
        trainer = self.restore_trainer_from_checkpoint()
        
        output_video = f"{self.env_name}_output_video.mp4"
        
        env = gym.make(self.env_name)

        video = VideoRecorder(env, output_video)

        observation = env.reset()

        done = False
        while not done:
            env.render()
            video.capture_frame()
            action = trainer.compute_single_action(observation)
            observation, reward, done, info = env.step(action)
            
            print("\n................................................................")
            print(f"Inference on {self.env_name}")
            print(f"Action: {action}")
            print(f"Observation: {observation}")
            print(f"Reward: {reward}")
            print("................................................................")
            
        video.close()
        env.close()


if __name__ == "__main__":
    ENV_NAME = "CartPole-v0"
    agent = TestAgent(ENV_NAME)
    agent.make_inference()