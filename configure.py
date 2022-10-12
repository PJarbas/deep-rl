

class CONFIG:
    def __init__(self, env_name):
        self.env_name = env_name
        self.local_dir = f"{self.env_name}_results"

    def get_config(self):
        
        config = {"env": self.env_name,

                # Change the following line to `“framework”: “tf”` to use tensorflow
                "framework": "torch",
                "model": {
                    "fcnet_hiddens": [32],
                    "fcnet_activation": "linear"
                },
        }
        
        return config