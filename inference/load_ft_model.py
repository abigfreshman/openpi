from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi0_tianyi")
checkpoint_dir = "/media/leon/output/pi0/clothes_washing/pi0_tianyi/clothes_washing_32_ft/30000"

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference on a dummy example.
example = {
    # "observation/exterior_image_1_left": ...,
    # "observation/wrist_image_left": ...,
    # ...
    # "prompt": "pick up the fork"
}
action_chunk = policy.infer(example)["actions"]