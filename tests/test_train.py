import pytest
import torch
import numpy as np
from gymnasium import spaces
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import train

# Test WallGoCNN
def test_wallgo_cnn_features_extractor():
    # Setup dummy env observation space
    obs_space = spaces.Box(low=0.0, high=1.0, shape=(6, 7, 7), dtype=np.float32)
    # Instantiate extractor
    cnn = train.WallGoCNN(obs_space, features_dim=256)
    
    # Check happy path: correct shape output
    sample_obs = torch.as_tensor(obs_space.sample()[None]).float()
    out = cnn(sample_obs)
    assert out.shape == (1, 256)

def test_selfplay_env_set_opponent():
    env = train.SelfPlayEnv()
    with pytest.raises(Exception):
        # We test that passing an invalid path triggers an error
        # Since MaskablePPO will fail to load the zip.
        env.set_opponent_path("non_existent_model.zip")
