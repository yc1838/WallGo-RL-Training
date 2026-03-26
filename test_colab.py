"""Colab diagnostic script — run this to find out what's broken."""
import sys
import os

# Force all output to be unbuffered and visible
sys.stdout.reconfigure(line_buffering=True)
sys.stderr = sys.stdout  # merge error output so we can see everything

# CRITICAL: Prevent TensorFlow from grabbing the GPU and causing CUDA conflicts on Colab.
# This prevents the "Unable to register cuFFT factory" errors and potential silent crashes.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    print("  TensorFlow GPU disabled to avoid CUDA conflicts.")
except ImportError:
    pass

print("=== STEP 1: Python OK ===")

print("=== STEP 2: importing torch ===")
import torch
print(f"  CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

print("=== STEP 3: importing sb3 ===")
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
print("  sb3 OK")

print("=== STEP 4: importing our code ===")
from wallgo_gym import WallGoGymEnv
from evaluate import RandomAgent
# We MUST import the custom CNN class so the MaskablePPO.load unpickler can find it!
from train import WallGoCNN
print("  our code OK")

print("=== STEP 5: checking checkpoint ===")
cp_dir = "checkpoints"
files = os.listdir(cp_dir)
print(f"  files: {files}")
cp_path = None
for f in files:
    if f.endswith(".zip"):
        cp_path = os.path.join(cp_dir, f)
        break
if not cp_path:
    print("  ERROR: no .zip checkpoint found!")
    sys.exit(1)
print(f"  using: {cp_path}")

print("=== STEP 6: loading model (this is where it probably fails) ===")
try:
    # Explicitly clear any existing CUDA cache just in case
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"  attempting to load: {cp_path} on cuda...")
    model = MaskablePPO.load(cp_path, device="cuda")
    print(f"  model loaded SUCCESS on: {model.device}")
except Exception as e:
    print(f"  FAILED to load on cuda: {e}")
    # Print more debug info
    import traceback
    traceback.print_exc()
    
    print("  trying cpu instead...")
    try:
        model = MaskablePPO.load(cp_path, device="cpu")
        print(f"  model loaded SUCCESS on: {model.device}")
    except Exception as e2:
        print(f"  FAILED on cpu too: {e2}")
        traceback.print_exc()
        sys.exit(1)

print("=== STEP 7: creating env ===")
from action_encoding import ACTION_SPACE_SIZE
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SimpleTestEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self._inner = WallGoGymEnv()
        self.observation_space = self._inner.observation_space
        self.action_space = self._inner.action_space
    def reset(self, **kw):
        return self._inner.reset(**kw)
    def step(self, a):
        return self._inner.step(a)
    def action_masks(self):
        return self._inner.action_masks()

def _mask_fn(env):
    return env.action_masks()

env = ActionMasker(SimpleTestEnv(), _mask_fn)
print("  env OK")

print("=== STEP 8: test predict ===")
obs, _ = env.reset()
mask = env.env.action_masks()
action, _ = model.predict(obs, action_masks=mask)
print(f"  predicted action: {action}")

print("=== STEP 9: test model.learn (tiny) ===")
model.set_env(env)
model.learn(total_timesteps=100, reset_num_timesteps=True)
print("  learn OK!")

print("\n=== ALL TESTS PASSED — Colab is ready for training! ===")
