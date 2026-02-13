import sys
import os
sys.path.insert(0, os.path.abspath("."))
from embodiedbench.envs.eb_navigation.EBNavEnv import EBNavigationEnv

print("Initializing EBNavigationEnv...")
try:
    env = EBNavigationEnv(eval_set='base', down_sample_ratio=0.01)
    print("Resetting env...")
    obs = env.reset()
    print("Success!")
    env.close()
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()
