from embodiedbench.envs.eb_manipulation.EBManEnv import EBManEnv
import sys, random, faulthandler
import numpy as np
faulthandler.enable()
sys.stdout.reconfigure(line_buffering=True)
try:
    print("Init EB-Manipulation Env...")
    env = EBManEnv(eval_set='base', down_sample_ratio=0.1)
    env.reset()
    print("EB-Manipulation Env reset done")
    action_space = env.action_space
    print(f"Action space: {action_space}")
    
    for i in range(5):
        if hasattr(action_space, 'n'):
            action_id = random.randint(0, action_space.n - 1)
            action_str = env.language_skill_set[action_id]
            print(f"Action {i}: {action_str} (Discrete)")
            action = action_id
        else:
            # Box action space
            action = action_space.sample()
            print(f"Action {i}: Sampled Box action")
            
        obs, reward, done, info = env.step(action)
        if done: break
    env.close()
    print("EB-Manipulation PASS")
except Exception as e:
    print(f"EB-Manipulation FAIL: {e}")
    import traceback; traceback.print_exc()
