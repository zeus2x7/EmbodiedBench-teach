from embodiedbench.envs.eb_alfred.EBAlfEnv import EBAlfEnv
import sys
import random

try:
    print("Initializing EB-ALFRED Environment...")
    # down_sample_ratio=0.01 to load very few episodes if possible, or just 1
    env = EBAlfEnv(eval_set='base', down_sample_ratio=0.01) 
    obs = env.reset()
    print("Environment reset successful.")
    print(f"Available actions: {len(env.language_skill_set)}")
    
    # Take a few random steps
    for i in range(5):
        # Sample a valid action index
        action_id = random.randint(0, len(env.language_skill_set) - 1)
        action_str = env.language_skill_set[action_id]
        print(f"Step {i+1}: Taking action {action_id} ({action_str})")
        obs, reward, done, info = env.step(action_id)
        print(f"Transition successful. Done: {done}, Reward: {reward}")
        if done:
            break
            
    env.close()
    print("EB-ALFRED Test PASSED.")
except Exception as e:
    print(f"EB-ALFRED Test FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
