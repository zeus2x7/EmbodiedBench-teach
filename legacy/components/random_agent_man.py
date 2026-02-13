
import os
import sys
import random
import cv2
import numpy as np
import faulthandler
faulthandler.enable()

sys.path.append(os.getcwd())
from agent_logger import AgentLogger

try:
    from embodiedbench.envs.eb_manipulation.EBManEnv import EBManEnv
    print("Init EB-Manipulation Agent...")
    env = EBManEnv(eval_set='base', down_sample_ratio=0.1)
    logger = AgentLogger("EB-Manipulation")

    num_episodes = 2
    max_steps = 20

    for ep in range(num_episodes):
        logger.start_episode()
        obs = env.reset()
        
        logger.log_step(0, obs, "reset", 0.0, False, {"instruction": getattr(env, 'episode_language_instruction', '')})
        
        done = False
        step = 1
        
        while not done and step < max_steps:
            if hasattr(env.action_space, 'n'):
                action = random.randint(0, env.action_space.n - 1)
                action_str = env.language_skill_set[action] if hasattr(env, 'language_skill_set') else str(action)
            else:
                action = env.action_space.sample()
                action_str = f"Box action: {action}"
            
            next_obs, reward, done, info = env.step(action)
            
            logger.log_step(step, next_obs, action_str, reward, done, info)
            print(f"  Step {step}: action='{action_str}', reward={reward}, done={done}")
            
            step += 1
            
        logger.end_episode()

    env.close()
    print("EB-Manipulation Agent Completed.")
    
except Exception as e:
    print(f"EB-Manipulation Agent Error: {e}")
    import traceback
    traceback.print_exc()
