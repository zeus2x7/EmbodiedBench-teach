
import os
import sys
import random
import cv2
import numpy as np

sys.path.append(os.getcwd())
from agent_logger import AgentLogger

try:
    from embodiedbench.envs.eb_habitat.EBHabEnv import EBHabEnv
    print("Init EB-Habitat Agent...")
    env = EBHabEnv(eval_set='base', down_sample_ratio=0.1, recording=True)
    logger = AgentLogger("EB-Habitat")

    num_episodes = 2
    max_steps = 20

    for ep in range(num_episodes):
        logger.start_episode()
        obs = env.reset()
        
        # Log initial obs
        logger.log_step(0, obs, "reset", 0.0, False, {"instruction": env.episode_language_instruction})
        
        done = False
        step = 1
        
        while not done and step < max_steps:
            action_id = random.randint(0, env.action_space.n - 1)
            action_str = env.language_skill_set[action_id]
            
            next_obs, reward, done, info = env.step(action_id)
            
            logger.log_step(step, next_obs, action_str, reward, done, info)
            print(f"  Step {step}: action='{action_str}', reward={reward}, done={done}")
            
            step += 1
            
        logger.end_episode()
    
    env.close()
    print("EB-Habitat Agent Completed.")
    
except Exception as e:
    print(f"EB-Habitat Agent Error: {e}")
    import traceback
    traceback.print_exc()
