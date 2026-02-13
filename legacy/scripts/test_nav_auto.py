from embodiedbench.envs.eb_navigation.EBNavEnv import EBNavigationEnv
import sys, random
try:
    print("Init EB-Navigation Env...")
    env = EBNavigationEnv(eval_set='base', down_sample_ratio=0.1)
    env.reset()
    for i in range(5):
        action_id = random.randint(0, env.action_space.n - 1)
        action_str = env.language_skill_set[action_id]
        print(f"Action {i}: {action_str}")
        obs, reward, done, info = env.step(action_id, "", 0)
        if done: break
    env.close()
    print("EB-Navigation PASS")
except Exception as e:
    print(f"EB-Navigation FAIL: {e}")
    import traceback; traceback.print_exc()
