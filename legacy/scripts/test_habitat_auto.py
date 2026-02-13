try:
    from embodiedbench.envs.eb_habitat.EBHabEnv import EBHabEnv
    import sys, random
    print("Init EB-Habitat Env...")
    env = EBHabEnv(eval_set='base', down_sample_ratio=0.01)
    env.reset()
    print("Habitat Reset Done")
    for i in range(5):
        # The code for EBHabEnv might differ, lets assume standard gym + some custom props
        # Checking language skill set
        action_id = random.randint(0, env.action_space.n - 1)
        if hasattr(env, 'language_skill_set'):
             action_str = env.language_skill_set[action_id]
        else:
             action_str = str(action_id)
             
        print(f"Action {i}: {action_str}")
        # Step signature might be standard
        step_result = env.step(action_id)
        # Handle unpacking depending on length
        if len(step_result) == 4:
            obs, reward, done, info = step_result
        else:
            print(f"Unexpected step result length: {len(step_result)}")
            obs, reward, done, info = step_result[0], step_result[1], step_result[2], {}
            
        if done: break
    env.close()
    print("EB-Habitat PASS")
except Exception as e:
    print(f"EB-Habitat FAIL: {e}")
    import traceback; traceback.print_exc()
