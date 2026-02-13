"""
Cosmos-Reason2-8B Agent for EmbodiedBench.

Runs a single task in each of the 4 environments, logging:
- The full prompt/input the VLM receives
- The raw output text
- The reasoning (chain-of-thought)
- The parsed action
- Observation frames at each step
"""

import os
import sys
import json
import time
import cv2
import numpy as np
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cosmos_agent.prompts import (
    ALFRED_SYSTEM_PROMPT, ALFRED_USER_TEMPLATE, 
    HABITAT_SYSTEM_PROMPT, HABITAT_USER_TEMPLATE,
    NAVIGATION_SYSTEM_PROMPT, NAVIGATION_USER_TEMPLATE,
    MANIPULATION_SYSTEM_PROMPT, MANIPULATION_USER_TEMPLATE,
    REASONING_FORMAT, MANIPULATION_REASONING_FORMAT, get_action_list_str, format_history,
)
from cosmos_agent.cosmos_model import CosmosReason2Model

def fix_json(text):
    """Try to extract and fix JSON from model output."""
    # Try to find JSON block
    json_match = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
    if json_match:
        text = json_match.group(1).strip()
    else:
        # If no code block, try to find first { or [ and last } or ]
        match_obj = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
        if match_obj:
            text = match_obj.group(1).strip()
    
    # Remove trailing commas before } or ]
    text = re.sub(r',\s*([}\]])', r'\1', text)
    
    return text

def parse_action_from_json(text, max_action_id):
    """Parse action id(s) from JSON output."""
    try:
        cleaned = fix_json(text)
        data = json.loads(cleaned)
        
        # Handle { "executable_plan": [...] }
        if isinstance(data, dict) and "executable_plan" in data:
            steps = data["executable_plan"]
        # Handle [...] directly
        elif isinstance(data, list):
            steps = data
        else:
            return [-1]
            
        actions = []
        for step in steps:
            if isinstance(step, dict) and "action_id" in step:
                try:
                    aid = int(step["action_id"])
                    if 0 <= aid <= max_action_id:
                        actions.append(aid)
                except (ValueError, TypeError):
                    continue
        return actions if actions else [-1]
    except Exception as e:
        print(f"JSON parse error: {e}")
        return [-1]


def parse_raw_action_ids(text, max_action_id):
    """Fallback: Parse action id(s) from plain text/comma-separated list."""
    # Find all integers
    ids = re.findall(r'\b\d+\b', text)
    valid_ids = []
    for s in ids:
        try:
            aid = int(s)
            if 0 <= aid <= max_action_id:
                valid_ids.append(aid)
        except (ValueError, TypeError):
            continue
    return valid_ids if valid_ids else [-1]

def parse_action_from_response(answer, action_content, max_action_id):
    """Parse action id(s) from either JSON answer or direct action_content."""
    # 1. Try JSON in answer
    ids = parse_action_from_json(answer, max_action_id)
    if ids != [-1]:
        return ids
    
    # 2. Try raw action_content
    if action_content:
        ids = parse_raw_action_ids(action_content, max_action_id)
        if ids != [-1]:
            return ids
            
    return [-1]


class CosmosAgent:
    """Agent that uses Cosmos-Reason2-8B to interact with EmbodiedBench environments."""
    
    def __init__(self, model):
        self.model = model
        self.step_logs = []
    
    def save_frame(self, obs, step, episode_dir, env_type="alfred"):
        """Extract and save frame from observation."""
        frame = None
        if isinstance(obs, dict):
            for key in ['head_rgb', 'rgb', 'image', 'front_rgb', 'left_shoulder_rgb']:
                if key in obs and isinstance(obs[key], np.ndarray) and obs[key].ndim == 3:
                    frame = obs[key]
                    break
        elif isinstance(obs, np.ndarray) and obs.ndim == 3:
            frame = obs

        if frame is not None:
            if frame.dtype in [np.float32, np.float64]:
                frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8) if frame.max() <= 1 else frame.astype(np.uint8)
            
            path = os.path.join(episode_dir, f"frame_{step:04d}.png")
            cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            return path
        return None

    def _save_logs(self, logs, path):
        with open(path, "w") as f:
            json.dump(logs, f, indent=2, default=str)

    def _init_video_writer(self, video_path, frame):
        """Initialize video writer based on frame size."""
        if frame is None:
            return None
        height, width = frame.shape[:2]
        # Use mp4v codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(video_path, fourcc, 5.0, (width, height)) # 5 fps

    def _write_frame_to_video(self, video_writer, frame):
        """Write frame to video writer if active."""
        if video_writer is not None and frame is not None:
             # Convert RGB to BGR for cv2
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(bgr_frame)

    def _get_frame_from_obs(self, obs):
        """Extract frame from observation for video."""
        frame = None
        if isinstance(obs, dict):
            for key in ['head_rgb', 'rgb', 'image', 'front_rgb', 'left_shoulder_rgb']:
                if key in obs and isinstance(obs[key], np.ndarray) and obs[key].ndim == 3:
                    frame = obs[key]
                    break
        elif isinstance(obs, np.ndarray) and obs.ndim == 3:
            frame = obs

        if frame is not None:
            if frame.dtype in [np.float32, np.float64]:
                frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8) if frame.max() <= 1 else frame.astype(np.uint8)
        return frame

    # ===============================================
    # EB-ALFRED
    # ===============================================
    def run_alfred(self, output_dir="cosmos_outputs/EB-ALFRED", num_episodes=2, start_episode=0):
        from embodiedbench.envs.eb_alfred.EBAlfEnv import EBAlfEnv
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine range
        if num_episodes == -1:
            num_episodes = 1000 # Cap at reasonable max or let it fail
        
        end_episode = start_episode + num_episodes
        
        for episode_idx in range(start_episode, end_episode):
            try:
                # Init env per episode to avoid memory leaks/stale state
                # selected_indexes takes 0-based index
                env = EBAlfEnv(eval_set='base', down_sample_ratio=1.0, selected_indexes=[episode_idx])
            except IndexError:
                print(f"Episode {episode_idx} not found in dataset. Stopping.")
                break
            except Exception as e:
                print(f"Error initializing episode {episode_idx}: {e}")
                continue

            ep_dir = os.path.join(output_dir, f"episode_{episode_idx + 1}")
            os.makedirs(ep_dir, exist_ok=True)
            
            print(f"\n--- Starting Episode {episode_idx} ---", flush=True)
            print("Resetting environment...", flush=True)
            obs = env.reset()
            print("Environment reset successful.", flush=True)
            
            instruction = env.episode_language_instruction
            actions = env.language_skill_set
            action_list_str = get_action_list_str(actions)
            max_action_id = len(actions) - 1
            
            system_prompt = ALFRED_SYSTEM_PROMPT.format(reasoning_format=REASONING_FORMAT)
            
            print(f"Task: {instruction}", flush=True)
            print(f"Num actions: {len(actions)}", flush=True)
            
            step_logs = []
            act_history = []
            done = False
            step = 0
            
            # Save initial frame
            frame_path = self.save_frame(obs, step, ep_dir)
            print(f"Initial frame saved to {frame_path}", flush=True)
            
            # Init video
            video_path = os.path.join(ep_dir, "video.mp4")
            video_frame = self._get_frame_from_obs(obs)
            video_writer = self._init_video_writer(video_path, video_frame)
            self._write_frame_to_video(video_writer, video_frame)
            
            while not done and step < 50:
                history_section = ""
                if act_history:
                    hist_str = format_history(act_history, actions)
                    history_section = f"## Previous action history:\n{hist_str}\n\nReflect on the history and decide next action(s)."
                
                user_text = ALFRED_USER_TEMPLATE.format(
                    max_action_id=max_action_id,
                    action_list=action_list_str,
                    instruction=instruction,
                    history_section=history_section,
                )
                
                print(f"\n--- Step {step+1} ---", flush=True)
                print(f"Calling VLM...", flush=True)
                
                # Call model
                image_paths = [frame_path] if frame_path else None
                raw_output, thinking, answer, action_content = self.model.respond(system_prompt, user_text, image_paths)
                print(f"VLM responded.", flush=True)
                
                # Parse action
                action_ids = parse_action_from_response(answer, action_content, max_action_id)
                print(f"Parsed action ids: {action_ids}", flush=True)
                
                # Log
                step_log = {
                    "step": step + 1,
                    "vlm_input": {
                        "system_prompt": system_prompt[:200] + "...",
                        "user_prompt": user_text,
                        "image": frame_path,
                        "task_instruction": instruction,
                        "possible_actions": actions
                    },
                    "vlm_output": {
                        "raw": raw_output,
                        "thinking": thinking,
                        "answer": answer,
                        "action_content": action_content,
                    },
                    "parsed_action_ids": action_ids,
                }
                # Save logs incrementally
                self._save_logs(step_logs + [step_log], os.path.join(ep_dir, "cosmos_log.json"))
                
                # Execute action(s)
                for aid in action_ids:
                    if aid < 0:
                        step_log["env_feedback"] = "Invalid action from model"
                        print(f"  Invalid action parsed")
                        break
                        
                    action_str = actions[aid]
                    obs, reward, done, info = env.step(aid)
                    frame_path = self.save_frame(obs, step+1, ep_dir)
                    
                    # Write to video
                    video_frame = self._get_frame_from_obs(obs)
                    self._write_frame_to_video(video_writer, video_frame)
                    
                    act_history.append([aid, info.get('env_feedback', '')])
                    
                    step_log["executed_action"] = action_str
                    step_log["reward"] = float(reward)
                    step_log["done"] = bool(done)
                    step_log["env_feedback"] = info.get('env_feedback', '')
                    step_log["info"] = info
                    
                    print(f"  Action: {action_str}")
                    print(f"  Thinking: {thinking[:150]}...")
                    print(f"  Reward: {reward}, Done: {done}")
                    
                    if done:
                        break
                
                step_logs.append(step_log)
                # Full save at end of step
                self._save_logs(step_logs, os.path.join(ep_dir, "cosmos_log.json"))
                step += 1
            
            if video_writer:
                video_writer.release()
            
            env.close()
            print(f"Episode {episode_idx} complete.")

        print(f"\nEB-ALFRED complete. Logs: {output_dir}")
        return step_logs

    # ===============================================
    # EB-Habitat
    # ===============================================
    def run_habitat(self, output_dir="cosmos_outputs/EB-Habitat", num_episodes=2, start_episode=0):
        from embodiedbench.envs.eb_habitat.EBHabEnv import EBHabEnv
        
        os.makedirs(output_dir, exist_ok=True)
        
        if num_episodes == -1:
             num_episodes = 1000

        end_episode = start_episode + num_episodes
        
        for episode_idx in range(start_episode, end_episode):
            try:
                # Init env per episode
                # start_epi_index will skip to the correct episode
                env = EBHabEnv(eval_set='base', down_sample_ratio=1.0, start_epi_index=episode_idx)
                # Check if we actually got an episode or if start_epi_index exhausted the dataset
                if env._current_episode_num > env.number_of_episodes:
                     print(f"Episode {episode_idx} out of range (max {env.number_of_episodes}). Stopping.")
                     break
            except Exception as e:
                print(f"Error initializing episode {episode_idx}: {e}")
                break

            ep_dir = os.path.join(output_dir, f"episode_{episode_idx + 1}")
            os.makedirs(ep_dir, exist_ok=True)
            
            print(f"\n--- Starting Episode {episode_idx} ---")
            obs = env.reset()
            
            instruction = env.episode_language_instruction
            actions = env.language_skill_set
            action_list_str = get_action_list_str(actions)
            max_action_id = len(actions) - 1
            
            system_prompt = HABITAT_SYSTEM_PROMPT.format(reasoning_format=REASONING_FORMAT)
            
            print(f"Task: {instruction}")
            print(f"Num actions: {len(actions)}")
            
            step_logs = []
            act_history = []
            done = False
            step = 0
            
            frame_path = self.save_frame(obs, step, ep_dir)
            
            # Init video
            video_path = os.path.join(ep_dir, "video.mp4")
            video_frame = self._get_frame_from_obs(obs)
            video_writer = self._init_video_writer(video_path, video_frame)
            self._write_frame_to_video(video_writer, video_frame)
            
            while not done and step < 50:
                history_section = ""
                if act_history:
                    hist_str = format_history(act_history, actions)
                    history_section = f"## Previous action history:\n{hist_str}\n\nReflect on the history and decide next action(s)."
                
                user_text = HABITAT_USER_TEMPLATE.format(
                    max_action_id=max_action_id,
                    action_list=action_list_str,
                    instruction=instruction,
                    history_section=history_section,
                )
                
                print(f"\n--- Step {step+1} ---")
                
                image_paths = [frame_path] if frame_path else None
                raw_output, thinking, answer, action_content = self.model.respond(system_prompt, user_text, image_paths)
                
                print(f"VLM Answer: {answer}", flush=True)
                action_ids = parse_action_from_response(answer, action_content, max_action_id)
                print(f"Parsed action ids: {action_ids}", flush=True)
                
                step_log = {
                    "step": step + 1,
                    "vlm_input": {
                        "system_prompt": system_prompt[:200] + "...", 
                        "user_prompt": user_text, 
                        "image": frame_path,
                        "task_instruction": instruction,
                        "possible_actions": actions
                    },
                    "vlm_output": {
                        "raw": raw_output,
                        "thinking": thinking,
                        "answer": answer,
                        "action_content": action_content
                    },
                    "parsed_action_ids": action_ids,
                }
                
                # Save logs incrementally
                self._save_logs(step_logs + [step_log], os.path.join(ep_dir, "cosmos_log.json"))
                
                for aid in action_ids:
                    if aid < 0:
                        step_log["env_feedback"] = "Invalid action from model"
                        print(f"  Invalid action parsed")
                        break
                    
                    action_str = actions[aid]
                    obs, reward, done, info = env.step(aid)
                    frame_path = self.save_frame(obs, step+1, ep_dir)
                    
                    # Write to video
                    video_frame = self._get_frame_from_obs(obs)
                    self._write_frame_to_video(video_writer, video_frame)
                    
                    act_history.append([aid, info.get('env_feedback', '')])
                    step_log["executed_action"] = action_str
                    step_log["reward"] = float(reward)
                    step_log["done"] = bool(done)
                    step_log["env_feedback"] = info.get('env_feedback', '')
                    step_log["info"] = info
                    
                    print(f"  Action: {action_str}")
                    print(f"  Thinking: {thinking[:150]}...")
                    print(f"  Reward: {reward}, Done: {done}")
                    if done:
                        break
                
                step_logs.append(step_log)
                self._save_logs(step_logs, os.path.join(ep_dir, "cosmos_log.json"))
                step += 1 # end of step loop
            
            if video_writer:
                video_writer.release()
            
            env.close()
            print(f"Episode {episode_idx} complete.")

        print(f"\nEB-Habitat complete. Logs: {output_dir}")
        return step_logs

    # ===============================================
    # EB-Navigation
    # ===============================================
    def run_navigation(self, output_dir="cosmos_outputs/EB-Navigation", num_episodes=2, start_episode=0):
        from embodiedbench.envs.eb_navigation.EBNavEnv import EBNavigationEnv
        
        os.makedirs(output_dir, exist_ok=True)
        
        if num_episodes == -1:
            num_episodes = 1000

        end_episode = start_episode + num_episodes
        
        for episode_idx in range(start_episode, end_episode):
            try: 
                 # Init env per episode
                 env = EBNavigationEnv(eval_set='base', down_sample_ratio=1.0, selected_indexes=[episode_idx])
            except IndexError:
                print(f"Episode {episode_idx} not found in dataset. Stopping.")
                break
            except Exception as e:
                print(f"Error initializing episode {episode_idx}: {e}")
                continue

            ep_dir = os.path.join(output_dir, f"episode_{episode_idx + 1}")
            os.makedirs(ep_dir, exist_ok=True)
            
            print(f"\n--- Starting Episode {episode_idx} ---")
            obs = env.reset()
            
            instruction = env.episode_language_instruction
            actions = env.language_skill_set
            action_list_str = get_action_list_str(actions)
            max_action_id = len(actions) - 1
            
            system_prompt = NAVIGATION_SYSTEM_PROMPT.format(reasoning_format=REASONING_FORMAT)
            
            print(f"Task: {instruction}")
            print(f"Num actions: {len(actions)}")
            
            step_logs = []
            act_history = []
            done = False
            step = 0
            
            frame_path = self.save_frame(obs, step, ep_dir)
            
            # Init video
            video_path = os.path.join(ep_dir, "video.mp4")
            video_frame = self._get_frame_from_obs(obs)
            video_writer = self._init_video_writer(video_path, video_frame)
            self._write_frame_to_video(video_writer, video_frame)
            
            while not done and step < 50:
                history_section = ""
                if act_history:
                    hist_str = format_history(act_history, actions)
                    history_section = f"## Previous action history:\n{hist_str}"
                
                user_text = NAVIGATION_USER_TEMPLATE.format(
                    max_action_id=max_action_id,
                    action_list=action_list_str,
                    instruction=instruction,
                    history_section=history_section,
                )
                
                print(f"\n--- Step {step+1} ---")
                
                image_paths = [frame_path] if frame_path else None
                raw_output, thinking, answer, action_content = self.model.respond(system_prompt, user_text, image_paths)
                
                print(f"VLM Answer: {answer}", flush=True)
                action_ids = parse_action_from_response(answer, action_content, max_action_id)
                print(f"Parsed action ids: {action_ids}", flush=True)
                
                step_log = {
                    "step": step + 1,
                    "vlm_input": {
                        "system_prompt": system_prompt[:200] + "...", 
                        "user_prompt": user_text, 
                        "image": frame_path,
                        "task_instruction": instruction,
                        "possible_actions": actions
                    },
                    "vlm_output": {
                        "raw": raw_output, 
                        "thinking": thinking, 
                        "answer": answer,
                        "action_content": action_content
                    },
                    "parsed_action_ids": action_ids,
                }
                # Save logs incrementally
                self._save_logs(step_logs + [step_log], os.path.join(ep_dir, "cosmos_log.json"))
                
                for aid in action_ids:
                    if aid < 0:
                        step_log["env_feedback"] = "Invalid action from model"
                        print(f"  Invalid action parsed")
                        break

                    action_str = actions[aid]
                    obs, reward, done, info = env.step(aid, reasoning=thinking, i_flag=step)
                    frame_path = self.save_frame(obs, step+1, ep_dir)
                    
                    # Write to video
                    video_frame = self._get_frame_from_obs(obs)
                    self._write_frame_to_video(video_writer, video_frame)
                    
                    act_history.append([aid, info.get('env_feedback', info.get('action_description', ''))])
                    
                    step_log["executed_action"] = action_str
                    step_log["reward"] = float(reward)
                    step_log["done"] = bool(done)
                    step_log["env_feedback"] = str(info) 
                    step_log["info"] = info
                    
                    print(f"  Action: {action_str}")
                    print(f"  Thinking: {thinking[:150]}...")
                    print(f"  Reward: {reward}, Done: {done}")
                    
                    if done:
                        break
                
                step_logs.append(step_log)
                self._save_logs(step_logs, os.path.join(ep_dir, "cosmos_log.json"))
                step += 1
            
            if video_writer:
                video_writer.release()
            
            env.close()
            print(f"Episode {episode_idx} complete.")

        print(f"\nEB-Navigation complete. Logs: {output_dir}")
        return step_logs

    # ===============================================
    # EB-Manipulation
    # ===============================================
    def run_manipulation(self, output_dir="cosmos_outputs/EB-Manipulation", num_episodes=2, start_episode=0):
        from embodiedbench.envs.eb_manipulation.EBManEnv import EBManEnv
        import numpy as np
        import cv2
        
        os.makedirs(output_dir, exist_ok=True)
        
        if num_episodes == -1:
             num_episodes = 1000

        end_episode = start_episode + num_episodes
        
        for episode_idx in range(start_episode, end_episode):
            try:
                # Init env per episode
                env = EBManEnv(eval_set='base', down_sample_ratio=1.0, render_mode='rgb_array', selected_indexes=[episode_idx])
            except IndexError:
                print(f"Episode {episode_idx} not found in dataset. Stopping.")
                break
            except Exception as e:
                print(f"Error initializing episode {episode_idx}: {e}")
                continue

            ep_dir = os.path.join(output_dir, f"episode_{episode_idx + 1}")
            os.makedirs(ep_dir, exist_ok=True)
            
            print(f"\n--- Starting Episode {episode_idx} ---")
            obs = env.reset()
            
            instruction = getattr(env, 'episode_language_instruction', 'Complete the manipulation task')
            
            # Get manipulation-specific params
            max_coord = 100
            max_rot = 100
            rot_degrees = 3.6
            
            system_prompt = MANIPULATION_SYSTEM_PROMPT.format(
                reasoning_format=MANIPULATION_REASONING_FORMAT,
                max_coord=max_coord,
                max_rot=max_rot,
                rot_degrees=rot_degrees,
            )
            
            print(f"Task: {instruction}")
            print(f"Action space: {env.action_space}")
            
            step_logs = []
            done = False
            step = 0
            
            frame_path = self.save_frame(obs, step, ep_dir, env_type="manipulation")
            
            # Init video
            video_writer = None
            video_path = os.path.join(ep_dir, "video.mp4")
            video_frame = self._get_frame_from_obs(obs)
            video_writer = self._init_video_writer(video_path, video_frame)
            self._write_frame_to_video(video_writer, video_frame)
            
            while not done and step < 50:
                # Get object info if available
                object_info = ""
                if isinstance(obs, dict) and 'object_informations' in obs:
                    object_info = str(obs['object_informations'])
                
                user_text = MANIPULATION_USER_TEMPLATE.format(
                    instruction=instruction,
                    object_info=object_info if object_info else "See images for object positions",
                    history_section="",
                )
                
                print(f"\n--- Step {step+1} ---")
                
                image_paths = [frame_path] if frame_path else None
                raw_output, thinking, answer, action_content = self.model.respond(system_prompt, user_text, image_paths)
                
                print(f"VLM Answer: {answer}", flush=True)
                
                # Parse actions
                parsed_actions = []
                try:
                    cleaned = fix_json(answer)
                    data = json.loads(cleaned)
                    if "executable_plan" in data and isinstance(data["executable_plan"], list):
                        for plan_step in data["executable_plan"]:
                            act_data = plan_step.get("action", None)
                            if act_data and len(act_data) >= 7:
                                action = np.array(act_data[:8], dtype=np.float32)
                                action = np.clip(action, env.action_space.low, env.action_space.high)
                                parsed_actions.append(action)
                except Exception:
                    pass

                # Fallback to single random action if none parsed
                if not parsed_actions:
                    parsed_actions = [env.action_space.sample()]

                step_log = {
                    "step": step + 1,
                    "vlm_input": {
                        "system_prompt": system_prompt[:200] + "...", 
                        "user_prompt": user_text, 
                        "image": frame_path,
                        "task_instruction": instruction,
                        "possible_actions": "Continuous control (7 DOF + gripper)"
                    },
                    "vlm_output": {
                        "raw": raw_output, 
                        "thinking": thinking, 
                        "answer": answer,
                        "action_content": action_content
                    },
                    "parsed_actions_cnt": len(parsed_actions),
                }
                # Save logs incrementally
                self._save_logs(step_logs + [step_log], os.path.join(ep_dir, "cosmos_log.json"))
                
                for i, action in enumerate(parsed_actions):
                    step_log["action_used"] = action.tolist() if isinstance(action, np.ndarray) else str(action)
                    
                    obs, reward, done, info = env.step(action)
                    frame_path = self.save_frame(obs, step+1, ep_dir, env_type="manipulation")
                    
                    # Write to video
                    video_frame = self._get_frame_from_obs(obs)
                    self._write_frame_to_video(video_writer, video_frame)
                    
                    step_log["reward"] = float(reward)
                    step_log["done"] = bool(done)
                    step_log["info"] = info
                    step_log["env_feedback"] = info.get('env_feedback', '')
                    
                    print(f"  Action {i+1}/{len(parsed_actions)}")
                    print(f"  Thinking: {thinking[:150]}...")
                    print(f"  Reward: {reward}, Done: {done}")
                    
                    if done:
                        break
                
                step_logs.append(step_log)
                self._save_logs(step_logs, os.path.join(ep_dir, "cosmos_log.json"))
                step += 1
            
            if video_writer:
                video_writer.release()
            
            env.close()
            print(f"Episode {episode_idx} complete.")

        print(f"\nEB-Manipulation complete. Logs: {output_dir}")
        return step_logs

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Cosmos Agent on EmbodiedBench environments")
    parser.add_argument("--env", type=str, choices=["alfred", "habitat", "navigation", "manipulation", "all"], default="all", help="Environment to run")
    parser.add_argument("--model_path", type=str, default="nvidia/Cosmos-Reason2-8B", help="Path to Cosmos model")
    parser.add_argument("--num_episodes", type=int, default=2, help="Number of episodes to run (-1 for all)")
    parser.add_argument("--start_episode", type=int, default=0, help="Episode index to start from")
    parser.add_argument("--output_dir", type=str, default=None, help="Base directory for log outputs")
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_path}...")
    try:
        model = CosmosReason2Model(model_name=args.model_path)
        agent = CosmosAgent(model)
    except OSError as e:
        if "gated repo" in str(e) or "401 Client Error" in str(e):
            print("\n" + "="*60)
            print("ERROR: Authentication failed for Cosmos-Reason2-8B.")
            print("To fix this:")
            print("1. Get your token from https://huggingface.co/settings/tokens")
            print("2. Accept the model license at https://huggingface.co/nvidia/Cosmos-Reason2-8B")
            print("3. Export your token:")
            print("   export HF_TOKEN=hf_xxxxxxxxxxxxxx")
            print("   Or add it to .env file if using the wrapper scripts.")
            print("="*60 + "\n")
            import sys
            sys.exit(1)
        else:
            raise e
    
    if args.env == "alfred" or args.env == "all":
        try:
            kwargs = {"num_episodes": args.num_episodes, "start_episode": args.start_episode}
            if args.output_dir:
                kwargs["output_dir"] = os.path.join(args.output_dir, "EB-ALFRED")
            agent.run_alfred(**kwargs)
        except Exception as e:
            print(f"Error running ALFRED: {e}")
            import traceback
            traceback.print_exc()

    if args.env == "habitat" or args.env == "all":
        try:
            kwargs = {"num_episodes": args.num_episodes, "start_episode": args.start_episode}
            if args.output_dir:
                kwargs["output_dir"] = os.path.join(args.output_dir, "EB-Habitat")
            agent.run_habitat(**kwargs)
        except Exception as e:
            print(f"Error running Habitat: {e}")
            import traceback
            traceback.print_exc()

    if args.env == "navigation" or args.env == "all":
        try:
            kwargs = {"num_episodes": args.num_episodes, "start_episode": args.start_episode}
            if args.output_dir:
                kwargs["output_dir"] = os.path.join(args.output_dir, "EB-Navigation")
            agent.run_navigation(**kwargs)
        except Exception as e:
            print(f"Error running Navigation: {e}")
            import traceback
            traceback.print_exc()

    if args.env == "manipulation" or args.env == "all":
        try:
            kwargs = {"num_episodes": args.num_episodes, "start_episode": args.start_episode}
            if args.output_dir:
                kwargs["output_dir"] = os.path.join(args.output_dir, "EB-Manipulation")
            agent.run_manipulation(**kwargs)
        except Exception as e:
            print(f"Error running Manipulation: {e}")
            import traceback
            traceback.print_exc()

