import os
import cv2
import json
import numpy as np

class AgentLogger:
    def __init__(self, env_name, output_dir="agent_outputs"):
        self.env_name = env_name
        self.output_dir = os.path.join(output_dir, env_name)
        os.makedirs(self.output_dir, exist_ok=True)
        self.episode_count = 0
        self.current_episode_dir = None
        self.video_writer = None
        self.log_data = []

    def start_episode(self):
        self.episode_count += 1
        self.current_episode_dir = os.path.join(self.output_dir, f"episode_{self.episode_count}")
        os.makedirs(self.current_episode_dir, exist_ok=True)
        self.video_writer = None
        self.log_data = []
        print(f"[{self.env_name}] Starting Episode {self.episode_count}")

    def _extract_frame(self, observation):
        """Extract an RGB image frame from various observation formats."""
        frame = None
        
        if isinstance(observation, dict):
            # Priority order for common image keys
            image_keys = ['rgb', 'head_rgb', 'image', 'frame', 'pixels', 'third_rgb']
            
            # Exact match first
            for key in image_keys:
                if key in observation:
                    val = observation[key]
                    if isinstance(val, np.ndarray) and val.ndim == 3:
                        frame = val
                        break
            
            # Fuzzy match if exact didn't work
            if frame is None:
                for k, v in observation.items():
                    if isinstance(v, np.ndarray) and v.ndim == 3:
                        if any(img_key in k.lower() for img_key in ['rgb', 'image', 'frame', 'pixel']):
                            frame = v
                            break
            
            # Last resort: any 3D array that looks like an image
            if frame is None:
                for k, v in observation.items():
                    if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[2] in [3, 4]:
                        frame = v
                        break
                        
        elif isinstance(observation, np.ndarray) and observation.ndim == 3:
            frame = observation
        
        return frame

    def _summarize_observation(self, observation):
        """Create a JSON-serializable summary of the observation."""
        obs_summary = {}
        
        if isinstance(observation, dict):
            for k, v in observation.items():
                if isinstance(v, np.ndarray):
                    obs_summary[k] = f"ndarray(shape={v.shape}, dtype={v.dtype})"
                elif isinstance(v, (list, tuple)):
                    obs_summary[k] = f"{type(v).__name__}(len={len(v)})"
                elif isinstance(v, (int, float, bool, str)):
                    obs_summary[k] = v
                else:
                    obs_summary[k] = str(type(v).__name__)
        elif isinstance(observation, np.ndarray):
            obs_summary["raw"] = f"ndarray(shape={observation.shape}, dtype={observation.dtype})"
        else:
            obs_summary["raw"] = str(type(observation).__name__)
        
        return obs_summary

    def _normalize_frame(self, frame):
        """Normalize frame to uint8 BGR for cv2."""
        if frame is None:
            return None
            
        # Handle float images
        if frame.dtype in [np.float32, np.float64]:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)
        
        # Handle RGBA
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]
        
        # Convert RGB to BGR for cv2
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame_bgr

    def _make_info_serializable(self, info):
        """Make info dict JSON serializable."""
        if not isinstance(info, dict):
            return str(info)
        
        result = {}
        for k, v in info.items():
            if isinstance(v, np.ndarray):
                result[k] = f"ndarray(shape={v.shape})"
            elif isinstance(v, (np.floating, np.integer)):
                result[k] = float(v)
            elif isinstance(v, (np.bool_,)):
                result[k] = bool(v)
            elif isinstance(v, dict):
                result[k] = self._make_info_serializable(v)
            elif isinstance(v, (list, tuple)):
                result[k] = str(v)[:200]
            elif isinstance(v, (int, float, bool, str)):
                result[k] = v
            else:
                result[k] = str(v)[:200]
        return result

    def log_step(self, step, observation, action, reward, done, info):
        """Log a single environment step: save frame, add to video, record metadata."""
        # Extract and save frame
        frame = self._extract_frame(observation)
        final_frame = self._normalize_frame(frame)
        
        if final_frame is not None:
            # Save individual frame
            frame_path = os.path.join(self.current_episode_dir, f"frame_{step:04d}.png")
            cv2.imwrite(frame_path, final_frame)

            # Initialize Video Writer if needed
            if self.video_writer is None:
                height, width = final_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_path = os.path.join(self.current_episode_dir, "episode.mp4")
                self.video_writer = cv2.VideoWriter(video_path, fourcc, 5.0, (width, height))
            
            self.video_writer.write(final_frame)

        # Log Data
        step_log = {
            "step": step,
            "action": str(action),
            "reward": float(reward) if reward is not None else 0.0,
            "done": bool(done),
            "info": self._make_info_serializable(info),
            "observation_summary": self._summarize_observation(observation),
            "frame_saved": final_frame is not None,
        }
        self.log_data.append(step_log)

    def end_episode(self):
        """Finalize episode: release video writer, save JSON log."""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        # Save JSON logs
        log_path = os.path.join(self.current_episode_dir, "log.json")
        with open(log_path, "w") as f:
            json.dump(self.log_data, f, indent=2, default=str)
        
        # Count frames saved
        frames = [f for f in os.listdir(self.current_episode_dir) if f.startswith("frame_")]
        has_video = os.path.exists(os.path.join(self.current_episode_dir, "episode.mp4"))
        
        print(f"[{self.env_name}] Episode {self.episode_count}: "
              f"{len(self.log_data)} steps, {len(frames)} frames saved, "
              f"video={'yes' if has_video else 'no'}")
        print(f"  -> {self.current_episode_dir}")
