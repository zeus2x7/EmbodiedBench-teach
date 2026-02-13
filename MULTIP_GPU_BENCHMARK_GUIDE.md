# EmbodiedBench: High-Performance Multi-GPU Benchmarking Guide
## (Optimized for 4x NVIDIA RTX 5090 Systems)

This document provides a technical roadmap for scaling EmbodiedBench evaluations to a 4x GPU cluster (specifically targeting high-VRAM setups like the RTX 5090).

---

### 1. Hardware Utilization Strategy

With 4x RTX 5090 GPUs, your primary constraints shift from VRAM to I/O and process management. Each 5090 (assumed 32GB+ VRAM) permits two possible scaling patterns:

*   **Pattern A (High-Fidelity):** 1 Agent + 1 Simulator instance per GPU (Total: 4 concurrent tasks).
*   **Pattern B (High-Throughput):** 1 Agent (quantized) + 4-8 Simulator instances per GPU using a Shared Model Server (Total: 16-32 concurrent tasks).

---

### 2. The Shared Model Server (vLLM recommended)

Instead of loading the Cosmos-Reason2-8B model 4 times (which consumes ~100GB+ VRAM), run a centralized model server.

1.  **Start vLLM Server:**
    ```bash
    # Run on GPU 0-1 (using pipeline parallelism)
    python -m vllm.entrypoints.openai.api_server \
        --model nvidia/Cosmos-Reason2-8B \
        --tensor-parallel-size 2 \
        --port 8000
    ```
2.  **Agent Logic:** Modify `cosmos_agent.py` to use an API client instead of local Weights. This frees up the remaining GPUs (2 and 3) to run purely simulator instances.

---

### 3. Orchestration: The Master Controller

Implement a master script that manages the worker pool.

#### Parallelization Logic (Python):
```python
import multiprocessing as mp
from functools import partial

def run_task_batch(gpu_id, episode_range):
    """
    Worker function executed in a separate process.
    """
    # 1. Set GPU affinity
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 2. Assign unique X-Display for THOR/ALFRED
    display_id = gpu_id + 10  # e.g., :10, :11
    start_xvfb(display_id)
    os.environ["DISPLAY"] = f":{display_id}"

    # 3. Initialize Agent pointing to vLLM or local model
    agent = CosmosAgent(model_url="http://localhost:8000/v1")
    
    # 4. Loop through assigned episodes
    for ep_idx in episode_range:
        agent.run_xxxx(start_episode=ep_idx, num_episodes=1)

if __name__ == "__main__":
    num_gpus = 4
    all_episodes = list(range(1000))  # Total benchmark size
    
    # Split tasks across 4 workers
    chunks = np.array_split(all_episodes, num_gpus)
    
    with mp.Pool(num_gpus) as pool:
        pool.starmap(run_task_batch, zip(range(num_gpus), chunks))
```

---

### 4. Environment Specific Scaling

#### **EB-ALFRED & EB-Navigation (AI2-THOR)**
*   **The Xvfb Bottle-neck:** On a 4-GPU system, create 4 separate Xvfb buffers.
*   **Command:** `Xvfb :10 -screen 0 1024x768x24 &`
*   **GPU Pinning:** AI2-THOR renders on the GPU that the X-server is started on. Use `nvidia-xconfig` to bind X-displays to specific BusIDs.

#### **EB-Habitat**
*   **Multi-Instance:** Habitat is extremely lightweight. On a 5090, you can easily run 10+ Habitat instances per GPU.
*   **Strategy:** Use Pattern B (Shared Model Server) and scale the number of worker processes to match your CPU core count (49).

#### **EB-Manipulation (PyRep)**
*   **Process Isolation:** PyRep/CoppeliaSim requires strict process isolation.
*   **VRAM:** It consumes very little VRAM compared to the VLM. You can over-subscribe these workers (e.g., 2 workers per GPU).

---

### 5. Automated Deployment with Docker

To make migration to the 4x 5090 system seamless, use **Docker Compose**.

1.  **Define a Base Image:** Containing Miniconda + CoppeliaSim + Habitat binaries.
2.  **Service Scaling:**
    ```yaml
    services:
      worker:
        image: embodied_bench_v1
        deploy:
          resources:
            reservations:
              devices:
                - driver: nvidia
                  count: 1  # Each container gets 1 GPU
        command: python cosmos_agent.py --env all
    ```
3.  **Run:** `docker-compose up --scale worker=4`

---

### 6. Logging and Result Merging

In a parallel setup, local JSON logging will create fragmented files.
1.  **Unique Output Directories:** Ensure each worker saves to `outputs/worker_{id}/`.
2.  **Final Merge Script:** Use a Python script to glob all `cosmos_log.json` files and aggregate the total success rate and stats into a single `final_benchmark_report.json`.

---
*Created by Antigravity AI - System Migration Guide v1.0*
