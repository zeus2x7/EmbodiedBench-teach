# EmbodiedBench Docker Setup & Development Guide

This guide provides a comprehensive walkthrough for setting up, developing, and extending the EmbodiedBench project using Docker.

---

## 🚀 1. Environment Setup

### Prerequisites
- **Host OS**: Linux (Ubuntu 20.04+ recommended).
- **GPU**: NVIDIA GPU with 16GB+ VRAM (for running models like Cosmos-Reason2-8B).
- **Drivers**: NVIDIA Driver 535+ installed on the host.
- **Tools**:
  - [Docker](https://docs.docker.com/engine/install/)
  - [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (Essential for GPU access inside containers).

### Step-by-Step Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd EmbodiedBench
   ```

2. **Configure Environment Variables**:
   Create a `.env` file in the root directory:
   ```bash
   # .env
   HF_TOKEN=your_huggingface_token_here
   DISPLAY=:1
   ```
   *Note: Ensure you have accepted the model license on Hugging Face (e.g., [NVIDIA Cosmos](https://huggingface.co/nvidia/Cosmos-Reason2-8B)).*

3. **Build the Docker Image**:
   ```bash
   docker build -t embodied_bench_v1:latest .
   ```

4. **Run the Complete Benchmark**:
   ```bash
   bash run_benchmark_docker.sh
   ```
   This script starts a container, runs the model on all four environments, and saves logs to `complete_cosmos_outputs_for_2episodes/`.

---

## 🛠 2. Development & Debugging inside Docker

### Changing Code
The Docker setup uses **Volume Mounting**. This means the code on your host machine is mapped directly to `/app` inside the container.
- **Edit code on your host** using your favorite IDE (VS Code, Vim, etc.).
- The changes are **instantly reflected** inside the container.
- Because the packages are installed in "editable" mode (`pip install -e .`), you don't need to rebuild the image for most Python changes.

### Running an Interactive Session
To debug manually or run specific scripts:
```bash
docker run --rm -it \
    --gpus all \
    --env-file .env \
    -v $(pwd):/app \
    --net host \
    embodied_bench_v1:latest \
    bash
```

### Debugging with Test Scripts
You can run minimal test scripts to verify specific environments:
```bash
# Test ALFRED (AI2-THOR)
conda run -n embench python test_alfred_minimal.py

# Test Manipulation (PyRep/CoppeliaSim)
conda run -n embench_man python test_man_minimal.py
```

---

## 🤖 3. Testing Another VLM or AI Model

### Option A: Change the Model Path
If the new model is compatible with the `transformers` / `AutoModelForVision2Seq` API used in our agent:
1. Open `run_benchmark_docker.sh`.
2. Update the `--model_path` argument:
   ```bash
   python cosmos_agent/cosmos_agent.py --model_path "your/new-model-path" ...
   ```

### Option B: Implement a New Model Class
If the model requires a custom inference setup (e.g., proprietary API or different library):
1. Navigate to `cosmos_agent/cosmos_model.py`.
2. Create a new class (e.g., `class GPT4VModel`) following the interface of `CosmosReason2Model`.
3. Update `cosmos_agent/cosmos_agent.py` to instantiate your new model class based on a command-line flag.

---

## 📈 4. Integrating Other Benchmarks

EmbodiedBench is designed to be modular. To add a new benchmark (e.g., "EB-MyNewEnv"):

### 1. Create the Environment Wrapper
Create a new directory `embodiedbench/envs/eb_mynewenv/`.
Implement a Python class that inherits from `gym.Env`. It should provide:
- `reset()`: Returns initial observation.
- `step(action)`: Returns `obs, reward, done, info`.
- `language_skill_set`: A list of strings describing available actions.

### 2. Update the Dockerfile
If the new benchmark requires specific system libraries or a new Conda environment:
```dockerfile
# Add system deps
RUN apt-get install -y lib-required-by-new-env

# Create new conda env
RUN conda create -n embench_new python=3.9 -y
RUN conda run -n embench_new pip install -r requirements_new.txt
```

### 3. Update the Agent
In `cosmos_agent/cosmos_agent.py`:
1. Add a `run_mynewenv` method to the `CosmosAgent` class.
2. Define the system and user prompts for the new environment in `cosmos_agent/prompts.py`.
3. Add the new environment option to the `if __name__ == "__main__":` block.

---

## ❓ Troubleshooting
- **GPU Mismatch**: If you see `NVML: Driver/library version mismatch`, reboot your host machine.
- **X11/Display Issues**: EmbodiedBench uses `Xvfb` for headless rendering. If an environment fails to initialize, check if an X server is already running on `:1` or change the `DISPLAY` variable.
- **Numpy Version**: Several environments are sensitive to numpy versions. Always ensure you are using `numpy==1.23.5` inside the container environments.
