# Cosmos-Reason2-8B Integration Documentation

This document outlines the integration of the **Cosmos-Reason2-8B** model into the **EmbodiedBench** framework. It describes the project structure, new files created, modifications to existing files, and instructions for running and modifying the system.

## 1. Project Structure & Key Files

The integration is primarily contained within the `cosmos_agent/` directory, with supporting scripts in the root directory.

### New/Modified Directories & Files

```
EmbodiedBench/
├── cosmos_agent/                 # [NEW] Core integration module
│   ├── __init__.py
│   ├── cosmos_agent.py           # Main agent logic (loops, logging, execution)
│   ├── cosmos_model.py           # Model wrapper for Cosmos-Reason2-8B
│   └── prompts.py                # Prompt templates for all 4 environments
│
├── cosmos_outputs/               # [NEW] Output directory for logs and frames
│   ├── EB-ALFRED/
│   ├── EB-Habitat/
│   ├── EB-Navigation/
│   └── EB-Manipulation/
│
├── run_alfred_wrapper.sh         # [NEW] Helper script to run EB-ALFRED
├── run_nav_wrapper.sh            # [NEW] Helper script to run EB-Navigation
├── run_man_wrapper.sh            # [NEW] Helper script to run EB-Manipulation
├── run_cosmos_agents.sh          # [NEW] Master script to run all agents
│
└── embodiedbench/                # [EXISTING] Core benchmark code (modified)
    ├── envs/
    │   ├── eb_alfred/            # Modified for X display settings
    │   ├── eb_habitat/           # Modified for X display settings
    │   └── eb_manipulation/      # Modified dataset loading & downsampling
    └── ...
```

### Detailed File Descriptions

#### `cosmos_agent/cosmos_agent.py`
This is the main entry point for the agent. It contains the `CosmosAgent` class which:
- Initializes the model (`CosmosReason2Model`).
- Implements `run_alfred`, `run_habitat`, `run_navigation`, and `run_manipulation` methods.
- Handles the interaction loop:
  1.  Resets the environment.
  2.  Captures observation frames.
  3.  Constructs VLM prompts (using templates from `prompts.py`).
  4.  Calls the model for reasoning and action.
  5.  Parses the model's JSON output.
  6.  Executes the action in the environment.
  7.  Logs the step details (input, output, action, reward) to `cosmos_outputs/`.
- **Key Modification**: Now logs `task_instruction` and `possible_actions` in every step log.

#### `cosmos_agent/cosmos_model.py`
A wrapper around the Hugging Face `transformers` library for the **NVIDIA/Cosmos-Reason2-8B** model.
- Handles model loading (using `bfloat16` and `device_map="auto"`).
- Implements the `respond()` method which formats text/image inputs and generates the model's response.
- Includes logic to parse the `<think>` and `<answer>` tags from the model's output.

#### `cosmos_agent/prompts.py`
Contains the system prompts and user templates for all four environments.
- **ALFRED**: Householding tasks (JSON output).
- **Habitat**: Object rearrangement (JSON output).
- **Navigation**: Object navigation (JSON output).
- **Manipulation**: Tabletop manipulation (Continuous control JSON output).
- **Modifying Prompts**: Edit this file to change the agent's persona, reasoning strategy, or output format.

#### `embodiedbench/envs/...` (Modifications)
- **`eb_alfred/gen/constants.py`**: Updated `X_DISPLAY` to `:1` to match the headless X server.
- **`eb_manipulation/EBManEnv.py`**: Adjusted dataset loading logic to handle potential empty dataset splits and `down_sample_ratio`.

---

## 2. How to Run the System

The system relies on specific Conda environments for each benchmark to avoid dependency conflicts. We have provided wrapper scripts to handle environment activation and display setup.

### Prerequisites
- **X Server**: An X server must be running on display `:1`. (Usually handled by `startx` script or background process).
- **Conda Environments**: `embench`, `embench_nav`, `embench_man` must be created and dependencies installed.

### Running EB-Navigation
```bash
./run_nav_wrapper.sh
```
*   Activates `embench_nav`.
*   Sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (for memory stability).
*   Runs `cosmos_agent/run_cosmos_nav.py`.
*   Logs output to `cosmos_outputs/navigation.log`.

### Running EB-Manipulation
```bash
./run_man_wrapper.sh
```
*   Activates `embench_man`.
*   Sets up CoppeliaSim environment variables.
*   Runs `cosmos_agent/run_cosmos_man.py`.
*   Logs output to `cosmos_outputs/manipulation.log`.

### Running EB-ALFRED
```bash
./run_alfred_wrapper.sh
```
*   Activates `embench` and sets `LD_LIBRARY_PATH`.
*   Runs `cosmos_agent/run_cosmos_alfred.py`.
*   Logs output to `cosmos_outputs/alfred.log`.

### Running EB-Habitat
```bash
./run_habitat_wrapper.sh
```
*   Activates `embench`.
*   Runs `cosmos_agent/run_cosmos_habitat.py`.
*   Logs output to `cosmos_outputs/habitat.log`.

---

## 3. How to Make Changes

### Changing Prompts
1.  Open `cosmos_agent/prompts.py`.
2.  Locate the relevant `{ENV}_SYSTEM_PROMPT` or `{ENV}_USER_TEMPLATE`.
3.  Modify the text. Ensure `{keys}` match the formatting arguments passed in `cosmos_agent.py`.

### Changing Model Logic
1.  Open `cosmos_agent/cosmos_model.py`.
2.  Modify `__init__` to change model loading parameters (e.g., quantization).
3.  Modify `respond` to change generation parameters (e.g., `temperature`, `max_new_tokens`).

### Changing Agent Loop / Logging
1.  Open `cosmos_agent/cosmos_agent.py`.
2.  Go to the specific `run_{env}` method.
3.  Modify the loop to change step limits, logging fields, or action execution logic.
4.  **Logging**: The `step_log` dictionary defines what is saved to JSON.

---

## 4. Output Logs

Logs are saved in `cosmos_outputs/{ENV_NAME}/episode_{N}/cosmos_log.json`.
Each log entry contains:
- `step`: Step number.
- `vlm_input`: The full input to the VLM (system prompt, user prompt, image path, task instruction).
- `vlm_output`: Raw output, parsed thinking, and parsed answer.
- `parsed_action_ids`: Actions extracted from the response.
- `executed_action`: The actual action string sent to the environment.
- `reward`: Reward received.
- `done`: Whether the episode ended.
- `env_feedback`: Feedback/error from the environment.

Frames are saved as `frame_{N}.png` in the same directory.
