
# Atari Pong with Deep Q-Learning

[Report](docs/REPORT.md) | [License: MIT](LICENSE)

## 👥 Group and Project Information

- **Group members:** Lorenzo La Rocca
- **Project ID:** 35

## 📝 Project Description

This project implements a Deep Q-Network (DQN) agent to play Atari Pong using PyTorch and Gymnasium. A convolutional neural network processes four stacked grayscale frames to learn a policy through experience replay, epsilon-greedy exploration, and a target network. Additional experiments investigate exploration schedules, Double DQN, and rally-based reward shaping to analyze their effects on learning efficiency, Q-value estimation, and agent performance.

> 📖 **Official Report:** For theoretical details, architecture, experimental analysis, and limitations, see [**REPORT.md**](docs/REPORT.md).

## 🛠 Technical Reproducibility

### 1. Data and Environment Setup

**Prerequisites:** Git, Conda, and a CUDA-enabled GPU (recommended for training).

Clone the repository and create the environment:

```bash
git clone https://github.com/<YOUR-USERNAME>/Atari-Pong-Deep-RL.git
cd Atari-Pong-Deep-RL

conda env create -f environment.yml
conda activate dl-project
```

**Dataset:** No external dataset is required. Training observations and rewards are generated online using the `ALE/Pong-v5` environment provided by Gymnasium and `ale-py`. The experience replay buffer is populated during training.

### 2. Network Training

Run the following commands from the repository root.

**Baseline DQN Training:**

```bash
python -m src.training.train --config experiments/configs/dqn_baseline.yaml
```

**Improved Model Training — Reward Shaping:**

```bash
python -m src.training.train --config experiments/configs/dqn_reward_shaping.yaml
```

**Additional Experiments — Epsilon-Greedy Ablation:**

```bash
python -m src.training.train --config experiments/configs/dqn_exponential.yaml
python -m src.training.train --config experiments/configs/dqn_constant.yaml
```

**Additional Experiment — Double DQN:**

```bash
python -m src.training.train --config experiments/configs/dqn_double.yaml
```

All final experiments use 6,000,000 environment steps. Training hyperparameters and exploration strategies are specified in the corresponding YAML configuration files.

Training checkpoints and logs are stored in separate experiment directories.

**Resume Training:**

```bash
python -m src.training.train \
  --config experiments/configs/dqn_baseline.yaml \
  --resume-from experiments/checkpoints/<RUN_NAME>/dqn_step_2800000.pth
```

### 3. Evaluation

Models are evaluated over 100 episodes using epsilon = 0.01 and seeds from 1000 to 1099.

Replace `<RUN_NAME>` with the corresponding experiment directory.

**Baseline Evaluation:**

```bash
python -m src.evaluation.final_evaluation \
  --checkpoints experiments/checkpoints/<BASELINE_RUN>/dqn_step_6000000.pth \
  --output-dir experiments/results \
  --episodes 100 \
  --epsilon 0.01 \
  --seed-start 1000
```

**Improved Model Evaluation — Reward Shaping:**

```bash
python -m src.evaluation.final_evaluation \
  --checkpoints experiments/checkpoints/<REWARD_SHAPING_RUN>/dqn_step_6000000.pth \
  --output-dir experiments/results \
  --episodes 100 \
  --epsilon 0.01 \
  --seed-start 1000
```

The same evaluation command can be used for the exponential decay, constant epsilon, and Double DQN models by providing their respective checkpoints.

Results are saved as `metrics.csv` and `episode_rewards.csv` in `experiments/results/<RUN_NAME>/`.

### 4. Gameplay Recording

Generate a gameplay video of the trained agent, including Grad-CAM visualizations and Q-value estimates:

```bash
python -m src.evaluation.record_xray_gameplay \
  --checkpoint experiments/checkpoints/<BASELINE_RUN>/dqn_step_6000000.pth \
  --output experiments/results/<BASELINE_RUN>/gameplay.mp4 \
  --seed 1000 \
  --gradcam-layer features.0 \
  --gradcam-frequency 5 \
  --max-steps 4000
```

The generated video is saved in MP4 format and displays the agent's gameplay alongside the neural network's visual attention and action-value estimates.

### 5. Q-Value Estimation Analysis

To analyze Q-value estimation errors for DQN and Double DQN:

```bash
python -m src.evaluation.analyze_overestimation \
  --checkpoints \
    experiments/checkpoints/<BASELINE_RUN>/dqn_step_6000000.pth \
    experiments/checkpoints/<DOUBLE_DQN_RUN>/dqn_step_6000000.pth \
  --output-dir experiments/results \
  --episodes 100 \
  --epsilon 0.01 \
  --gamma 0.99 \
  --seed-start 1000
```

This analysis compares predicted Q-values with observed discounted returns to investigate differences in value estimation between DQN and Double DQN.

Detailed experimental results and discussions are provided in the official report.

```**

---

*For the declaration of individual tasks and the use of AI, refer to `docs/REPORT.md`.*
