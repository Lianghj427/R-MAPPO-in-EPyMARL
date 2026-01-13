# R-MAPPO in EPyMARL

This project faithfully implements the **R-MAPPO (Recurrent Multi-Agent Proximal Policy Optimization)** algorithm within the [EPyMARL](https://github.com/uoe-agents/epymarl) (Extended PyMARL) framework.

The core contribution of this repository is the porting of the critical **Data Chunking (Recurrent Mini-batching)** mechanism from the [Official MAPPO (on-policy)](https://github.com/marlbenchmark/on-policy) codebase. This addresses the fundamental challenge in standard PyMARL PPO implementations: balancing the IID requirement of PPO with the temporal continuity required by RNNs.

## 🌟 Key Features

Compared to other PPO implementations in PyMARL, this project includes the following key features:

### 1. Recurrent Data Chunking (Core Mechanism)
* **Exact Implementation**: Implements the exact data chunking logic from the MAPPO paper. Full episodes are sliced into fixed-length **Chunks**.
* **Hidden State Passing**: Stores hidden states during the rollout phase and correctly initializes the RNN with the saved states at the start of each chunk during training. This ensures accurate Truncated BPTT.
* **Chunk Shuffling**: Shuffles data at the "chunk" level (instead of the step or episode level), achieving a balance between random sampling and temporal dependency.

### 2. R-Centralized Critic
* Introduces a GRU-based Centralized Critic (`RCentralVCritic`), enabling the critic to handle partial observability effectively.
* The critic training also follows the Chunking mechanism with dynamic hidden state passing.

### 3. Value Normalization & PopArt
* Integrates `RunningMeanStd` to normalize Value Targets, significantly improving training stability.

### 4. Correct GAE & Masking
* Fixes dimension broadcasting issues in PyMARL's GAE calculation logic.
* Strictly applies masks during the calculation of Policy Loss, Value Loss, and Entropy, excluding padding data from optimization.

## 📂 File Structure

Key modified and added files:

* `src/learners/rmappo_learner.py`: **The Core Learner**. Implements the Data Chunking generator (`_generate_data_chunks`) and the PPO update loop with hidden state passing (`_ppo_update`).
* `src/modules/critics/r_centralV.py`: A new Centralized Critic with RNN support.
* `src/config/algs/mappo.yaml`: Complete hyperparameter configuration aligned with the official MAPPO paper (e.g., `grad_norm_clip=10`, `use_valuenorm=True`).
* `src/components/standarize_stream.py`: Fixed and enhanced `RunningMeanStd` to support `denormalize` operations.

## 🚀 Quick Start

### 1. Installation

This project is based on EPyMARL. Please ensure you have StarCraft II and the SMAC/SMACv2 environment installed.

```bash
# Install dependencies (based on epymarl requirements)
pip install -r requirements.txt
```

### 2. Training
Run experiments using the mappo configuration. Below is an example running on the SMAC map `3s5z`:

```bash
python3 src/main.py --config=mappo --env-config=sc2 with env_args.map_name=3s5z
```

You can override parameters like chunk length or batch size directly from the command line:

```bash
python3 src/main.py --config=mappo --env-config=sc2 with env_args.map_name=3s5z data_chunk_length=10 mini_batch_size=32
```

## ⚙️ Key Hyperparameters

You can find these settings in `src/config/algs/mappo.yaml`:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `data_chunk_length` | 10 | The length of sequence chunks fed into the RNN. |
| `mini_batch_size` | 32 | The mini-batch size for PPO updates (in terms of chunks). |
| `use_valuenorm` | True | Whether to use Value Normalization. |
| `grad_norm_clip` | 10.0 | Gradient clipping threshold (MAPPO suggests a larger value). |
| `entropy_coef` | 0.01 | Coefficient for entropy regularization. |
| `epoch` | 15 | Number of PPO update epochs per rollout. |

## 📚 Citation

If you find this code useful, please cite the original MAPPO paper and the EPyMARL framework:

**MAPPO (The Surprising Effectiveness of PPO in Multi-Agent Games):**
```bibtex
@article{yu2022surprising,
  title={The surprising effectiveness of ppo in multi-agent games},
  author={Yu, Chao and Velu, Akash and Vinitsky, Eugene and Gao, Jiaxuan and Wang, Yu and Bayen, Alexandre and Wu, Yi},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={24611--24624},
  year={2022}
}
```

**EPyMARL (Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks):**
```bibtex
@article{papoudakis2021benchmarking,
  title={Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks},
  author={Papoudakis, Georgios and Christianos, Filippos and Schäfer, Lukas and Albrecht, Stefano V},
  journal={arXiv preprint arXiv:2006.07869},
  year={2021}
}
```

## 📝 License
This project is licensed under the Apache License 2.0