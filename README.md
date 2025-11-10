# OptiVerse: Optimization Universe Gym Environments

A collection of Gymnasium environments for engineering optimization problems, designed for reinforcement learning (RL) research.

The environments cover multiple domains such as video streaming, energy optimization, traffic control, and general control systems, making OptiVerse a unified platform for RL in real-world engineering tasks.



## ✨ Features
- Video Streaming (ABR): Adaptive bitrate streaming simulation under diverse network traces (FCC, HSDPA, Oboe).
- TODO: Energy Optimization: Smart grid, demand response, and energy scheduling tasks.
- TODO: Traffic Control & Routing: Traffic light control, routing optimization.
- TODO: Control Systems: Extended pendulum, quadrotor, and other benchmark dynamics.


## 🔧 Installation

### Install from source
```bash
git clone git@github.com:tinyzqh/OptiVerse.git
cd OptiVerse
pip install .
```
### Or install directly from GitHub
```bash
pip install git+https://github.com/tinyzqh/OptiVerse.git
```

### Requirements
- Python >= 3.10
- gymnasium
- numpy

## 🚀 Quick Start

```python
import OptiVerse   # Ensure all environments are registered
import gymnasium as gym

env = gym.make(
    "VideoStreaming-v0",
    trace_name="fcc",
    bandwidth_type="high",
    qoe_type="normal",
    seed=42
)

obs, info = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(reward, info)
    if terminated or truncated:
        done = True

env.close()
```


## Directory Structure
```
OptiVerse/
├── setup.py
├── README.md
├── OptiVerse/
│   ├── __init__.py
│   ├── registry.py
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── video/
│   │   │   ├── __init__.py
│   │   │   └── VideoStreaming.py
│   │   ├── energy/
│   │   ├── traffic/
│   │   └── control/
│   ├── datasets/
│   │   ├── video/
│   │   │   ├── trace/
│   │   │   │   ├── fcc/
│   │   │   │   ├── hsdpa/
│   │   │   │   └── oboe/
│   │   │   └── videosize/
│   │   │       └── SSB/
│   │   ├── energy/
│   │   ├── traffic/
│   │   └── control/
├── tests/
│   └── test_videostreaming.py
└── examples/
    └── run_videostreaming.py
```

## 📖 References
- Oboe: [Akhtar et al., SIGCOMM 2018](https://dl.acm.org/doi/10.1145/3230543.3230558)
- FCC: [FCC Broadband America 2016](https://www.fcc.gov/reports-research/reports/measuringbroadband-america/raw-data-measuring-broadband-america-2016)
- 3G/HSDPA: [Riiser et al., MMSys 2013](https://dl.acm.org/doi/abs/10.1145/2483977.2483991)

- [option-critic-arch](https://github.com/alversafa/option-critic-arch), [The Option-Critic Architecture](https://arxiv.org/pdf/1609.05140)


# TODO
- [Multi-Agent-Learning-Environments](https://github.com/Bigpig4396/Multi-Agent-Reinforcement-Learning-Environment)
- [parametrized action-space environments](https://github.com/thomashirtz/gym-hybrid)
- [Multi-Pass Deep Q-Networks](https://github.com/cycraig/MP-DQN)
- [Non-Stationary bandits](https://github.com/dquail/NonStationaryBandit)
- [Agents with Imagination](https://github.com/higgsfield/Imagination-Augmented-Agents)
- [PPO with multi-head/ autoregressive actions](https://github.com/henrycharlesworth/multi_action_head_PPO)

## 📌 Citation
If you use OptiVerse in your research, please cite the original datasets and this repository.

## 📜 License
MIT License