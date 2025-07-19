# VideoStreaming Gym Environment

A standard [OpenAI Gym](https://github.com/openai/gym) environment for simulating adaptive bitrate (ABR) video streaming under diverse network conditions. This environment is designed for reinforcement learning research on video streaming QoE optimization.

## Features
- Simulates real-world network traces (FCC, HSDPA, Oboe)
- Multiple video bitrate levels and buffer dynamics
- Customizable QoE (Quality of Experience) reward
- Compatible with standard RL libraries (Stable Baselines3, RLlib, etc.)

## Installation

### Install via pip (from source)
```bash
git clone https://github.com/tinyzqh/video_streaming_gym.git
cd video_streaming_gym
pip install .
```
或
```bash
pip install git+https://github.com/tinyzqh/video_streaming_gym.git
```

### Requirements
- Python >= 3.10
- gym
- numpy

## Quick Start

```python
import gym
import videostreaming.envs  # Ensure the environment is registered

env = gym.make('VideoStreaming-v0', trace_name='fcc', bandwidth_type='high', qoe_type='normal', seed=42)
obs = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(reward, info)
```

## Directory Structure
```
videostreaming_gym/
├── setup.py
├── README.md
├── videostreaming/
│   ├── __init__.py
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── VideoStreaming.py
│   │   └── utils.py
│   ├── trace/
│   │   ├── fcc/
│   │   ├── hsdpa/
│   │   └── oboe/
│   └── videosize/
│       └── SSB/
└── ...
```

## References
- Oboe: [Akhtar et al., SIGCOMM 2018](https://dl.acm.org/doi/10.1145/3230543.3230558)
- FCC: [FCC Broadband America 2016](https://www.fcc.gov/reports-research/reports/measuringbroadband-america/raw-data-measuring-broadband-america-2016)
- 3G/HSDPA: [Riiser et al., MMSys 2013](https://dl.acm.org/doi/abs/10.1145/2483977.2483991)

## Citation
If you use this environment in your research, please cite the original datasets and this repository.

## License
MIT