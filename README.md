# LeKiwi Household Platform

A mobile robotics project for household object collection, built on [Hugging Face LeRobot](https://github.com/huggingface/lerobot). The system combines language-guided task planning, camera and LiDAR observations, and learned manipulation skills to find objects and place them in a basket.

## Features

- **Task planning:** a multimodal LLM selects robot actions from natural-language instructions and sensor observations.
- **Perception:** camera overlays show angles and floor distances; LiDAR processing provides obstacle maps and an approximate movement history.
- **Robot control:** base movement, arm positioning, and SmolVLA pick-and-place skills.
- **Data collection:** teleoperation, demonstration recording, and policy evaluation scripts.

## Repository layout

```text
src/
├── main.py                    # Main task entry point
├── agent/                     # LLM planner and robot controller
├── image_feature_extraction/  # Camera overlays
├── lidar/                     # Scan processing and visualisation
├── teleoperate/               # Manual robot control
└── training-a-policy/         # Record, resume recording, and evaluate
```

The scripts require a configured LeKiwi robot, a LeRobot environment, LLM API credentials, and a trained SmolVLA checkpoint. Update robot addresses, serial ports, and model paths for your setup.
