
import cv2
import base64
import time
import numpy as np

import subprocess

from matplotlib.figure import Figure
from pathlib import Path
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.tools import StructuredTool, tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from src.agent.agent import LeKiwiAgent
from src.agent.controller import LeKiwiController
from src.lidar.lidar import Lidar
from src.image_feature_extraction.frame_augmentation import FeatureExtractor

from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig

load_dotenv(Path(__file__).with_name(".env"))

def main():

    reset_process = input("Press 1 to open SSH...\n")
    if str(reset_process) == "1":
        perth_timestamp = time.gmtime(time.time() + 8 * 3600)
        print(f"{time.strftime("%Y-%m-%d %H:%M:%S", perth_timestamp)}:  Starting SSH Process")
        print("SSH Steps")
        print("1. Type Password")
        print("2. Enter - conda activate lerobot")
        print("3. Enter - python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_kiwi --host.connection_time_s=3600")
        print()
        subprocess.Popen(
            ["cmd", "/k", "ssh", "chriscakir@lekiwi.local"],
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )

    task = input("Enter your desired task...\n")

    
    perth_timestamp = time.gmtime(time.time() + 8 * 3600)
    print(f"{time.strftime("%Y-%m-%d %H:%M:%S", perth_timestamp)}:  Executing Task")

    robot = LeKiwiClient(LeKiwiClientConfig(remote_ip="lekiwi.local", id="lekiwi"))
    controller = LeKiwiController(robot)
    controller.connect()
    try:
        manipulation_policy = "outputs/train/0_sock_cube_150/checkpoints/last/pretrained_model"
        controller.load_smolvla_policy(manipulation_policy)
        lidar = Lidar()
        feature_extractor = FeatureExtractor()
        agent = LeKiwiAgent(
            controller=controller, 
            lidar=lidar, 
            feature_extractor=feature_extractor,
            manipulation_policy=manipulation_policy
            )
        agent.go(task)
        
    finally:
        perth_timestamp = time.gmtime(time.time() + 8 * 3600)
        print(f"{time.strftime("%Y-%m-%d %H:%M:%S", perth_timestamp)}:  Disconnecting Gracefully")

        controller.disconnect()


if __name__ == "__main__":
    main()