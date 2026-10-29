import time 
import torch
from typing import Literal

from lerobot.common.control_utils import predict_action

from lerobot.datasets import LeRobotDataset

from lerobot.policies import make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig 
from lerobot.utils.utils import log_say
                

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from lerobot.processor import make_default_processors

from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say

ARM_KEYS = [
    'arm_shoulder_pan.pos', 'arm_shoulder_lift.pos', 'arm_elbow_flex.pos', 
    'arm_wrist_flex.pos', 'arm_wrist_roll.pos', 'arm_gripper.pos'
]

ARM_PRESETS: dict[str, dict[str, float] | None] = {
    "reset": {'arm_shoulder_pan.pos': -10.54945054945055, 'arm_shoulder_lift.pos': -43.86813186813187, 'arm_elbow_flex.pos': 2.4615384615384617, 'arm_wrist_flex.pos': 94.41758241758242, 'arm_wrist_roll.pos': -87.51648351648352, 'arm_gripper.pos': 1.0366275051831375},          
}

# Controller reference: https://github.com/alfredang/lerobot

class LeKiwiController:
    
    def __init__(self, robot: LeKiwiClient|None = None, fps: int = 30):
         
         if robot is None:
            robot=LeKiwiClient(LeKiwiClientConfig(remote_ip="lekiwi.local", id="lekiwi"))
         self.robot=robot

         self.fps=fps
         self.control_interval = 1.0/fps
    
    def connect(self):
        self.robot.connect()

    def disconnect(self):
        self.robot.disconnect()

    def _get_current_arm(self):
        obs = self.robot.get_observation()
        return {arm: obs[arm] for arm in ARM_KEYS}
    
    def move_base(self, 
                  x_vel: float = 0.0,
                  y_vel: float = 0.0, 
                  theta_vel: float = 0.0, 
                  duration: float = 1.0
                  ) -> str:
        """Last resort only: Move the base sideways at a constant speed for a fixed duration, 
        then stop.

        Always use rotate_base to turn and move_base_forward to move forward or backward. 
        Only use this tool if both have repeatedly failed to reposition the robot, such as when 
        it is stuck against furniture and cannot turn or reverse. 
        Never use sideways movement to align with an object. Rotate towards it instead.

        Args:
            x_vel: Always 0. Use move_base_forward instead.
            y_vel: Sideways velocity. Positive = left, negative = right. Range: -0.1 to 0.1.
            theta_vel: Always 0. Use rotate_base instead.
            duration: Movement duration in seconds. Range: 1 to 2.
        """
        time_start = time.perf_counter() 
        perth_timestamp_start = time.gmtime(time.time() + 8 * 3600)
        print(f"{time.strftime("%Y-%m-%d %H:%M:%S", perth_timestamp_start)}: Moving Base - x_vel={x_vel}, y_vel={y_vel}, theta_vel={theta_vel}, duration={duration}")

        current_arm = self._get_current_arm()

        t0 = time.perf_counter()
        while time.perf_counter() - t0 < duration:
                self.robot.send_action({
                    **current_arm,
                    "x.vel": x_vel,
                    "y.vel": y_vel,
                    "theta.vel": theta_vel 
                    })
                time.sleep(self.control_interval)
        
        self.robot.send_action({
            **current_arm,
            "x.vel": 0.0,
            "y.vel": 0.0,
            "theta.vel": 0.0 
            })
        
        time_end = time.perf_counter() 
        perth_timestamp_end = time.gmtime(time.time() + 8 * 3600)
        print(f"{time.strftime("%Y-%m-%d %H:%M:%S", perth_timestamp_end)}:  Moved Base Successfully ({(time_end - time_start):.2f} seconds) - x_vel={x_vel}, y_vel={y_vel}, theta_vel={theta_vel}, duration={duration}")

        return f"Moved base (x={x_vel}, y={y_vel}, theta={theta_vel}) for {duration}s."
    
    def rotate_base(self, degrees: Literal[-33, -11, -6, 6, 11, 33]) -> str:
        """Rotate the base in place by a fixed angle, then stop. The arm remains stationary.

        Use this tool for all turns instead of `move_base`. The direction matches the 
        front camera and lidar angle markers: negative = left, positive = right. To face 
        an object, rotate by the angle shown at its position 

        Args:
            degrees: Rotation angle. Negative = left (counter-clockwise), positive = right (clockwise). Allowed values: -33, -11, -6, 6, 11, 33.
        """
        time_start = time.perf_counter() 
        perth_timestamp_start = time.gmtime(time.time() + 8 * 3600)
        print(f"{time.strftime("%Y-%m-%d %H:%M:%S", perth_timestamp_start)}: Rotating - {degrees} degrees")
        rotate_mappings = {

                -33: [19, 2],
                -11: [14, 1],
                -6: [7.05, 1],
                6: [-7.05, 1],
                11: [-14, 1],
                33: [-19, 2]
        }
        log_say(f"Rotating base {str(degrees)}")
        self.move_base(theta_vel=rotate_mappings[degrees][0], duration=rotate_mappings[degrees][1])
        return f"Rotated base {degrees} degrees."

# -40, -30, -20, -10, -5, 5, 10, 20, 30, 40

    def move_base_forward(self, cm: Literal[-7, -12, 7, 13, 16, 33, 66]) -> str:
        """Drive the base straight forward or backward by a fixed distance, then stop. 
        The arm remains stationary.

        Rotate to face the object before moving, as this tool only drives straight. 
        To reach pick-up range (20 to 30 cm from the object), choose the largest distance 
        that does not overshoot the target. Use negative distances to reverse if needed. 
        Always check the lidar readings near 0 degrees before moving to avoid obstacles.

        Args:
            cm: Travel distance in centimetres. Positive = forward, negative = backward. Allowed values: -7, -12, 7, 13, 16, 33, 66.
        """
        time_start = time.perf_counter() 
        perth_timestamp_start = time.gmtime(time.time() + 8 * 3600)
        print(f"{time.strftime("%Y-%m-%d %H:%M:%S", perth_timestamp_start)}: Forward - {cm}cm")
        forward_mappings = {
                -7: [-0.08, 1],
                -12: [-0.12, 1],
                7: [0.08, 1],
                13: [0.13, 1],
                16: [0.19, 1],
                33: [0.17, 2],
                66: [0.17, 4],

        }

        log_say(f"Moving base {str(cm)}")
        self.move_base(x_vel=forward_mappings[cm][0], duration=forward_mappings[cm][1])
        return f"Moved base forward {cm}cm."

    def move_arm(self, position: Literal["reset"]) -> str:
        """Smoothly move the arm to a preset position over 2 seconds. The base remains stationary.

        Use reset to return the arm to its default position before driving, after completing 
        a skill, or when the arm is left in an awkward position.

        Args:
            position: The preset to move to. "reset" = reset arm.
        """
        time_start = time.perf_counter()
        perth_timestamp_start = time.gmtime(time.time() + 8 * 3600)
        print(f"{time.strftime("%Y-%m-%d %H:%M:%S", perth_timestamp_start)}: Moving Arm - {position}")

        target = ARM_PRESETS[position]
        if target is None:
            print(f"{time.strftime("%Y-%m-%d %H:%M:%S", perth_timestamp_start)}: ERROR! - Arm preset '{position}' not set in ARM_PRESETS")
            return f"Arm preset '{position}' is not available. The arm did not move."

        current = self._get_current_arm()
        duration = 2.0
        steps = int(duration * self.fps)

        for i in range(1, steps + 1):
            progress = i / steps
            self.robot.send_action({
                **{arm: current[arm] + (target[arm] - current[arm]) * progress for arm in ARM_KEYS},
                "x.vel": 0.0,
                "y.vel": 0.0,
                "theta.vel": 0.0
                })
            time.sleep(self.control_interval)

        time_end = time.perf_counter()
        perth_timestamp_end = time.gmtime(time.time() + 8 * 3600)
        print(f"{time.strftime("%Y-%m-%d %H:%M:%S", perth_timestamp_end)}: Moved Arm Successfully ({(time_end - time_start):.2f} seconds) - {position}")

        return f"Moved arm to '{position}'."

    def request_operator_input(self, question: str) -> str:
        """Ask the human operator a question and wait for their response. 
        The robot remains stationary.

        Only use this tool as a last resort after attempting to resolve the problem 
        independently, such as when the target object cannot be found after several full 
        scans or the robot remains stuck. Make your own decisions whenever possible. 
        Ask one short, specific question explaining what you have already tried.

        Args:
            question: A short, specific question for the operator, including what has already been attempted.
        """
        time_start = time.perf_counter()
        perth_timestamp_start = time.gmtime(time.time() + 8 * 3600)
        print(f"{time.strftime("%Y-%m-%d %H:%M:%S", perth_timestamp_start)}: Asking Question - {question}")
        log_say(f"Asking {question}")
        answer = input("Answer: \n")

        time_end = time.perf_counter()
        perth_timestamp_end = time.gmtime(time.time() + 8 * 3600)
        print(f"{time.strftime("%Y-%m-%d %H:%M:%S", perth_timestamp_end)}: Answer: ({(time_end - time_start):.2f} seconds) - {answer}")

        return f"Operator answer: '{answer}'."



    # NOTE: based on src\training-a-policy\evaluate.py
    def load_smolvla_policy(self, model_id: str = "outputs/train/0_sock_pick_up/checkpoints/080000/pretrained_model"):
        time_start = time.perf_counter() 
        perth_timestamp_start = time.gmtime(time.time() + 8 * 3600)
        print(f"{time.strftime("%Y-%m-%d %H:%M:%S", perth_timestamp_start)}: Model Loading - {model_id}")

        self.policy = SmolVLAPolicy.from_pretrained(model_id)
        self.device = torch.device(self.policy.config.device)
        self.policy.to(self.device)

        action_features = hw_to_dataset_features(self.robot.action_features, ACTION)

        obs_features = hw_to_dataset_features(self.robot.observation_features, OBS_STR)
        self.dataset_features = {**action_features, **obs_features}

        self.dataset = LeRobotDataset.create(
            repo_id=f"chriscakir/agent_tmp_{int(time.time())}",
            fps=self.fps,
            features=self.dataset_features,
            robot_type=self.robot.name,
            use_videos=True,
            image_writer_threads=4,
        )

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy,
            pretrained_path=model_id,
            dataset_stats=self.dataset.meta.stats,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )

        _, self.robot_action_processor, self.robot_observation_processor = make_default_processors()

        time_end = time.perf_counter() 
        perth_timestamp_end = time.gmtime(time.time() + 8 * 3600)
        print(f"{time.strftime("%Y-%m-%d %H:%M:%S", perth_timestamp_end)}: Model Loaded Successfully ({(time_end - time_start):.2f} seconds) - {model_id}")

        

    def run_policy_task(
        self,
        task: Literal[
            "Pick up the sock and place it in the basket",
            "Pick up the blue cube and place it in the basket",
            ],
        duration: float = 60,
    ) -> str:
        """Run a learned SmolVLA skill to pick up and place an object.

        Before running the skill, ensure the object is 20–30 cm in front of the robot and visible in both the front and wrist cameras. The skill does not report success, so check the next camera images to confirm the object was successfully placed.

        Args:
            task: The exact trained skill instruction to execute.
            duration: Policy execution time in seconds. Range: 20 to 40.
        """

        task = str(task)
        log_say(f"Running task {task}")
        time_start = time.perf_counter() 
        perth_timestamp_start = time.gmtime(time.time() + 8 * 3600)
        print(f"{time.strftime("%Y-%m-%d %H:%M:%S", perth_timestamp_start)}: Running Task - {task}")

        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

        t0 = time.perf_counter()
        while time.perf_counter() - t0 < duration:
            start_loop_t = time.perf_counter()

            obs = self.robot.get_observation()
            obs_processed = self.robot_observation_processor(obs)
            observation_frame = build_dataset_frame(self.dataset_features, obs_processed, prefix=OBS_STR)

            action_tensor = predict_action(
                observation=observation_frame,
                policy=self.policy,
                device=self.device,
                preprocessor=self.preprocessor,
                postprocessor=self.postprocessor,
                use_amp=self.device.type == "cuda",
                task=task,
                robot_type=self.robot.name
            )

            action_values=make_robot_action(action_tensor, self.dataset.features)
            robot_action_to_send = self.robot_action_processor((action_values, obs))
            self.robot.send_action(robot_action_to_send)

            dt_s = time.perf_counter() - start_loop_t
            precise_sleep(max(self.control_interval - dt_s, 0.0))

        
        time_end = time.perf_counter() 
        perth_timestamp_end = time.gmtime(time.time() + 8 * 3600)
        print(f"{time.strftime("%Y-%m-%d %H:%M:%S", perth_timestamp_end)}: Completed Task ({(time_end - time_start):.2f} seconds) - {task}")
        
        self.move_arm("reset")

        return f"Ran skill '{task}' for {duration}s."


        




if __name__ == "__main__":
     # NOTE: test class functionality

    robot_config = LeKiwiClientConfig(remote_ip="lekiwi.local", id="lekiwi")
    robot = LeKiwiClient(robot_config)

    bubbles = LeKiwiController(robot)

    bubbles.connect()
    try:
        print('test')
        bubbles.move_arm("reset")
        # bubbles.rotate_base(-20)
        # bubbles.move_base_forward(cm=30)
        # bubbles.load_smolvla_policy("outputs/train/5_main/checkpoints/100000/pretrained_model")
        # bubbles.run_policy_task("Pick up the blue cube and place it in the basket")
    finally:
        bubbles.disconnect()

    
#     WARNING:root:Record loop is running slower (29.6 Hz) than the target FPS (30 Hz). Dataset frames might be dropped and robot control might be unstable. Common causes are: 1) Camera FPS not keeping up 2) Policy inference taking too long 3) CPU starvation
# WARNING:root:Record loop is running slower (29.5 Hz) than the target FPS (30 Hz). Dataset frames might be dropped and robot control might be unstable. Common causes are: 1) Camera FPS not keeping up 2) Policy inference taking too long 3) CPU starvation
# WARNING:root:Record loop is running slower (30.0 Hz) than the target FPS (30 Hz). Dataset frames might be dropped and robot control might be unstable. Common causes are: 1) Camera FPS not keeping up 2) Policy inference taking too long 3) CPU starvation
# WARNING:root:Record loop is running slower (29.6 Hz) than the target FPS (30 Hz). Dataset frames might be dropped and robot control might be unstable. Common causes are: 1) Camera FPS not keeping up 2) Policy inference taking too long 3) CPU starvation
# WARNING:root:Record loop is running slower (29.4 Hz) than the target FPS (30 Hz). Dataset frames might be dropped and robot control might be unstable. Common causes are: 1) Camera FPS not keeping up 2) Policy inference taking too long 3) CPU starvation
# WARNING:root:Record loop is running slower (29.6 Hz) than the target FPS (30 Hz). Dataset frames might be dropped and robot control might be unstable. Common causes are: 1) Camera FPS not keeping up 2) Policy inference taking too long 3) CPU starvation
# WARNING:root:Record loop is running slower (30.0 Hz) than the target FPS (30 Hz). Dataset frames might be dropped and robot control might be unstable. Common causes are: 1) Camera FPS not keeping up 2) Policy inference taking too long 3) CPU starvation
# WARNING:root:Record loop is running slower (29.7 Hz) than the target FPS (30 Hz). Dataset frames might be dropped and robot control might be unstable. Common causes are: 1) Camera FPS not keeping up 2) Policy inference taking too long 3) CPU starvation
# WARNING:root:Record loop is running slower (29.0 Hz) than the target FPS (30 Hz). Dataset frames might be dropped and robot control might be unstable. Common causes are: 1) Camera FPS not keeping up 2) Policy inference taking too long 3) CPU starvation
# WARNING:root:Record loop is running slower (29.9 Hz) than the target FPS (30 Hz). Dataset frames might be dropped and robot control might be unstable. Common causes are: 1) Camera FPS not keeping up 2) Policy inference taking too long 3) CPU starvation
# WARNING:root:Record loop is running slower (29.2 Hz) than the target FPS (30 Hz). Dataset frames might be dropped and robot control might be unstable. Common causes are: 1) Camera FPS not keeping up 2) Policy inference taking too long 3) CPU starvation


# INFO 2026-09-22 21:30:35 ot_train.py:414 num_total_params=450046176 (450M)
# Training:   0%|                                                                           | 0/100000 [00:00<?, ?step/s]INFO 2026-09-22 21:30:35 ot_train.py:565 Start offline training on a fixed dataset, with effective batch size: 8
# Training:   0%|                                                              | 59/100000 [02:57<897:21:13, 32.32s/step]