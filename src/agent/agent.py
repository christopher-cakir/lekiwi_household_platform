import cv2
import base64
import csv
import json
import time

from pathlib import Path
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.tools import StructuredTool, tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from src.agent.controller import LeKiwiController
from src.lidar.lidar import Lidar

import argparse

from src.image_feature_extraction.frame_augmentation import FeatureExtractor

from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig

load_dotenv(Path(__file__).with_name(".env"))

STORAGE_DIR = Path(__file__).resolve().parents[1] / "0_storage"

STEP_COLUMNS = [
    "run_id", "step", "timestamp", "llm_call_time_s", "input_tokens", "output_tokens",
    "n_tool_calls", "tool_index", "tool_name", "tool_args", "tool_result", "tool_ok",
    "tool_duration_s", "lidar_ok", "front_img", "wrist_img", "lidar_img",
]

SYSTEM_PROMPT = """You control a LeKiwi mobile manipulator with an omnidirectional base, 6-DOF arm, and gripper.

SENSORS
Each turn provides:
- Front camera: Shows objects, horizontal angles (negative = left, positive = right), and floor distance lines at 20, 30, 50, and 100cm.
- Wrist camera: Shows the gripper and nearby objects. The dark jaw, red jaw, and blue base are robot components, not objects.
- Lidar: Top-down obstacle map centred on the robot, with 0° straight ahead, negative angles to the left, and 0.5m rings extending to 3m. The area behind the robot is not scanned.
- Previous stops: Red dots on the lidar map show where the robot stopped before; the red star marks the starting position. Both are approximate.

AVAILABLE TOOLS
- rotate_base(degrees): Rotate in place. Negative = left, positive = right. Use small rotations for alignment and 40 degree increments for scanning.
- move_base_forward(cm): Drive forward (positive) or backward (negative).
- move_arm("reset"): Return the arm to its driving pose. Use when the arm is awkwardly positioned.
- run_skill: Execute a learned pick-and-place action, placing an object in the basket.
- request_operator_input: Ask the operator one specific question, only after attempting to resolve the problem yourself.
- finish_task: Call only when the task completion conditions are satisfied.

GENERAL RULES
1. Perform ONE tool action at a time. Inspect the updated cameras and lidar before deciding the next action.
2. Before every tool call, explain your reasoning and retain important information, including collected objects, remaining targets, and their approximate locations.
3. Use the front camera to identify objects. Lidar detects obstacles and furniture, not small objects. An area is only considered searched when viewed by the front camera.
4. If the front camera is dark or obstructed by fabric, back away and turn towards open space.

SEARCH STRATEGY
1. Begin every task with a full 360° scan before driving or manipulating objects. First inspect the initial view and headings 30 to either side, then complete a full rotation using consecutive rotate_base() actions, observing after each turn.
2. During scanning, record every detected object, its approximate distance, and the rotation at which it appeared.
3. If a target is found, you may interrupt the scan to collect it.
4. After each successful pickup, perform another full scan before continuing exploration.
5. When no targets are known, navigate to an unsearched part of the room, at least 50cm from all previous scanning positions, and perform another full scan.
6. Choose new search positions using lidar clearance and the red movement path. Prioritise open areas not previously visited, but never consider an area searched from lidar alone.
7. Continue exploring and collecting until the task's completion conditions are met.

NAVIGATION SAFETY
1. Maintain at least 50cm clearance from walls and furniture, except when approaching an object that requires closer access. Keep in mind your body is approximately 30cm wide. 
2. Before moving forward, verify that the nearest lidar obstacle ahead is more than the planned travel distance plus 30cm away. Otherwise, shorten the movement or turn towards open space.
3. Avoid corners and narrow gaps where turning or reversing is difficult.
4. If obstacles are within 30cm on two or more sides, reverse and turn towards the most open direction.
5. If an object appears to only be 50cm away, but repeated movements controls arent working you're stuck. You need to reverse and then rotate.

OBJECT APPROACH AND PICKUP
1. Rotate to centre the target near 0° in the front camera.
2. Approach until the object is approximately 20–30cm away, using the bottom edge where it touches the floor to judge distance.
3. Only call run_skill when:
   - The object's entire bottom edge is clearly below the 30cm line.
   - The object is visible in the wrist camera.
   - You have stated in your reasoning which two distance lines contain the object's bottom edge.
4. If the bottom edge is on or near the 30cm line, or the object lies between the 30cm and 50cm lines, move closer first. If uncertain, do not run the skill.
5. After run_skill, verify success using the front camera. If the object has disappeared from the floor, count it as collected; it should now be in the basket, not the gripper.
6. If pickup fails, improve alignment or distance, and retry.
7. after run_skill, if an object is still in the gripper, rerun the skill until the object is in the basket.

TASK COMPLETION
- If a target count is specified, call finish_task once that number has been successfully collected.
- If no count is specified (e.g. 'collect all cubes'), assume more may exist.
- For tasks without a count, finish only after full scans from at least THREE distinct positions spread across the room have found no additional targets, including a full scan after the final pickup.
- Never finish solely because the room appears small or already explored.

OPERATOR ASSISTANCE
If repeatedly unsuccessful, trapped, or unable to understand the task, attempt to resolve the issue independently first. Only then use request_operator_input with one short, specific question explaining what has already been attempted. Follow the operator's response.

PRIORITY
Search systematically, collect known targets before exploring new areas, maintain safe navigation clearance, verify each pickup, and satisfy all completion conditions before finishing.
"""


@tool
def finish_task() -> str:
    """Call this when the overall task is complete"""
    return "Task finished."


class LeKiwiAgent:
    def __init__(self, 
                 controller, 
                 agent_model="anthropic:claude-haiku-5-5",
                 lidar: Lidar|None = None, 
                 feature_extractor: FeatureExtractor|None = None,
                 manipulation_policy: str|None = None
                 ): 
        self.controller = controller 
        self.feature_extractor = feature_extractor
        self.lidar = lidar
        self.agent_model = agent_model
        self.manipulation_policy = manipulation_policy

        self.tools = [
            StructuredTool.from_function(controller.rotate_base, parse_docstring=True),
            StructuredTool.from_function(controller.move_base_forward, parse_docstring=True),
            StructuredTool.from_function(controller.move_base, parse_docstring=True),
            StructuredTool.from_function(controller.move_arm, parse_docstring=True),
            StructuredTool.from_function(controller.run_policy_task, name="run_skill", parse_docstring=True),
            StructuredTool.from_function(controller.request_operator_input, parse_docstring=True),
            finish_task,
        ]
        self.tool_name_to_tool = {tool.name: tool for tool in self.tools}

        self.llm = init_chat_model(
            agent_model,
            betas=["thinking-binding-controls-2026-08-01"],
            thinking={"type": "adaptive", "display": "summarized", "block_binding": {"prefix_mismatch_behavior": "drop_block"}},
            ).bind_tools(self.tools)

        self.system_message = SystemMessage(SYSTEM_PROMPT)
        self.message_history = [self.system_message]
        self.task = None
        self.done = False
        self.run_dir = None
        self.log_path = None
        self.step = 0

    def _start_run_log(self):
        perth_timestamp = time.gmtime(time.time() + 8 * 3600)
        self.run_dir = STORAGE_DIR / time.strftime("%Y-%m-%d_%H-%M-%S", perth_timestamp)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "log.txt"
        self.steps_csv_path = self.run_dir / "log.csv"
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write(f"AGENT: {self.agent_model}\n\nMANIPULATION_POLICY: {self.manipulation_policy}\n\nTASK: {self.task}\n\nSYSTEM PROMPT:\n{SYSTEM_PROMPT}\n{'=' * 80}\n")
        print(f"Logging run to {self.run_dir}")

    def _log(self, text):
        with open(self.log_path, "a", encoding="utf-8") as file:
            file.write(text + "\n")

    def _append_csv(self, path, columns, row):
        write_header = not path.exists()
        with open(path, "a", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _frame_b64(self, frame):
        ok, buf = cv2.imencode(".jpg", frame)
        return base64.b64encode(buf.tobytes()).decode("utf-8")
    
    def _strip_old_images(self):
        for i, msg in enumerate(self.message_history):
            if not isinstance(msg, HumanMessage):
                continue
            if not isinstance(msg.content, list):
                continue

            new_content = []
            for block in msg.content:
                if not isinstance(block, dict):
                    new_content.append(block)
                    continue

                block_type = block.get("type")

                if block_type == "image_url":
                    new_content.append({"type": "text", "text": "[old image removed]"})
                else:
                    new_content.append(block)

            self.message_history[i] = HumanMessage(new_content)

    def main_loop_content(self):
        perth_timestamp = time.gmtime(time.time() + 8 * 3600)
        print(f"{time.strftime("%Y-%m-%d %H:%M:%S", perth_timestamp)}:  Running Inference")
        step_timestamp = time.strftime("%Y-%m-%d %H:%M:%S", perth_timestamp)
        lidar_error = self.lidar is None
        time.sleep(3)
        obs = self.controller.robot.get_observation()

        front_image = obs["front"]
        front_image = cv2.cvtColor(front_image, cv2.COLOR_BGR2RGB)
        if self.feature_extractor is not None:
            front_image = self.feature_extractor.image_augmentaton(front_image)
        cv2.imwrite(r'src\0_storage\current_front_frame.jpg', front_image)
        front_path = self.run_dir / f"step_{self.step:03d}_front.jpg"
        cv2.imwrite(str(front_path), front_image)

        wrist_image = obs["wrist"]
        wrist_image = cv2.cvtColor(wrist_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(r'src\0_storage\current_wrist_frame.jpg', wrist_image)
        wrist_path = self.run_dir / f"step_{self.step:03d}_wrist.jpg"
        cv2.imwrite(str(wrist_path), wrist_image)

        if self.lidar is not None:
            try:
                lidar_error = False
                self.lidar.clean_scan(scan=obs["lidar"])

                fig = self.lidar.plot()
                fig.savefig(r'src\0_storage\current_lidar_scan.png')
                lidar_path = self.run_dir / f"step_{self.step:03d}_lidar.png"
                fig.savefig(lidar_path)
                self.lidar.save_scan(self.run_dir / f"step_{self.step:03d}_lidar.txt")
            except:
                lidar_error = True
                perth_timestamp = time.gmtime(time.time() + 8 * 3600)
                print(f"{time.strftime("%Y-%m-%d %H:%M:%S", perth_timestamp)}:  ERROR! - LiDAR disconnected")
                

        content = [
            {"type": "text", "text": "Front camera view:"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self._frame_b64(front_image)}"}},
            {"type": "text", "text": "Wrist camera view:"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self._frame_b64(wrist_image)}"}},
        ]

        image_names = [front_path.name, wrist_path.name]

        if self.lidar is not None and not lidar_error:
            lidar_b64 = base64.b64encode(lidar_path.read_bytes()).decode("utf-8")
            content.append({"type": "text", "text": "Lidar map:"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{lidar_b64}"}})
            image_names.append(lidar_path.name)

        if self.task:
            content.append({'type': "text", "text": f"\n\nYour task is: '{self.task}"})

        image_names = iter(image_names)
        input_lines = [f"[image: {next(image_names)}]" if block["type"] == "image_url" else block["text"] for block in content]
        self._log(f"\n--- STEP {self.step} ({time.strftime('%Y-%m-%d %H:%M:%S', perth_timestamp)}) ---\nINPUT:\n" + "\n".join(input_lines))

        self._strip_old_images()
        self.message_history.append(HumanMessage(content))
        llm_start = time.perf_counter()
        response = self.llm.invoke(self.message_history)
        llm_call_time = time.perf_counter() - llm_start
        self.message_history.append(response)

        usage = response.usage_metadata or {}
        self._step_row = {
            "run_id": self.run_dir.name,
            "step": self.step,
            "timestamp": step_timestamp,
            "llm_call_time_s": f"{llm_call_time:.2f}",
            "input_tokens": usage.get("input_tokens", ""),
            "output_tokens": usage.get("output_tokens", ""),
            "n_tool_calls": len(response.tool_calls),
            "lidar_ok": "" if self.lidar is None else not lidar_error,
            "front_img": front_path.name,
            "wrist_img": wrist_path.name,
            "lidar_img": "" if lidar_error else lidar_path.name,
        }

        self._log("\nOUTPUT:")
        response_blocks = response.content if isinstance(response.content, list) else [response.content]
        for block in response_blocks:
            if isinstance(block, str):
                self._log(block)
            elif block.get("type") == "thinking":
                self._log(f"[thinking] {block.get('thinking', '')}")
            elif block.get("type") == "text":
                self._log(block["text"])
        for tool_call in response.tool_calls:
            self._log(f"[tool call] {tool_call['name']}({tool_call['args']})")

        if response.content:
            print(f"Reasoning: {response.content[1:]}")

        if not response.tool_calls:
            print(f"(no tool call) model said: {response.content}")
            self._append_csv(self.steps_csv_path, STEP_COLUMNS, self._step_row)
            return
        for tool_call in response.tool_calls:
            print(f"Calling {tool_call['name']} with {tool_call['args']}")
        self.execute_tool_calls(response.tool_calls)

    def execute_tool_calls(self, tool_calls):
        perth_timestamp = time.gmtime(time.time() + 8 * 3600)
        print(f"{time.strftime("%Y-%m-%d %H:%M:%S", perth_timestamp)}:  Executing Tool")
        for tool_index, tool_call in enumerate(tool_calls):
            tool_ok = True
            tool_start = time.perf_counter()
            try:
                result = self.tool_name_to_tool[tool_call["name"]].invoke(tool_call["args"])
                if self.lidar is not None:
                    self.lidar.update_path(tool_call["name"], tool_call["args"])
            except Exception as e:
                tool_ok = False
                result = f"ERROR calling {tool_call['name']}: {e}"
                print(result)
            tool_duration = time.perf_counter() - tool_start
            self._log(f"[tool result] {tool_call['name']}: {result}")
            self._append_csv(self.steps_csv_path, STEP_COLUMNS, {
                **self._step_row,
                "tool_index": tool_index,
                "tool_name": tool_call["name"],
                "tool_args": json.dumps(tool_call["args"]),
                "tool_result": result,
                "tool_ok": tool_ok,
                "tool_duration_s": f"{tool_duration:.2f}",
            })
            self.message_history.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
            if tool_call["name"] == "finish_task":
                self.done = True

    def go(self, task, max_steps=100):
        self.task = task
        self.done = False
        self._start_run_log()
        for step in range(max_steps):
            self.step = step
            if self.done:
                print("Task complete.")
                self._log("\nTask complete.")
                return
            self.main_loop_content()
        print("Reached max steps.")
        self._log("\nReached max steps.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Pick up all of the blue cubes in the room")
    return parser.parse_args()

if __name__ == "__main__":
    robot = LeKiwiClient(LeKiwiClientConfig(remote_ip="lekiwi.local", id="lekiwi"))
    controller = LeKiwiController(robot)
    controller.connect()
    try:
        controller.load_smolvla_policy("outputs/train/6_blue_cube_pick_up/checkpoints/100000/pretrained_model")
        lidar = Lidar()
        feature_extractor = FeatureExtractor()
        agent = LeKiwiAgent(controller, lidar=lidar, feature_extractor=feature_extractor)
        task = "Pick up all of the blue cubes in the room"
        agent.go(task)
        
    finally:
        perth_timestamp = time.gmtime(time.time() + 8 * 3600)
        print(f"{time.strftime("%Y-%m-%d %H:%M:%S", perth_timestamp)}:  Disconnecting Gracefully")

        controller.disconnect()