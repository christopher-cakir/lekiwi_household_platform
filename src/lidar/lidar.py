import math
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.ndimage import median_filter

from src.agent.controller import LeKiwiController

import time

class Lidar():
    def __init__(self):
        self.current_angles = None
        self.current_distances = None

        self.raw_angles = None
        self.raw_distances = None

        self.stops = []


    def clean_scan(self, scan):
        angles = np.array(scan["a_deg"])
        distances = np.array(scan["d_mm"], dtype=float)

        self.raw_angles = angles
        self.raw_distances = distances

        angles = (angles + 180) % 360 - 180

        neighbour_radius_mm = 50
        min_neighbours = 3

        x = distances * np.sin(np.radians(angles))
        y = distances * np.cos(np.radians(angles))

        keep = []
        for i in range(len(distances)):
            gaps = np.hypot(x - x[i], y - y[i])
            neighbours = np.sum(gaps < neighbour_radius_mm) - 1
            keep.append(neighbours >= min_neighbours)

        keep = np.array(keep)
        angles = angles[keep]
        distances = distances[keep]

        order = np.argsort(angles)
        angles = angles[order]
        distances = distances[order]

        distances = median_filter(distances, size=5, mode="nearest")

        self.current_angles = angles
        self.current_distances = distances

        return angles, distances


    def update_path(self, tool_name, tool_args):
        
        if tool_name == "rotate_base":
            self.rotate_stops(tool_args["degrees"])
        if tool_name == "move_base_forward":
            self.move_stops(tool_args["cm"])

    def clean_test(self):
        controller = LeKiwiController()
        controller.connect()
        time.sleep(1)

        self.clean_scan(controller.robot.get_observation()["lidar"])
        controller.disconnect()

        axes = plt.subplot(projection="polar")
        axes.set_theta_zero_location("N")
        axes.set_theta_direction(-1)
        axes.set_thetamin(-90)
        axes.set_thetamax(90)
        axes.set_ylim(0, 1500)
        axes.set_thetagrids(range(-120, 121, 30), labels=["-120", "-90", "-60", "-30", "0°", "30", "60", "90", "120"])
        axes.set_rgrids([250, 500, 750, 1000, 1250, 1500], labels=["", "0.5m", "", "1m", "", "1.5m"])
        axes.set_rlabel_position(-90)

        axes.scatter(np.radians(self.raw_angles), self.raw_distances, s=4, color="tab:red", label="Before cleaning")
        axes.scatter(np.radians(self.current_angles), self.current_distances, s=4, color="tab:blue", label="After cleaning")
        axes.legend()

        plt.show()

    def distance_test(self, distance: int):
        controller = LeKiwiController()
        controller.connect()

        observation = controller.robot.get_observation()
        scan = observation["lidar"]

        raw_angles = scan["a_deg"]
        raw_distances = scan["d_mm"]

        clean_angles, clean_distances = self.clean_scan(scan)

        time.sleep(1)
        figure = self.plot()

        # print(lidar.current_angles)
        before_values = list(self.get_distances_in_range(-0.5, 0.5).values())
        before = sum(before_values) / len(before_values)
        print("Before:", round(before, 1), "mm")

        figure.savefig(r'src\lidar\drift_scan_1.png')

        controller.move_base_forward(distance)
        time.sleep(1)
        
        observation = controller.robot.get_observation()
        scan = observation["lidar"]

        raw_angles = scan["a_deg"]
        raw_distances = scan["d_mm"]

        clean_angles, clean_distances = self.clean_scan(scan)

        figure = self.plot()

        after_values = list(self.get_distances_in_range(-0.5, 0.5).values())
        after = sum(after_values) / len(after_values)
        print("After:", round(after, 1), "mm")

        figure.savefig(r'src\lidar\drift_scan_2.png')

        estimated_cm = (before - after) / 10
        print("Commanded:", distance, "cm, Estimated:", round(estimated_cm, 1), "cm")
        
        controller.disconnect()

    def rotation_test(self, rotation:int):
        controller = LeKiwiController()
        controller.connect()

        self.clean_scan(controller.robot.get_observation()["lidar"])
        distances = list(lidar.get_distances_in_range(-0.5, 0.5).values())

        before = sum(distances) / len(distances)
        
        controller.rotate_base(degrees=rotation)

        time.sleep(2)

        self.clean_scan(controller.robot.get_observation()["lidar"])
        distances = list(lidar.get_distances_in_range(-0.5, 0.5).values())

        after = sum(distances) / len(distances)
        
        controller.disconnect()
        
        measured = math.degrees(math.acos(before / after))
        print("Commanded:", rotation, "Measured:", round(measured, 2))



    def rotate_stops(self, degrees):
        for stop in self.stops:
            stop[0] = stop[0] - degrees


    def move_stops(self, cm):
        
        self.stops.append([0.0, 0.0])

        distance_driven = cm * 10 

        for stop in self.stops:
            angle = stop[0]
            distance = stop[1]

            ahead = distance * math.cos(math.radians(angle))
            to_the_side = distance * math.sin(math.radians(angle))

            ahead = ahead - distance_driven

            stop[0] = math.degrees(math.atan2(to_the_side, ahead))
            stop[1] = math.hypot(ahead, to_the_side)


    def plot(self, title="robot at centre, 0° = straight ahead, red dots = previous stops"):
        """Polar plot of the current cleaned scan, with the places the robot stopped before, as a matplotlib Figure."""
        figure = Figure()
        axes = figure.add_subplot(projection="polar")
        axes.set_theta_zero_location("N")
        axes.set_theta_direction(-1) 
        axes.set_ylim(0, 1500)
        axes.set_thetagrids(range(0, 360, 45), labels=["0°", "45°", "90°", "135°", "180°", "-135°", "-90°", "-45°"])
        axes.set_rgrids([500, 1000, 1500], labels=["0.5m", "1m", "1.5m"])
        axes.set_title(title)

        axes.scatter(np.radians(self.current_angles), self.current_distances, s=4)

        for i in range(len(self.stops)):
            angle = self.stops[i][0]
            distance = self.stops[i][1]
            if i == 0:
                axes.plot(math.radians(angle), distance, "*", color="tab:red", markersize=12)
            else:
                axes.plot(math.radians(angle), distance, "o", color="tab:red", markersize=5)

        axes.plot(0, 0, "r^", markersize=10) 
        return figure


    def save_scan(self, path):
        """Write the current cleaned and raw scan to a text file."""
        with open(path, "w", encoding="utf-8") as file:
            file.write("CLEANED (angle_deg, distance_mm)\n")
            for i in range(len(self.current_angles)):
                angle = self.current_angles[i]
                distance = self.current_distances[i]
                file.write(f"{angle:.2f}, {distance:.1f}\n")

            file.write("\nRAW (angle_deg, distance_mm)\n")
            for i in range(len(self.raw_angles)):
                angle = self.raw_angles[i]
                distance = self.raw_distances[i]
                file.write(f"{angle:.2f}, {distance:.1f}\n")


    def get_distances_in_range(self, start_angle, end_angle):

        distances = {}
        for i in range(len(self.current_angles)):
            angle = self.current_angles[i]
            distance = self.current_distances[i]
            if angle >= start_angle and angle <= end_angle:

                distances[round(float(angle),2)] = round(float(distance), 2)

        return distances


if __name__ == '__main__':

    lidar = Lidar()
    lidar.distance_test(5)
    # lidar.rotation_test(40)
    # lidar.clean_test()

    # figure = lidar.plot()
    
    # plt.show()

    