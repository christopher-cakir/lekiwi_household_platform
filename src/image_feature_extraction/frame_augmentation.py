import cv2
import math

from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
import time
class FeatureExtractor():
    def __init__(self):
        pass
        
    def calculate_angle_marks(self, width, h_fov, center_angle, mark_len_angle=10):
        """Return (x_pixel, angle_degrees) tuples, one per tick mark."""
        nr_of_marks = int((h_fov / 2) // mark_len_angle * 2 + 1)
        pixels_per_mark = width / h_fov * mark_len_angle
        start_pixel = (width - (nr_of_marks - 1) * pixels_per_mark) / 2
        start_angle = mark_len_angle * math.trunc((-h_fov / 2 + center_angle) / mark_len_angle)
        return [
            (int(start_pixel + i * pixels_per_mark), start_angle + i * mark_len_angle)
            for i in range(nr_of_marks)
        ]


    def angle_augmentation(self, image, h_fov=45, center_angle=0):
        """Draw horizontal angle markers near the top of the image."""
        height, width = image.shape[:2]
        yellow = (0, 255, 255)
        orange = (0, 100, 255)
        y_pos = 25

        # cv2.line(image, (0, y_pos), (width, y_pos), yellow, 2)

        for x, angle in self.calculate_angle_marks(width, h_fov, center_angle):
            cv2.line(image, (x, 0), (x, height), orange, 2)
            cv2.putText(image, f"{angle} degrees", (x - 35, y_pos + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, yellow, 2)

        cv2.putText(image, "<=LEFT", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, yellow, 2)
        cv2.putText(image, "RIGHT=>", (width - 145, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, yellow, 2)
        return image

    def distance_augmenttion(self, img):


        distance_rows = {
            # cm: image row
            20: 300, 
            30: 239, 
            50: 190, 
            100: 153
            }

        h, w = img.shape[:2]
        orange = (0, 100, 255)
        yellow = (0, 255, 255)

        for d, y in distance_rows.items():
            if 0 <= y < h:
                cv2.line(img, (0, y), (w, y), orange, 2)
                cv2.putText(img, f"{d}cm", (10, y - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, yellow, 2)

        return img
    
    def image_augmentaton(self, img):
        img = self.angle_augmentation(img, h_fov=45)
        img = self.distance_augmenttion(img)

        return img
    
if __name__ == "__main__":

    robot_config = LeKiwiClientConfig(remote_ip="lekiwi.local", id="lekiwi")
    robot = LeKiwiClient(robot_config)

    # bubbles = LeKiwiController(robot)

    robot.connect()
    
    feature_extractor = FeatureExtractor()

    while True:
        obs = robot.get_observation()

        front_image = obs["front"]
        front_image = cv2.cvtColor(front_image, cv2.COLOR_BGR2RGB)
        wrist_image = obs["wrist"]
        wrist_image = cv2.cvtColor(wrist_image , cv2.COLOR_BGR2RGB)
        cv2.imwrite(r'src\image_feature_extraction\wrist_frame.jpg', front_image)
        cv2.imwrite(r'src\image_feature_extraction\front_frame.jpg', wrist_image)
        
        front_image = obs["front"]
        # img = cv2.imread("frame.jpg")
        img = cv2.cvtColor(front_image, cv2.COLOR_BGR2RGB)
        img_new = feature_extractor.image_augmentaton(img)
        cv2.imwrite(r'src\image_feature_extraction\frame_aug_1.jpg', img_new)
        time.sleep(0.5)
