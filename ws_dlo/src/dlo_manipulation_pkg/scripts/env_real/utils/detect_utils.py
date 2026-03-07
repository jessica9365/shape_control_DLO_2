import cv2
import numpy as np
# import pyrealsense2 as rs

class FpDetector:
    def __init__(self, lower_color, upper_color, pixel_range=(0,0,640,480), num_fp=9):
        # Initialize the color range
        self.lower_color = np.array(lower_color)
        self.upper_color = np.array(upper_color)
        self.num_fp = num_fp
        self.startX, self.startY, self.endX, self.endY = pixel_range

    def set_color_range(self, lower_color, upper_color):
        # Update the color range
        self.lower_color = np.array(lower_color, dtype="uint8")
        self.upper_color = np.array(upper_color, dtype="uint8")
    
    def drawDetectionRegion(self, img):
        return cv2.rectangle(img, (self.startX, self.startY), (self.endX, self.endY), (0, 255, 0), 1)
        

    def are_contours_close(self, cnt1, cnt2, threshold=10):
        x1, y1, w1, h1 = cv2.boundingRect(cnt1)
        x2, y2, w2, h2 = cv2.boundingRect(cnt2)
        distance = ((x1 + w1 / 2 - x2 - w2 / 2) ** 2 + (y1 + h1 / 2 - y2 - h2 / 2) ** 2) ** 0.5
        return distance < threshold

    def colorContour(self, img):
        # Crop the image to the specified pixel range
        cropped_img = img[self.startY:self.endY, self.startX:self.endX]

        hsv_image = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv_image, self.lower_color, self.upper_color)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contour_areas = [(cnt, cv2.contourArea(cnt)) for cnt in contours]
        contour_areas.sort(key=lambda x: x[1], reverse=True)
        top_area_contours = [cnt[0] for cnt in contour_areas[:self.num_fp]]
        filtered_contours = []
        for cnt1 in top_area_contours:
            keep = True
            for cnt2 in top_area_contours:
                if cnt2 is not cnt1 and self.are_contours_close(cnt1, cnt2) and cv2.contourArea(cnt1) <= cv2.contourArea(
                        cnt2):
                    keep = False
                    break
            if keep:
                filtered_contours.append(cnt1)

        # Draw on the original image, not the cropped one, adjust coordinates accordingly
        copied_img = img.copy()
        for contour in filtered_contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Adjust contour position back to original image coordinates
            x += self.startX
            y += self.startY
            cv2.circle(copied_img, (int(x + w / 2), int(y + h / 2)), radius=3, color=(0, 0, 255), thickness=-1)

        return copied_img, filtered_contours

    def draw_contour(self, img, keypoints):
        # Crop the image to the specified pixel range
        cropped_img = img[self.startY:self.endY, self.startX:self.endX]
        copied_img = img.copy()
        for (x,y) in keypoints:
            cv2.circle(copied_img, (int(x), int(y)), radius=3, color=(0, 255, 0), thickness=-1)

        return copied_img

    def assignKeypoints(self,keypoints):
        keypoints_assigned = sorted(
            [(x + self.startX + w / 2, y + self.startY + h / 2) for x, y, w, h in [cv2.boundingRect(cnt) for cnt in keypoints]],
            key=lambda x: x[0])
        return keypoints_assigned

    def label_keypoints(self, image, coordinates, labels=None):
        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (0, 0, 255)  # Blue in BGR
        thickness = 2
        text_offset_x = 10
        text_offset_y = 10

        # Loop through the coordinates and add text
        for idx, (x, y) in enumerate(coordinates, start=1):
            if labels is not None:
                text = str(labels[idx])
            else:
                text = str(idx)
            # Position the text around the coordinates
            cv2.putText(image, text, (int(x) + text_offset_x, int(y) + text_offset_y),
                        font, font_scale, font_color, thickness)
            # cv2.drawMarker(image, (int(x),int(y)), color=(255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)
            cv2.circle(image, (int(x), int(y)), radius=3, color=(255, 0, 0), thickness=-1)

        return image

    def label_keypoints_miss(self, image, coordinates, miss_idx, labels=None):

        copied_img = image.copy()

        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (0,0,0)  # Blue in BGR
        thickness = 2
        text_offset_x = 10
        text_offset_y = 10

        # Loop through the coordinates and add text
        for idx, (x, y) in enumerate(coordinates):
            if labels is not None:
                text = str(labels[idx])
            else:
                text = str(idx)
            # Position the text around the coordinates
            if idx not in miss_idx:
                cv2.putText(copied_img, text, (int(x) + text_offset_x, int(y) + text_offset_y),
                            font, font_scale, font_color, thickness)
                # cv2.drawMarker(copied_img, (int(x),int(y)), color=(255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)
                cv2.circle(copied_img, (int(x), int(y)), radius=3, color=(255,0,0), thickness=-1)
            else:
                cv2.putText(copied_img, text, (int(x) + text_offset_x, int(y) + text_offset_y),
                            font, font_scale, font_color, thickness)
                # cv2.drawMarker(copied_img, (int(x),int(y)), color=(255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)
                cv2.circle(copied_img, (int(x), int(y)), radius=3, color=(0, 0, 255), thickness=-1)

        return copied_img





