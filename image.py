import cv2
import numpy as np

image = cv2.imread("/Users/deevash/Desktop/iScreen Shoter - Safari - 250309183320.jpg")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_white = np.array([0, 0, 180])
upper_white = np.array([255, 50, 255])
white_mask = cv2.inRange(hsv, lower_white, upper_white)

lower_yellow = np.array([15, 100, 100])
upper_yellow = np.array([35, 255, 255])
yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

kernel = np.ones((5,5), np.uint8)
white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)

white_edges = cv2.Canny(white_mask, 50, 150)
yellow_edges = cv2.Canny(yellow_mask, 50, 150)

white_lane = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
yellow_lane = cv2.cvtColor(yellow_mask, cv2.COLOR_GRAY2BGR)

white_lane[np.where((white_lane == [255,255,255]).all(axis=2))] = [255, 255, 255]
yellow_lane[np.where((yellow_lane == [255,255,255]).all(axis=2))] = [0, 255, 255]

overlay = cv2.addWeighted(image, 0.8, white_lane, 1, 0)
overlay = cv2.addWeighted(overlay, 0.8, yellow_lane, 1, 0)

cv2.imwrite("processed_highway.jpg", overlay)
cv2.imwrite("white_lane_mask.jpg", white_mask)
cv2.imwrite("yellow_lane_mask.jpg", yellow_mask)

cv2.imshow("Processed Image", overlay)
cv2.imshow("White Lane Mask", white_mask)
cv2.imshow("Yellow Lane Mask", yellow_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
