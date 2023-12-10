import cv2
import numpy as np
from matplotlib import pyplot as plt


def image_read(FileIm):
    image = cv2.imread(FileIm)
    plt.imshow(image)
    plt.show()
    return image


def image_processing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 1.2)
    clahe = cv2.createCLAHE(clipLimit=2.45, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(gray)
    edged = cv2.Canny(enhanced_image, 60, 250)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    plt.imshow(closed)
    plt.show()
    return closed


def distance_between_points(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


def image_contours(image_entrance, min_distance=50):
    cnts, _ = cv2.findContours(image_entrance.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    filtered_cnts = []
    filtered_centers = []

    for c in cnts:
        if cv2.contourArea(c) > 0:
            epsilon = 0.02 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            if len(approx) > 5:
                center = (int(np.mean(approx[:, 0, 0])), int(np.mean(approx[:, 0, 1])))
                if not any(distance_between_points(center, fc) < min_distance for fc in filtered_centers):
                    filtered_centers.append(center)
                    filtered_cnts.append(c)

    return filtered_cnts


def image_recognition(image_entrance, image_cont, file_name):
    total = 0
    for c in image_cont:
        cv2.drawContours(image_entrance, [c], -1, (0, 255, 0), 4)
        total += 1

    print(f"Знайдено {total} шестикутників на зображенні")
    cv2.imwrite(file_name, image_entrance)
    plt.imshow(image_entrance)
    plt.show()
    return


images_url = "img_2"
image_entrance = image_read(f"{images_url}.png")
image_exit = image_processing(image_entrance)
image_cont = image_contours(image_exit)
image_recognition(image_entrance, image_cont, f"recognition_{images_url}.jpg")
