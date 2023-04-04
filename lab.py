import cv2
import numpy as np
import imutils


def f1():
    img = cv2.imread('4.jpeg', cv2.IMREAD_UNCHANGED)
    b, g, r = cv2.split(img)
    blue_img = np.zeros(img.shape)
    blue_img[:, :, 0] = b
    cv2.imwrite('4_blue.jpeg', blue_img)
    cv2.waitKey(0)


def f23():
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()

        if img.shape[1] > 600:
            img = imutils.resize(img, width=600)
        clone = img.copy()

        gray = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)

        bilateral_filtered_image = cv2.bilateralFilter(gray, 11, 15, 15)

        edge_detected_image = cv2.Canny(bilateral_filtered_image, 30, 30)

        contours, pas = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contour_list = []

        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            area = cv2.contourArea(contour)
            if ((len(approx) > 5) & (1000000 > area > 10000)):
                contour_list.append(contour)
                if min(list(contour)[0])[0] > clone.shape[1] // 2 + 50:
                    cv2.putText(clone, 'Right=True', (100, 150), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255), thickness=1)

        cv2.drawContours(clone, contour_list, -1, (0, 255, 0), 2)

        cv2.imshow('Result', clone)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


f1()
f23()
