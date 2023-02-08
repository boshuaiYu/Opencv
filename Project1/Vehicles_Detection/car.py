import cv2
import numpy as np

cv2.namedWindow("car_affine", cv2.WINDOW_NORMAL)
cv2.resizeWindow("car_affine", (1125, 633))

img = cv2.imread("img/car.jpg")
move = cv2.warpAffine(img, np.float32([[1, 0, 10], [0, 1, 20]]), (1125, 633))  # 平移
M_rotate = cv2.getRotationMatrix2D((1125, 633), 10, 1)
rotate = cv2.warpAffine(move, M_rotate, (1125, 633))  # 旋转
M_incline = cv2.getAffineTransform(src=np.float32([[0, 0], [1125, 0], [0, 633]]), dst=np.float32([[10, 0], [1135, 0], [0, 633]]))  # 倾斜
affine_all = cv2.warpAffine(rotate, M_incline, (1125, 633))

while True:
    cv2.imshow("car_affine", affine_all)
    cv2.imwrite("img/car_affine.jpg", affine_all)
    key = cv2.waitKey(0)
    if key == ord("q"):
        break
