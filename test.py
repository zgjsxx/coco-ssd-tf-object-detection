import cv2
image1 = cv2.imread("tmp/tmp.jpg")
sp = image1.shape
cv2.rectangle(image1,(128, 121), (576,465), (0, 0, 255), 2)

cv2.imshow('face', image1)
cv2.waitKey(0) # 让用户按下键盘任意一个键来退出此图片显示窗口(若没有图像会闪退)