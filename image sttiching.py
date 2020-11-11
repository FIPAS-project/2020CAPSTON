# USAGE
# python image_stitching.py --images images/scottsdale --output output.png --crop 1

# import the necessary packages
import os

from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import matplotlib.pyplot as plt

##########################################################################################
img = cv2.imread('C:/image/stitch/1.jpg')

cv2.imshow('image',img)
cv2.waitKey()
# [x,y] 좌표점을 4x2의 행렬로 작성
# 좌표점은 좌상->좌하->우상->우하
pts1 = np.float32([[155, 146], [56, 715], [539, 140], [673., 702]])

# 좌표의 이동점
pts2 = np.float32([[0, 0], [0, 961], [720, 0], [720, 961]])

# pts1의 좌표에 표시. perspective 변환 후 이동 점 확인.
# cv2.circle(img, (155,146), 20, (255,0,0),-1)
# cv2.circle(img, (56,715), 20, (0,255,0),-1)
# cv2.circle(img, (539,140), 20, (0,0,255),-1)
# cv2.circle(img, (673,702), 20, (0,0,0),-1)

M = cv2.getPerspectiveTransform(pts1, pts2)

dst = cv2.warpPerspective(img, M, (720, 961))

path1 = 'C:/image/stitch'
cv2.imwrite(os.path.join(path1, 'outp.jpg'), dst)

################################################################################################
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", type=str,
                help="path to input directory of images to stitch")
ap.add_argument("-o", "--output", type=str,
                help="path to the output image")
ap.add_argument("-c", "--crop", type=int, default=0,
                help="whether to crop out largest rectangular region")
args = vars(ap.parse_args())

# grab the paths to the input images and initialize our images list
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images("C:/image/stitch")))
images = []

# loop over the image paths, load each one, and add them to our
# images to stich list
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    images.append(image)

# initialize OpenCV's image sticher object and then perform the image
# stitching
print("[INFO] stitching images...")
stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
(status, stitched) = stitcher.stitch(images)

print("Status is =", status)


if status == 0:
    if 1 > 0:
        print("[INFO] cropping...")
        stitched = cv2.copyMakeBorder(stitched, 10, 10, 10, 10,
                                      cv2.BORDER_CONSTANT, (0, 0, 0))
        #스티칭된 이미지를 10픽셀 단위로 구분해준다.

        gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
        #10 픽셀 단위로 구분된 이미지를 회색 반전시켜준다.

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts2 = imutils.grab_contours(cnts)
        c = max(cnts2, key=cv2.contourArea)
        #thresh에 저장된 이미지의 경계선을 찾고, 그영역의 넓이를 비교하여 너 큰 넓이값을 c에 저장한다.

        mask = np.zeros(thresh.shape, dtype="uint8")
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        #그 넓이에 맞는 사각형을 생성. -> 스티칭된 이미지가 다 들어가는 가장 큰 초기 직사각형

        minRect = mask.copy()
        sub = mask.copy()

        while cv2.countNonZero(sub) > 0:
            minRect = cv2.erode(minRect, None)
            sub = cv2.subtract(minRect, thresh)
        #초기 사각형에서 thresh이미지를 빼준다. -> 결과값이 0이 될때까지 반복
        #-> 스티칭된 이미지에서 생긴 검은색 부분이 없는 최소 사각형의 크기 추출

        cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(c)
        #최소 크기 사각형에서 시작점(x,y), 가로세로 길이(w,h)추출.

        stitched = stitched[y:y + h, x:x + w]
        #시작점(x,y), 가로세로 길이(w,h)를 통해 최소 사각형에 맞는 이미지만 추출.

    path = 'C:/image/haedong'
    cv2.imwrite(os.path.join(path, 'output.jpg'), stitched)
        #원하는 경로에 결과 이미지 저장.

    # display the output stitched image to our screen
    cv2.imshow("Stitched", stitched)
    cv2.waitKey(0)

# otherwise the stitching failed, likely due to not enough keypoints)
# being detected
else:
    print("[INFO] image stitching failed ({})".format(status))
