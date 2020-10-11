import glob
import cv2
import os

img_path = glob.glob("data/origin_images/*")

for path in img_path:
    img = cv2.imread(path)
    img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (0, 0), 3)

    _, th = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY_INV)

    cv2.imshow(os.path.basename(path), th)

cv2.waitKey(0)
cv2.destroyAllWindows()