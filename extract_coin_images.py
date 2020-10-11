import glob
import cv2
import os

img_path = glob.glob("data/origin_images/*")

for path in img_path:
    # Read image
    img = cv2.imread(path)
    img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)

    # Convert image to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur
    blur = cv2.GaussianBlur(gray, (0, 0), 3)

    # Adaptive threshold
    th = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Contour
    contours, hier = cv2.findContours(
        th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    # Draw contour
    dst = img.copy()
    idx = 0
    while idx >= 0:
        # ellipse = cv2.fitEllipse(contours[idx])
        # cv2.ellipse(dst, ellipse, (0, 200, 0), 2)
        cnt = contours[idx]
        area = cv2.contourArea(cnt)
        if not (1000 > area and area > 10000):
            continue
        cv2.drawContours(dst, contours, idx, (0, 200, 0), 2)
        idx = hier[0, idx, 0]

    # Show
    title = os.path.basename(path)
    # cv2.imshow(title + " - img", img)
    # cv2.imshow(title + " - gray", gray)
    # cv2.imshow(title + " - th", th)
    cv2.imshow(title + " - dst", dst)

cv2.waitKey(0)
cv2.destroyAllWindows()
