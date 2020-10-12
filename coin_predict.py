import glob
import cv2
import os
import numpy as np
from keras.models import load_model

labels = ["100won", "10won", "500won", "50won"]

model = load_model('model/my_model.h5')

img_path = glob.glob("data/origin_images/*.jpg")

for path in img_path:
    # Read image
    org = cv2.imread(path)
    img = cv2.resize(org, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)

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
        # Filter area
        cnt = contours[idx]
        area = cv2.contourArea(cnt)
        if 500 > area or area > 6000:
            idx = hier[0, idx, 0]
            continue

        # Filter aspect ratio
        _, _, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h
        if abs(1 - aspect_ratio) > 0.4:
            idx = hier[0, idx, 0]
            continue

        # Convex hull
        hull = cv2.convexHull(contours[idx])

        # Fit rectangle
        x, y, w, h = cv2.boundingRect(hull)

        # Draw rectangle
        cv2.rectangle(dst, (x, y), (x+w, y+h), (0, 0, 255), 1)
        idx = hier[0, idx, 0]

        # Crop coin image
        coin = org[y*5:(y+h)*5, x*5:(x+w)*5, :]
        coin = cv2.resize(coin, (300, 300), interpolation=cv2.INTER_AREA)

        # Predict
        coin = coin.reshape(-1, 300, 300, 3)
        prediction = model.predict([coin])
        label = labels[np.argmax(prediction)]

        # Show label
        cv2.putText(dst, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))


    # Show
    title = os.path.basename(path)
    # cv2.imshow(title + " - img", img)
    # cv2.imshow(title + " - gray", gray)
    # cv2.imshow(title + " - th", th)
    cv2.imshow(title + " - dst", dst)

while cv2.waitKey(0) != ord('q'):
    pass
cv2.destroyAllWindows()