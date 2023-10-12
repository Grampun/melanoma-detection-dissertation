import numpy as np
import cv2
from PIL import Image
import os


def imageCleaning():
    # Read the image and perfrom an OTSU threshold
    fileCount = 0
    directory = '/Users/Grampun/Desktop/ISIC-Archive-Downloader-master/data_set/training_data/benign/'

    for filename in os.listdir(directory):
        fileCount += 1
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            file = directory + filename
            img = cv2.imread(file)
            kernel = np.ones((15, 15), np.uint8)

            # Binarize the image
            gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

            # Search for contours and select the biggest one
            contours, hierarchy = cv2.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            cnt = max(contours, key=cv2.contourArea)

            # Create a new mask for the result image
            h, w = img.shape[:2]
            mask = np.zeros((h, w), np.uint8)

            # Draw the contour on the new mask and perform the bitwise operation
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            res = cv2.bitwise_and(img, img, mask=mask)

            directoryNew = '/Users/Grampun/Desktop/ISIC-Archive-Downloader-master/masked_data_set/training_data/benign/'

            print(filename + ' is being saved in:  ' + directoryNew)

            cv2.imwrite(
                directoryNew + filename, res)

    print(str(fileCount) + ' skin-lesion images processed and saved to: ' + directoryNew)


imageCleaning()
