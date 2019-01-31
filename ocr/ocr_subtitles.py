# USAGE
# python ocr_subtitles.py --pathIn ../frames/ --pathOut ocr.log

# import the necessary packages
import argparse
import glob
import os
import time
from PIL import Image
import cv2
import numpy as np
import pytesseract
from string import maketrans


def blueFilter(image):
    output = None

    # define the list of boundaries - just blues
    boundaries = [
        ([136, 66, 0], [227, 188, 120])
    ]

    # loop over the boundaries
    for (lower, upper) in boundaries:
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(image, lower, upper)
        output = cv2.bitwise_and(image, image, mask=mask)

        # show the images
        # cv2.imshow("images", np.hstack([image, output]))
        # cv2.waitKey(0)

    return output


def threshold(image):
    # initialize a rectangular and square structuring kernel
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

    # smooth the image using a 3x3 Gaussian, then apply the blackhat
    # morphological operator to find dark regions on a light background
    smooth = cv2.GaussianBlur(image, (3, 3), 0)
    blackhat = cv2.morphologyEx(smooth, cv2.MORPH_BLACKHAT, rectKernel)

    # compute the Scharr gradient of the blackhat image and scale the
    # result into the range [0, 255]
    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

    # apply a closing operation using the rectangular kernel to close
    # gaps in between letters -- then apply Otsu's thresholding method
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # perform another closing operation, this time using the square
    # kernel to close gaps between lines of the MRZ, then perform a
    # series of erosions to break apart connected components
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
    thresh = cv2.erode(thresh, None, iterations=4)

    # during thresholding, it's possible that border pixels were
    # included in the thresholding, so let's set 5% of the left and
    # right borders to zero
    p = int(image.shape[1] * 0.05)
    thresh[:, 0:p] = 0
    thresh[:, image.shape[1] - p:] = 0

    return thresh


def extractROI(contours, original_image, gray_image):
    roi = None
    # loop over the contours
    for c in contours:
        # compute the bounding box of the contour and use the contour to
        # compute the aspect ratio and coverage ratio of the bounding box
        # width to the width of the image
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        crWidth = w / float(gray_image.shape[1])

        # check to see if the aspect ratio and coverage width are within
        # acceptable criteria
        if ar > 5 and crWidth > 0.15:
            # pad the bounding box since we applied erosions and now need
            # to re-grow it
            pX = int((x + w) * 0.03)
            pY = int((y + h) * 0.03)
            (x, y) = (x - pX, y - pY)
            (w, h) = (w + (pX * 2), h + (pY * 2))

            # extract the ROI from image
            roi = original_image[y:y + h, x:x + w].copy()

            # draw a bounding box surrounding the ROI
            # cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # show the output images
            # cv2.imshow("Image", original_image)
            # cv2.imshow("ROI", roi)
            # cv2.waitKey(0)

            break

    return roi

def scrunch(s):
    delchars = ''.join(c for c in map(chr, range(256)) if not c.isalnum())
    # print delchars
    table = maketrans(delchars, ' '*len(delchars))
    scrh = s.encode('utf-8').translate(table)
    return scrh

def ocr(roi):
    if roi is None:
        return ''

    # resize image
    img = roi
    scale_percent = 220  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # convert it to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # apply dilation and erosion to remove noise
    kernel = np.ones((1, 1), np.uint8)
    dil = cv2.dilate(gray, kernel, iterations=1)
    ers = cv2.erode(dil, kernel, iterations=1)

    # apply thresholding to preprocess image
    thresh = cv2.threshold(ers, 0, 255,
                             cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # write the grayscale image to disk as a temporary file so we can
    # apply OCR to it
    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, thresh)

    # load the image as a PIL/Pillow image, apply OCR, and then delete
    # the temporary file
    text = pytesseract.image_to_string(Image.open(filename))
    print text
    os.remove(filename)

    # strip newlines
    str = text.strip('\n')

    # remove non-word text
    str = scrunch(str)

    # remove replacements
    # str = str.replace('+', '')

    # strip leading/trailing spaces
    str = str.strip()

    return str.encode('utf-8')


def processFolder(folder, log_filename):
    count = 0

    # for each JPEG image in folder
    imgList = glob.glob(folder + '*.jpg')
    for img in imgList:
        print img

        while True:
            # temperature check
            temperature_milis = open('/sys/class/thermal/thermal_zone0/temp', 'r')
            temperature = int(temperature_milis.read()) / 1000
            # throttle down if temperature is too high
            if temperature > 89:
                time.sleep(5)
            else:
                break

        # load image
        test_image = cv2.imread(img)

        text = ''
        if test_image is not None:
            # apply blue mask
            blue = blueFilter(test_image)

            # convert to grayscale
            gray = cv2.cvtColor(blue, cv2.COLOR_BGR2GRAY)

            # invert image to apply blackhat later
            gray = cv2.bitwise_not(gray)

            # apply morphological operators to define image threshold
            thresh = threshold(gray)

            # find contours in thresholded image and sort them by their
            # size
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)[-2]

            # extract ROI from contours
            roi = extractROI(cnts, test_image, gray)

            # ocr image ROI
            text = ocr(roi)
            print text

        # log findings
        log_file = open(log_filename, 'a')
        log_file.write(img + ',"' + text + '"\n')
        log_file.close()

        count = count + 1

    return count


if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--pathIn", required=True, help="path to images directory")
    a.add_argument("--pathOut", required=True, help="path to log file")
    args = a.parse_args()

    # extract subtitles from JPEG images in pathIn folder
    cnt = processFolder(args.pathIn, args.pathOut)
    print '%d images processed' % cnt