# USAGE
# python detect_faces.py
#                   --pathIn ./frames/
#                   --pathOut ./detected_haar_frontal/
#                   --pathLog ./detected_haar_frontal/faces_haar_frontal.log
#                   --pathHaar ./haarcascade/haarcascade_frontalface_alt2.xml

# import the necessary packages
import argparse
import glob
import time

import cv2
#print(cv2.__version__)


def detectFaces(cascade, test_image, scaleFactor=1.1):
    # fail gracefully if image is empty
    if test_image is None:
        return None, 0

    # create a copy of the image to prevent any changes to the original one.
    image_copy = test_image.copy()

    # convert the test image to gray scale as opencv face detector expects gray images
    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

    # apply the haar classifier to detect faces
    faces_rect = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=5)

    count = 0
    # draw a green rectangle on each detected face
    for (x, y, w, h) in faces_rect:
        count = count + 1
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image_copy, count


def processFolder(folder_in, folder_out, log_filename, haar_filename):
    # load the classifier for profile face
    haar_cascade_classifier = cv2.CascadeClassifier(haar_filename)

    count = 0

    # for each JPEG image in folder
    img_list = glob.glob(folder_in + '*.jpg')
    for img in img_list:
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
        print img
        test_image = cv2.imread(img)

        # call the function to detect features using haar cascade classifier
        img_faces, cnt = detectFaces(haar_cascade_classifier, test_image)

        # if faces were found save frame as new JPEG file
        if cnt > 0:
            img_name = img[img.rfind('/') + 1:img.rfind('.')]
            print img_name

            filename = folder_out + img_name + '_%d.jpg' % cnt
            cv2.imwrite(filename, img_faces)
            print ('Wrote a new file: ', filename)

        # log findings
        log_file = open(log_filename, 'a')
        log_file.write(img + ',%d\n' % cnt)
        log_file.close()

        count = count + 1

    return count


if __name__=="__main__":
    # construct the argument parse and parse the arguments
    a = argparse.ArgumentParser()
    a.add_argument("--pathIn", required=True, help="path to original images directory")
    a.add_argument("--pathOut", required=True, help="path to output images directory")
    a.add_argument("--pathLog", required=True, help="path to log file")
    a.add_argument("--pathHaar", required=True, help="path to Haar cascade classifier file")
    args = a.parse_args()

    # detect faces in every JPEG image in pathIn folder
    cnt  = processFolder(args.pathIn, args.pathOut, args.pathLog, args.pathHaar)
    print '%d images processed' % cnt
