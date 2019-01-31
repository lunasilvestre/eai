# USAGE
# python detect_faces.py --video ../p5272_1_201901012044180587.mp4 --pathOut frames/

# import the necessary packages
import argparse
import time

import cv2
# print(cv2.__version__)


def extractImages(video, outputFolder):
    count = 0
    vidcap = cv2.VideoCapture(video)
    success,image = vidcap.read()
    success = True
    while success:
        while True:
            # check temperature
            temperature_milis = open('/sys/class/thermal/thermal_zone0/temp', 'r')
            temperature = int(temperature_milis.read()) / 1000
            print temperature
            # throttle down if temperature is too high
            if temperature > 89:
                time.sleep(5)
            else:
                break
        # read frame
        vidcap.set(cv2.cv.CV_CAP_PROP_POS_MSEC,(count*1000))    # added this line
        success,image = vidcap.read()
        print ('Read a new frame: ', success)

        # write frame as new image
        filename = outputFolder + "frame%d.jpg" % count
        cv2.imwrite(filename, image)     # save frame as JPEG file
        print ('Wrote a new file: ', filename)
        count = count + 1


if __name__=="__main__":
    # construct the argument parse and parse the arguments
    a = argparse.ArgumentParser()
    a.add_argument("--video", required=True, help="path to video")
    a.add_argument("--pathOut", required=True, help="path to images directory")
    args = a.parse_args()

    # extract frames as images from video
    extractImages(args.video, args.pathOut)
