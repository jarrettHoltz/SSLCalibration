###
### Command:
### ballDetection.py -v <filepath>
###

# import the necessary packages
from collections import deque
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
    help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
    help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
orangeLower = (0,100,100)
orangeUpper = (15,255,255)
pts = deque(maxlen=args["buffer"])
trackedPoints = []

frame_rate = 142
start_frame = 0
end_frame = 0

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
    camera = cv2.VideoCapture(0)

# otherwise, grab a reference to the video file
else:
    camera = cv2.VideoCapture(args["video"])
frameNumber = 0
# keep looping
while True:
    frameNumber = frameNumber + 1
    # grab the current frame
    (grabbed, frame) = camera.read()

    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if args.get("video") and not grabbed:
        break

    # resize the frame, blur it, and convert it to the HSV
    # color space
    #frame = imutils.resize(frame, width=600)
    #blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, orangeLower, orangeUpper)
    #mask = cv2.erode(mask, None, iterations=2)
    #mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    trackedPoints.append(None)
    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        # only proceed if the radius meets a minimum size
        if(M["m00"] != 0 and radius < 10):
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            trackedPoints[-1] = [center[0],center[1],frameNumber]
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),
                (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

            # update the points queue
            pts.appendleft(center)

    # loop over the set of tracked points
    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore them
        #if pts[i - 1] is None or pts[i] is None:
        #    continue

        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    # show the frame to our screen
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(0)

    if key == ord("p"):
        print(frameNumber)
    if key == ord("s"):
        print(frameNumber)
        start_frame = frameNumber
    if key == ord("e"):
        print(frameNumber)
        end_frame = frameNumber
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

f = open(args["video"]+'.csv', 'w')
f.write(str(frame_rate)+'\n')
f.write(str(float(end_frame - start_frame)/float(frame_rate))+'\n')
for loc in trackedPoints[start_frame:end_frame+1]:
    f.write(str(loc[0])+','+str(loc[1])+','+str(loc[2]-start_frame)+'\n')
f.close()
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
