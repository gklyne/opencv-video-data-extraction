# video-read-demo.py

from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='This program uses OpenCV to read and process a video of \
                                              a Monotype system paper tape.')
parser.add_argument('--input', type=str, help='Path to a video of a Monotype system tape.', default='Monotype-tape.avi')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()

if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()

capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
if not capture.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)

# Default resolutions of the frame are obtained.
# The default resolutions are system dependent.
# Convert the resolutions from float to integer.
frame_width = int(capture.get(3))   # Where is magic number defined???
frame_height = int(capture.get(4))

print("frame_height", frame_height, "frame_width", frame_width)

# writer = cv.VideoWriter(
#         'output.mp4',                               # where
#         cv.VideoWriter_fourcc('M','J','P','G'),     # CODEC
#         30,                                         # FPS
#         (frame_width*2,frame_height))               # Size
writer = cv.VideoWriter(
        'output.avi',                               # where
        cv.VideoWriter_fourcc('M','J','P','G'),     # CODEC
        30,                                         # FPS
        (frame_width*2,frame_height))               # Size

paused = False
step   = False

while True:

    if not paused or step:
        ret, frame = capture.read()
        if frame is None:
            break

        frame_number = capture.get(cv.CAP_PROP_POS_FRAMES)

        cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
        cv.putText(frame, str(frame_number), (15, 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
        cv.imshow('Frame', frame)

        # outframe = cv.cvtColor(fgMask, cv.COLOR_GRAY2RGB)
        fgMask   = backSub.apply(frame)
        # outframe = cv.copyTo(frame, mask=fgMask)

        frame_grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_blur = cv.GaussianBlur(frame_grey, (7, 7), 0)
        threshold = 200
        high_out  = 255 
        T, frame_thresholded = cv.threshold(
            frame_grey, threshold, high_out, 
            # cv.THRESH_BINARY_INV
            cv.THRESH_BINARY
            )
        
        # Write the frame into the file 'output.avi'
        frame_output = cv.cvtColor(frame_thresholded, cv.COLOR_GRAY2RGB)
        # fgMask       = backSub.apply(frame_output)
        # frame_masked = cv.copyTo(frame_output, mask=fgMask)
        cv.imshow('FG Threshold', frame_output)
        if frame_number > 75:
            frame_pair = np.hstack((frame,frame_output))
            writer.write(frame_pair)

    # Check for keyboard interrupt, or move to next frame after 20ms
    step = False
    keyboard = cv.waitKey(20)
    if keyboard != -1: 
    	print(keyboard)
    if keyboard == ord('p'):
        paused = True
    if keyboard == ord('r'):
        paused = False
    if keyboard == ord('s'):
        step = True
    if keyboard == ord('q') or keyboard == 27:
        break

# Shut down
capture.release()
writer.release()
cv.destroyAllWindows()

