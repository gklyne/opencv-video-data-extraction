# video-read-demo.py

from __future__ import print_function
import cv2 as cv
import argparse
parser = argparse.ArgumentParser(description='This program uses OpenCV to read and process a video of \
                                              a Monotype system paper tape.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
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

writer = cv.VideoWriter(
        'output.mp4',                               # where
        cv.VideoWriter_fourcc('M','J','P','G'),     # CODEC
        10,                                         # FPS
        (frame_width,frame_height))                 # Size

while True:
    ret, frame = capture.read()
    if frame is None:
        break
    
    fgMask = backSub.apply(frame)
    
    cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    cv.imshow('Frame', frame)

    # outframe = cv.cvtColor(fgMask, cv.COLOR_GRAY2RGB)
    outframe = cv.copyTo(frame, mask=fgMask)
    cv.imshow('FG Mask', outframe)
    
    # Write the frame into the file 'output.avi'
    writer.write(outframe)

    # Check for keyboard interrupt, or move to next frame after 20ms
    keyboard = cv.waitKey(20)
    if keyboard != -1: 
    	print(keyboard)
    if keyboard == ord('q') or keyboard == 27:
        break

# Shut down
capture.release()
writer.release()
cv.destroyAllWindows()

