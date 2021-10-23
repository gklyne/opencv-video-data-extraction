# video-read-demo.py

from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import math
import sys

# Video I/O functions

def open_video_input(filepath):
    # Open video for reading
    #
    # returns (video_capture, frame_width, frame_height)
    #
    capture = cv.VideoCapture(filepath)
    if capture.isOpened():
        # Default resolutions of the frame are obtained.
        # The default resolutions are system dependent.
        # Convert the resolutions from float to integer.
        frame_width = int(capture.get(3))   # Where is magic number defined???
        frame_height = int(capture.get(4))
    else:
        capture      = None
        frame_width  = 0
        frame_height = 0
    return (capture, frame_width, frame_height)

def close_video_input(video_capture):
    video_capture.release()
    return

def close_video_output(video_writer):
    video_writer.release()
    return

def open_video_output(filepath, frame_width, frame_height):
    # Open video for writing
    #
    # returns (video_writer)
    #
    writer = cv.VideoWriter(
        'output.avi',                               # where
        cv.VideoWriter_fourcc('M','J','P','G'),     # CODEC
        30,                                         # FPS
        (frame_width,frame_height)                  # Size
        )
    return writer

def read_video_frame(video_capture):
    # Read frame from VideoCapture object
    #
    # returns (frame_number, frame_data), or None
    ret, frame   = video_capture.read()
    if frame is None:
        return (None, None)
    frame_number = video_capture.get(cv.CAP_PROP_POS_FRAMES)
    return (frame_number, frame)

def show_video_frame(frame_label, frame_number, frame_data):
    # Display frame, and return displayed value
    frame_show   = frame_data.copy()
    cv.rectangle(frame_show, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(frame_show, str(frame_number), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    cv.imshow(frame_label, frame_show)
    return frame_show

def show_video_frame_mask(frame_label, frame_number, frame_data):
    # Display mask frame, and return displayed value
    frame_show   = cv.cvtColor(frame_data, cv.COLOR_GRAY2RGB)
    return show_video_frame(frame_label, frame_number, frame_show)

def show_video_frame_mask_centroids(frame_label, frame_number, frame_data, centroid_data):
    # Display mask frame with centroid data, and return displayed frame value
    frame_show = cv.cvtColor(frame_data, cv.COLOR_GRAY2RGB)
    # NOTE colour channels are B,G,R:
    # see https://docs.opencv.org/4.5.3/d8/d01/group__imgproc__color__conversions.html#ga397ae87e1288a81d2363b61574eb8cab
    colour_red = (0, 0, 255)
    for a in centroid_data:
        cx = round(a['xcen'])
        cy = round(a['ycen'])
        cr = math.ceil( math.sqrt((a['area'])) )
        cv.circle(frame_show, (cx,cy), cr, colour_red, thickness=2)
    return show_video_frame(frame_label, frame_number, frame_show)

def write_video_frame_pair(video_writer, frame_number, frame_1, frame_2):
    # Write a pair of frames side-by-side to a video output channel
    if frame_number > 75:       # @@@ Hack to keep demo videos shorter
        frame_pair = np.hstack((frame_1,frame_2))
        video_writer.write(frame_pair)
    else:
        frame_pair = None
    return frame_pair

def close_video_windows():
    cv.destroyAllWindows()
    return

# Video processing pipeline methods

def convert_frame_to_greyscale(frame):
    # Return fra,me converted to greyscale
    return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

def select_highlight_regions(frame):
    # Select bright highlight regions
    #
    # Returns frame of B/W pixels coded as 0 or 255
    #
    # Is this helpful?  Assuming not
    #??? frame_blur = cv.GaussianBlur(frame_grey, (7, 7), 0)

    threshold = 200
    high_out  = 255 
    T, frame_thresholded = cv.threshold(
        frame_grey, threshold, high_out, 
        # cv.THRESH_BINARY_INV
        cv.THRESH_BINARY
        )
    return frame_thresholded

def filter_region_coordinates(coord_list):
    def f(c):
        return (c['area'] >= 20) and (c['area'] <= 120)
        #   { xcen: (float),   ycen: (float),
        #     xmin: (integer), ymin: (integer),
        #     xmax: (integer), ymax: (integer),
        #     area: (integer) 
        #   }
    return list(filter(f, coord_list))

def calc_region_centroid_area(region_pixels):
    # Input:
    #
    #   { xmin, ymin, xmax, ymax, [pixelcoords] }
    #
    # Output:
    #
    #   { xcen: (float),   ycen: (float),
    #     xmin: (integer), ymin: (integer),
    #     xmax: (integer), ymax: (integer),
    #     area: (integer) 
    #   }
    #
    pcnt = len(region_pixels['pixelcoords'])   # Pixel count
    xsum = 0                                    # Sum of X coords
    ysum = 0                                    # Sum of Y coords
    # print(pcnt, region_pixels['xmin'], region_pixels['ymin'], region_pixels['xmax'], region_pixels['ymax'], region_pixels)
    for [px, py] in region_pixels['pixelcoords']:   # @@@ use reducer function here
        xsum += px
        ysum += py
    return (
        { 'xcen': xsum / pcnt,           'ycen': ysum / pcnt,
          'xmin': region_pixels['xmin'], 'ymin': region_pixels['ymin'],
          'xmax': region_pixels['xmax'], 'ymax': region_pixels['ymax'],
          'area': pcnt
        })

def get_region_coordinates(frame):
    # Returns list of coordinates, where each coordinate is:
    #
    #   { xcen: (float),   ycen: (float),
    #     xmin: (integer), ymin: (integer),
    #     xmax: (integer), ymax: (integer),
    #     area: (integer) 
    #   }
    global paused

    #  Get coords of non-zero pixels
    pixel_coords  = np.column_stack(np.where(frame))
    # print("@@@1", pixel_coords)

    # Merge non-zero pixels into regions.
    region_pixels = []  # (xmin, ymin, xmax, ymax, [pixelcoords] )
    for ipy,ipx in pixel_coords:
        region_numbers = []
        # print("@@@1", ipx, ipy)
        # print("@@@1", ipx, ipy, region_pixels)
        for nr in range(len(region_pixels)):
            r = region_pixels[nr]
            # print("@@@2", nr, r)
            merge = False
            if ( r and
                 (ipx >= r['xmin']-1) and (ipx <= r['xmax']+1) and
                 (ipy >= r['ymin']-1) and (ipy <= r['ymax']+1) ):
                # Look for adjacency:
                for rpx,rpy in r['pixelcoords']:
                    # print("@@@3", rpx, rpy)
                    if ( (ipx >= rpx-1) and (ipx <= rpx+1) and
                         (ipy >= rpy-1) and (ipy <= rpy+1) ):
                        # New pixel adjacent to existing region: merge in
                        if ipx < r['xmin']: r['xmin'] = ipx
                        if ipx > r['xmax']: r['xmax'] = ipx
                        if ipy < r['ymin']: r['ymin'] = ipy
                        if ipy > r['ymax']: r['ymax'] = ipy
                        merge = True
            if merge:
                r['pixelcoords'].append([ipx,ipy])
                region_numbers.append(nr)

        if len(region_numbers) == 0:
            # Create new region
            r = { 'xmin': ipx, 'xmax': ipx, 
                  'ymin': ipy, 'ymax': ipy, 
                  'pixelcoords': [[ipx, ipy]]
                }
            region_pixels.append(r)
        elif len(region_numbers) > 1:
            # Merge newly connected regions
            r = region_pixels[region_numbers[0]]
            # print("@@@4 (merge regions) ", region_numbers, r, ipx, ipy) 
            # paused = True
            for n in range(1, len(region_numbers)):
                r1 = region_pixels[region_numbers[n]]
                # print("@@@5                 ", n, r1) 
                if r1['xmin'] < r['xmin']: r['xmin'] = r1['xmin']
                if r1['xmax'] > r['xmax']: r['xmax'] = r1['xmax']
                if r1['ymin'] < r['ymin']: r['ymin'] = r1['ymin']
                if r1['ymax'] > r['ymax']: r['ymax'] = r1['ymax']
                r['pixelcoords'].extend(r1['pixelcoords'])
                region_pixels[region_numbers[n]] = None
            # Update first merged region
            region_pixels[region_numbers[0]] = r
        else:
            # pixel merged with exactly one existing region - nothing to do.
            pass

    # For each region, calculate centroid and area
    # See https://numpy.org/doc/stable/reference/generated/numpy.frompyfunc.html#numpy.frompyfunc
    #region_coords = np.frompyfunc(calc_region_centroid_area, 1, 1)(region_pixels)
    region_coords = []
    for n in range(len(region_pixels)):
        r = region_pixels[n]
        if r:
            region_coords.append(calc_region_centroid_area(r))

    return np.array(region_coords)



###### Main program ######

parser = argparse.ArgumentParser(description='This program uses OpenCV to read and process a video of \
                                              a Monotype system paper tape.')
parser.add_argument('--input', type=str, help='Path to a video of a Monotype system tape.', default='Monotype-tape.avi')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()

if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()

video_capture, frame_width, frame_height = open_video_input(
    cv.samples.findFileOrKeep(args.input)
    )
if not video_capture:
    print('Unable to open: ' + args.input)
    exit(0)
print("frame_height", frame_height, "frame_width", frame_width)

video_writer = open_video_output(
        'output.avi',                   # where
        frame_width*2, frame_height     # size
    )


#  Main frame-processing loop

paused = False
step   = False

while True:

    if not paused or step:
        # Read frame
        frame_number, frame = read_video_frame(video_capture)
        if frame is None:
            break

        # Display frame (copy)
        frame_show_orig = show_video_frame("Original video", frame_number, frame)

        # @@@@ Extract foreground from background?
        # outframe = cv.cvtColor(fgMask, cv.COLOR_GRAY2RGB)
        # fgMask   = backSub.apply(frame)
        # outframe = cv.copyTo(frame, mask=fgMask)

        # Convert to greyscale
        frame_grey = convert_frame_to_greyscale(frame)

        # np.set_printoptions(threshold=sys.maxsize)
        # for i in range(0,150):
        #     print("@@@0", i, frame_grey[i,450:460])

        frame_highlights = select_highlight_regions(frame_grey)

        # Display highlight regions
        # frame_show_highlights = show_video_frame_mask("Highlights", frame_number, frame_highlights)

        # Extract coordinates of higlight regions
        coord_list = get_region_coordinates(frame_highlights)

        # Filter out small and large highlight regions.  
        # Hopefully, just sprocket and data holes remain...
        filtered_coord_list = filter_region_coordinates(coord_list)

        # Display coords
        frame_show_coords = show_video_frame_mask_centroids(
            "Centroids", frame_number, frame_highlights, 
            filtered_coord_list
            )

        # Write the frame into the file 'output.avi'
        write_video_frame_pair(video_writer, frame_number, frame_show_orig, frame_show_coords)

        # Coalesce regions from successive frames as raw data for row-by-row data extraction

        print("Frame ", frame_number)
        for c in filtered_coord_list:
            print(c)
        # paused = True

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

# Shut down on exit

close_video_input(video_capture)
close_video_output(video_writer)
close_video_windows()

# End.
