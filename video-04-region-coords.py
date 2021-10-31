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
    return (math.ceil(frame_number), frame)

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
        frame, threshold, high_out, 
        # cv.THRESH_BINARY_INV
        cv.THRESH_BINARY
        )
    return frame_thresholded

def get_region_coordinates(frame_number, frame):
    # Returns list of coordinates, where each coordinate is:
    #
    #   { xcen: (float),   ycen: (float),
    #     xmin: (integer), ymin: (integer),
    #     xmax: (integer), ymax: (integer),
    #     area: (integer) 
    #   }
    global paused, step

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
                 (ipx >= r['xmin']-2) and (ipx <= r['xmax']+2) and
                 (ipy >= r['ymin']-1) and (ipy <= r['ymax']+1) ):
                # Look for adjacency:
                for rpx,rpy in r['pixelcoords']:
                    # print("@@@3", rpx, rpy)
                    if ( (ipx >= rpx-2) and (ipx <= rpx+2) and
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
            r = { 'fnum': frame_number,
                  'xmin': ipx, 'xmax': ipx, 
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

def filter_region_coordinates(coord_list):
    def f(c):
        return (c['area'] >= 12) and (c['area'] <= 150)
        #   { xcen: (float),   ycen: (float),
        #     xmin: (integer), ymin: (integer),
        #     xmax: (integer), ymax: (integer),
        #     area: (integer) 
        #   }
    return list(filter(f, coord_list))

def format_region_coords(coords):
    return (
        f"centroid ({coords['xcen']:6.1f}, {coords['ycen']:6.1f}), "
        f"X ({coords['xmin']:4d}..{coords['xmax']:<4d}), "
        f"Y ({coords['ymin']:4d}..{coords['ymax']:<4d}), "
        f"area {coords['area']:4d}"
        )

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
        { 'fnum': region_pixels['fnum'],
          'xcen': xsum / pcnt,           'ycen': ysum / pcnt,
          'xmin': region_pixels['xmin'], 'ymin': region_pixels['ymin'],
          'xmax': region_pixels['xmax'], 'ymax': region_pixels['ymax'],
          'area': pcnt
        })

# Row detector methods

def format_row_detect_buffer(buffer):
    return (
        f"ccmin {buffer['ccmin']:3d}, ccmax {buffer['ccmax']:3d}, "
        f"ccprv {buffer['ccprv']:3d}, ccdip {buffer['ccdip']:3d}, "
        f"endfr {buffer['endfr']:3d}"
        )

def row_detect_add_frame_coords(coord_list, buffer):
    # Add coordinate data for next frame to row detector buffer, and
    # returns coordinate count for the frame, and the updated buffer
    #
    # coord_list    is a list of region coordinate values for a single frame, where
    #               each such value consists of:
    #               { 'fnum': (int),    # Video frame number
    #                 'xcen': (float),  # X-coordinate of region centroid
    #                 'ycen': (float),  # Y-coordinate of region centroid
    #                 'xmin': (int),    # Min X-coordinate of region
    #                 'ymin': (int),    # Min Y-coordinate of region
    #                 'xmax': (int),    # Max X-coordinate of region
    #                 'ymax': (int),    # Max Y-coordinate of region
    #                 'area': (int)     # Ara (num pixels) of region
    #               }
    # buffer        is a row detection buffer, consisting of data accumulated over multiple 
    #               video frames, thus:
    #               { ccmin: (int),     # min region count seen for any frame so far
    #                 ccmax: (int),     # max region count seen for any frame so far
    #                 ccdip: (int),     # min region count seen following ccmax
    #                 endfr: (int),     # index of frame in buffer where ccdip seen
    #                 ccprv: (int),     # region count from previous frame
    #                 rcoords: []       # list of coord_list values for all frames
    #               }
    #
    buffer['rcoords'].append(coord_list)
    cc = len(coord_list)
    if cc < buffer['ccmin']:
        buffer['ccmin'] = cc
    if cc > buffer['ccmax']:
        buffer['ccmax'] = cc
    # Possible end of frame dip?
    if (cc < buffer['ccprv']):
        buffer['ccdip'] = cc
        buffer['endfr'] = len(buffer['rcoords'])
    buffer['ccprv'] = cc
    return cc, buffer

def row_detect_buffer_reset(frame_coords=[]):
    # Reset value for row buffer, taking account of supplied coordinate data
    # from frames after the previous row.
    #
    # frame_coords  a list of frame coordinate data that is used to initialize the
    #               buffer for the next frame.  Empty list if there is no left-over 
    #               frame data to use.  Leading empty frames in this data are skipped.
    #
    print(f"======= row_detect_buffer_reset: {len(frame_coords):d} frames")
    reset_buffer = (
        { 'ccmin': 99999,           # minimum frame coordinate count anywhere
          'ccmax': 0,               # max region count seen for row
          'ccdip': 99999,           # min region count seen after max
          'endfr': 0,               # buffer frame where ccdip seen
          'ccprv': 0,               # coordinate count from previous frame
          'rcoords': []             # list of region coordinates seen for row
        })
    first_frame = True
    for fr in frame_coords:
        # Skip multiple leading empty frames
        if fr or first_frame or (reset_buffer['ccmax'] > 0):
            first_frame = False
            cc, reset_buffer = row_detect_add_frame_coords(fr, reset_buffer)
            # print("=== row_detect_buffer_reset: Frames")
            # for r in fr:
            #     print("coords: ", format_region_coords(r))
            # print("buffer: ", format_row_detect_buffer(reset_buffer))
    return reset_buffer

def row_detect(coord_list, buffer):
    # Tape row detector.  Process the next frame of coordinates from the tape,
    # detect when a complete tape row has been scanned, and when detected, extract
    # and summarize coordinate data for that row.
    #
    # Looks for dip and rise of at least 2 region coordinates to signal end of row
    # NOTE: assumes fairly clean data.  If noise, may also need to take account of
    # total area of regions within bounding box?
    # 
    #   next_row, buffer = row_detect(coord_list, buffer)
    # 
    # coord_list    coordinate data from a single video frame, presented in sequence.
    # buffer        buffer for frame data from a single row of tape holes, or empty list.
    # next_row      summarized coordinate data returned when next tape row is detected, or
    #               None if there is no new row detected.
    # 

    # Update row detector statistics with next frame data
    cc, buffer = row_detect_add_frame_coords(coord_list, buffer)

    # Detect end of row - is coordinate count 2 up from most recent dip?
    end_of_row = (
        (cc >  buffer['ccdip'])   and               # Coord count is rising from dip
        (cc >= buffer['ccmin']+2) and               # Coord count is at least 2 above min
        (buffer['ccdip'] <= buffer['ccmax']-2) and  # Coord count has dipped from max
        (buffer['ccdip'] <= buffer['ccmin']+2)      # Coord count has dipped close to min
        )

    # Sort out result to return
    if end_of_row:
        #@@@
        # printbuffer = buffer.copy()
        # printbuffer['rcoords'] = printbuffer['rcoords'][buffer['endfr']-2:]
        # print("@@@end ", printbuffer)
        #@@@
        next_buffer = row_detect_buffer_reset(buffer['rcoords'][buffer['endfr']:])
        next_row    = dict(buffer, endfr=0, rcoords=buffer['rcoords'][:buffer['endfr']])
    else:
        next_row    = None
        next_buffer = buffer

    return (next_row, next_buffer)

def test_row_detect():
    buffer        = row_detect_buffer_reset([])
    frame_regions = (
        (79, [(( 461.4,  139.8), ( 460, 462 ), ( 136, 144 ),  18)]),
        (80, [(( 459.2,  141.6), ( 456, 462 ), ( 134, 149 ), 101),
              (( 455.0, 1144.7), ( 452, 458 ), (1138, 1152),  85)]),
        (81, [(( 457.0,  140.7), ( 455, 459 ), ( 135, 147 ),  54),
              (( 453.2, 1144.3), ( 451, 456 ), (1137, 1152),  80)]),
        (82, []),
        (83, [(( 459.1,  140.7), ( 458, 460 ), ( 137, 144 ),  21)]),
        (84, [(( 457.3,  141.6), ( 454, 460 ), ( 134, 150 ), 104),
              (( 453.6, 1145.4), ( 450, 457 ), (1138, 1153),  93)]),
        (85, [(( 457.7,  142.2), ( 455, 461 ), ( 135, 150 ),  97),
              (( 453.8, 1145.9), ( 451, 457 ), (1138, 1154), 106)]),
        (86, [(( 456.9,  142.0), ( 456, 458 ), ( 138, 146 ),  25),
              (( 453.6, 1145.8), ( 452, 455 ), (1141, 1151),  35)]),

        )
    for fr in frame_regions:
        frame_region_coords = [
            { 'fnum': fr[0],
              'xcen': r[0][0], 'ycen': r[0][1],
              'xmin': r[1][0], 'ymin': r[2][0],
              'xmax': r[1][1], 'ymax': r[2][1],
              'area': r[3]
            } for r in fr[1] ]
        next_row, buffer = row_detect(frame_region_coords, buffer)
        print("=== Frame ", fr[0])
        for r in frame_region_coords:
            print("coords: ", format_region_coords(r))
        print("buffer: ", format_row_detect_buffer(buffer))
        if next_row:
            print("=== Next row:")
            print(format_row_detect_buffer(next_row))
            print("===")

###### Main program ######

def main():
    global paused, step

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

    row_detect_buffer = row_detect_buffer_reset([])
    row_number        = 0

    while True:

        if not paused:
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
            coord_list = get_region_coordinates(frame_number, frame_highlights)
            # if frame_number == 97:
            #     print(f"==== Frame {frame_number:d} coords:")
            #     print("coord_list", coord_list)
            #     paused = True

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

            # print("Frame ", frame_number)
            # for c in filtered_coord_list:
            #     print("  ", format_region_coords(c))
            # # paused = True

            # Coalesce regions from successive frames as raw data for row-by-row data extraction

            next_row, row_detect_buffer = row_detect(filtered_coord_list, row_detect_buffer)
            if next_row:
                row_number += 1
                print("=== Next row", row_number, 
                    "min/max regions", next_row['ccdip'], next_row['ccmax'], "frames", len(next_row['rcoords']),
                    "===")
                cdec = False
                cinc = False
                cmin = 0
                cprv = 0
                for f in next_row['rcoords']:
                    if f:
                        print(f"  frame {f[0]['fnum']:5d}")
                        for c in f:
                            print("    ", format_region_coords(c))
                    else:
                        print("  empty frame")
                    # Double peak detector
                    cnxt = len(f)
                    if cnxt < cprv: 
                        cdec = True     # Decreased num coords
                        cmin = cnxt
                    if cnxt > cprv: 
                        cinc = cdec     # Increased num coords after decrease
                        cmin = cnxt
                    paused = cinc and (cprv <= cmin+2)
                    cprv = cnxt
                print("=== Next row start ===")
                for f in row_detect_buffer['rcoords']:
                    if f:
                        print(f"  frame {f[0]['fnum']:5d}")
                        for c in f:
                            print("    ", format_region_coords(c))
                    else:
                        print("  empty frame")
                print("=== Ends ===\n")

        # Check for keyboard interrupt, or move to next frame after 20ms
        paused   = paused or step
        keyboard = cv.waitKey(20)
        if keyboard != -1: 
        	print(keyboard)
        if keyboard == ord('p'):
            paused = True
            step   = False
        if keyboard == ord('r'):
            paused = False
            step   = False
        if keyboard == ord('s'):
            step   = True
            paused = False
        if keyboard == ord('q') or keyboard == 27:
            break

    # Shut down on exit

    close_video_input(video_capture)
    close_video_output(video_writer)
    close_video_windows()


# Top level script

paused = False
step   = False
main()
# test_row_detect()

# End.
