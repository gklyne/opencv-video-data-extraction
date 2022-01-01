# video-05-region-trace.py
#
# This version will take the detected region coordinates, 
# and build traces of regions that are detected in successive frames.
# It represents a departure from an earlier approach of trying to
# detect rows by finding frames where the transient regions (corresponding
# to tape holes) had all disappeared.  By focusing in traces, this new
# approach attempts avoid problems where new traces are starting to appear 
# before all previous traces have disappeared (e.g. due to skewing of 
# the tape/camera)

import cv2 as cv
import numpy as np
import argparse
import math
import statistics
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

def seek_video_frame(video_capture, frame_num):
    # Seek to specified video frame 
    #
    # returns True if the seek option is accepted accepted.
    return video_capture.set(cv.CAP_PROP_POS_FRAMES, frame_num)

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

def draw_region_centroids(frame_show, centroid_data):
    # NOTE colour channels are B,G,R:
    # see https://docs.opencv.org/4.5.3/d8/d01/group__imgproc__color__conversions.html#ga397ae87e1288a81d2363b61574eb8cab
    colour_red = (0, 0, 255)
    for a in centroid_data:
        cx = round(a['xcen'])
        cy = round(a['ycen'])
        cr = math.ceil( math.sqrt((a['area'])) )
        cv.circle(frame_show, (cx,cy), cr, colour_red, thickness=2)
    return

def show_video_frame_mask_centroids(frame_label, frame_number, frame_data, centroid_data):
    # Display mask frame with centroid data, and return displayed frame value
    frame_show = cv.cvtColor(frame_data, cv.COLOR_GRAY2RGB)
    draw_region_centroids(frame_show, centroid_data)
    # # NOTE colour channels are B,G,R:
    # # see https://docs.opencv.org/4.5.3/d8/d01/group__imgproc__color__conversions.html#ga397ae87e1288a81d2363b61574eb8cab
    # colour_red = (0, 0, 255)
    # for a in centroid_data:
    #     cx = round(a['xcen'])
    #     cy = round(a['ycen'])
    #     cr = math.ceil( math.sqrt((a['area'])) )
    #     cv.circle(frame_show, (cx,cy), cr, colour_red, thickness=2)
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
        return (c['area'] >= 12) # and (c['area'] <= 150)
    return list(filter(f, coord_list))

def format_region_coords(coords):
    return (
        f"centroid ({coords['xcen']:6.1f}, {coords['ycen']:6.1f}), "
        f"X ({coords['xmin']:4d}..{coords['xmax']:<4d}), "
        f"Y ({coords['ymin']:4d}..{coords['ymax']:<4d}), "
        f"area {coords['area']:4d}"
        )

# Region trace methods
#
# region_coords: region in single frame (from above):
#
#       { 
#         xcen: (float),   ycen: (float),
#         xmin: (integer), ymin: (integer),
#         xmax: (integer), ymax: (integer),
#         area: (integer) 
#       }
#
# frame_coords:   list of region coordinates within a single frame.
#
#       [ (region_coords) ]
#
# region_trace: trace of region through several frames
#
#       { 
#         frnum:   (integer)            # Start frame number
#         frend:   (integer)            # End frame number, or -1 if still open
#         rcoords: [(region_coords)]    # region coords for each fame in range
#                                       # - index +frnum to get actual frame number
#         # Summary data added after trace is ended:
#         frlen:    (integer),          # Overall frame extent (length) of trace 
#         xcen:     (float),            # Mean X position
#         ycen:     (float),            # Mean Y position 
#         xmin:     (integer),          # Overall min X coord
#         ymin:     (integer),          # Overall min Y coord
#         xmax:     (integer),          # Overall max X coord
#         ymax:     (integer),          # Overall max Y coord
#         area:     (integer),          # Total area summed over all frames
#       }
#
# NOTE: end frame is frame *after* the last frame of the trace.
#
# @@ ??? possible add mean position/total area/average area in frame for closed traces
#
# open_region_traces:
#
#       [ (region_trace) ]              # Arbitrary ordering (could be set, bag)
#
# closed_region_traces:
#
#       [ (region_trace) ]              # Ordered by ending frame number
#

def region_overlaps(rc, rtrace):
    # Determine if given region coords 'rc' overlaps with the region trace 'rtrace'.
    #
    if len(rtrace) == 0:
        return False
    rx = round(rc['xcen'])
    ry = round(rc['ycen'])
    r1 = rtrace['rcoords'][-1]         # Last position in trace
    r1x = round(r1['xcen'])
    r1y = round(r1['ycen'])
    return ( ( (rx >= r1['xmin']) and (rx <= r1['xmax']) and
               (ry >= r1['ymin']) and (ry <= r1['ymax']) ) or
             ( (r1x >= rc['xmin']) and (r1x <= rc['xmax']) and
               (r1y >= rc['ymin']) and (r1y <= rc['ymax']) ) 
           )


# Add a new frame of region coordinates to the currently open traces.
# Any traces that are terminated in this frame are returned as a separate
# list of closed traces.
#
# frnum         is the frame number for the supplied region coordinates
# frame_coords  is a list of region coordinates detected in the current frame
# open_traces   is a buffer of detected region traces that are still active in
#               the preceding frame.
#
# ending_traces, open_traces = region_trace_detect(frnum, frame_coords, open_traces)
#
def region_trace_detect(frnum, frame_coords, open_traces):
    new_traces = []
    for rt in open_traces:
        rt['frend'] = frnum                 # Assume closed unless extension seen
    for rc in frame_coords:
        for rt in open_traces:
            if region_overlaps(rc, rt):
                rt['rcoords'].append(rc)    # Extend trace
                rt['frend'] = -1            # Flag trace still open
                break # to next rc
        else:
            nt = { 'frnum': frnum, 'frend': -1, 'rcoords': [rc] }
            new_traces.append(nt)
    # Separate out open/closed traces for result
    ongoing_traces = [ rt for rt in open_traces if rt['frend'] < 0 ] + new_traces
    ending_traces  = [ rt for rt in open_traces if rt['frend'] > 0 ]
    return (ending_traces, ongoing_traces)

# Add summary data to region traces.  This is used to add trace summary data when
# newly ended traces are encountered
#
# rtraces   region traces to be summarized,  For each trace, defined fields are:
#               frnum - start frame number
#               frend - end frame number (NOTE: only closed traces should be provided)
#               rcoords - per-frame region coordinates
#
# Returns list of summarized traces
#
# rtraces = region_trace_summary(rtraces)
#
def region_trace_summary(rtraces):
    for rt in rtraces:
        assert (rt['frend'] > 0), "Can't summarize open region trace"
        rt['frlen'] = rt['frend'] - rt['frnum']
        rt['xcen']  = statistics.fmean([ rc['xcen'] for rc in rt['rcoords'] ])
        rt['ycen']  = statistics.fmean([ rc['ycen'] for rc in rt['rcoords'] ])
        rt['xmin']  = min([ rc['xmin'] for rc in rt['rcoords'] ]) 
        rt['ymin']  = min([ rc['ymin'] for rc in rt['rcoords'] ]) 
        rt['xmax']  = max([ rc['xmax'] for rc in rt['rcoords'] ]) 
        rt['ymax']  = max([ rc['ymax'] for rc in rt['rcoords'] ]) 
        rt['area']  = sum([ rc['area'] for rc in rt['rcoords'] ])
    return rtraces

# Add newly ending traces to buffer of closed traces.
#
# frnum         is the end frame number for the newly ended traces
# ending_traces is a list of newly ended traces
# closed_traces is a buffer of closed traces that is updated by this method.
#
# The closed traces buffer is maintained in order of ending frame number.  
# This is currently achieved by assuming that newly closed traces are presented
# in order of the closing frame number as a result of sequential processing of
# video data.
#
# closed_traces = region_trace_add(frame_number, ending_traces, closed_traces)
#
def region_trace_add(frnum, ending_traces, closed_traces):
    closed_traces.extend(ending_traces)
    return closed_traces


def draw_region_traces(frame_show, frnum, traces):
    # NOTE colour channels are B,G,R:
    # see https://docs.opencv.org/4.5.3/d8/d01/group__imgproc__color__conversions.html#ga397ae87e1288a81d2363b61574eb8cab
    colour_green = (0, 255, 0)
    for rt in traces:
        w    = (rt['frend'] - rt['frnum'])*5
        h    = rt['area'] / w
        xoff = (rt['frend'] - frnum)*5
        if (w < 1000) and (rt['xcen'] + xoff > 0):
            x1   = round(rt['xcen'] - w + xoff)
            y1   = round(rt['ycen'] - h/2)
            x2   = round(rt['xcen'] + xoff)
            y2   = round(rt['ycen'] + h/2)
            cv.rectangle(frame_show, (x1,y1), (x2, y2), colour_green, thickness=2)
    return


# show_region_traces(frame_label, frame_number, frame_data, rcoords, rtraces)
#
# frame_label   is a display label for the presented video frame
# frame_number  is a frame number to be displayed
# frame_data    is raw video data of the mask frame to be displayed
# rcoords       is the region coordinates to be displayed for the current frame
# rtraces       is a list of closed region traces to be displayed
#
# Returns displayed video frame
#
def show_video_frame_mask_region_traces(frame_label, frame_number, frame_data, rcoords, rtraces):
    # Display mask frame with centroid data, and return displayed frame value
    frame_show = cv.cvtColor(frame_data, cv.COLOR_GRAY2RGB)
    draw_region_centroids(frame_show, rcoords)
    draw_region_traces(frame_show, frame_number, rtraces)
    return show_video_frame(frame_label, frame_number, frame_show)

# Log list of region traces to console
#
def log_region_traces(frnum, traces):
    print(f"frame {frnum:d}")
    for rt in traces:
        print(f"frnum {rt['frnum']:4d}, frend {rt['frend']:4d}")
        for rc in rt['rcoords']:
            print("  "+format_region_coords(rc))
    return


# Row methods
#
# A row is a set of closed regions that are assessed as representing a row of holes
# in the tape.
#
# Representation for a closed region
#       { 
#         frnum:   (integer)            # Start frame number
#         frend:   (integer)            # End frame number, or -1 if still open
#         rcoords: [(region_coords)]    # region coords for each fame in range
#                                       # - index +frnum to get actual frame number
#         frlen:    (integer),          # Overall frame extent (length) of trace 
#         xcen:     (float),            # Mean X position
#         ycen:     (float),            # Mean Y position 
#         xmin:     (integer),          # Overall min X coord
#         ymin:     (integer),          # Overall min Y coord
#         xmax:     (integer),          # Overall max X coord
#         ymax:     (integer),          # Overall max Y coord
#         area:     (integer),          # Total area summed over all frames
#         # Added when a row is detected
#         hpos:     (float)             # Hole position ('k' in constraints below)
#       }
#
# A row is a (maximal) set of regions (approximately) satisfying the followjng constraints:
#
# (P[xcen], P[ycen]) = (xbase + k*xspan, ybase + k*yspan); 0 <= k <= 1
#   - for all regions P in a row, where xbase, ybase, xspan, yspan are the same for each such P.
#     This effectively requires that the regions in a row sit on a line, at multiples of some unit 
#     separation distance.  
#     If (xbase,ybase) represents one of the sprocket holes, then (xspan,yspan) represents 
#     the offset to the sprocket hole on the other side of the tape.  
#     NOTE: sprocket holes are approx 5mm from data holes; data hole spacing is approx 3mm.
#
# P[frbeg] < Q[frend] 
#   - for all regions P and Q in a row.  
#     No region in a row can start after another has ended.
#     The intent is that this provides a basis for determining that no further regions can be
#     presented, assuming that regions are presented in order of starting or ending frame number.
#
# P[frlen] < 2*Q[frlen]
#   - for all regions P and Q in a row.  
#     No region is more than twice the duration of any other region in a row.
#     The intent is to eliminate extended regions that don't correspond to tape holes, based
#     on the assumption that all regions in a row will be of similar duration.
#     @@NOTE: this might cause problems if some genuine regions are very short; 
#     e.g. if tape is moved very fast.  The constraint might be adjusted to limit the length 
#     of the longest region with respect to the average of all regions in the row.
#
# These constraints are designed to allow a RANSAC-like algorithm [1] to be used to separate real
# data from noise (regions not corresponding to tape holes), without making assumptions about the
# camera orientation (may be skewed) or speed at which the tape is being drawn over the reader bar.
# In this application it is possible that the total number of regions under consideration at any time 
# is small enough to allow an exhaustive search rather than random sampling.
#
# [1] https://en.wikipedia.org/wiki/Random_sample_consensus
#
# ----------


# Look for completed row of tape data in trace data.
#
# Representation of row parameters:
#       {
#         frbegmin: (integer)               # Minimum (first) region begin frame number
#         frbegmax: (integer)               # Maximum (final) region begin frame number
#         frendmin: (integer)               # Minimum (first) region end frame number
#         frendmax: (integer)               # Maximum (final) region end frame number
#         frlenmin: (integer)               # Minimum region frame length
#         frlenmax: (integer)               # Maximum region frame length
#         frlenave: (float)                 # Average region frame length
#         xbase:    (float)                 # Base (hole 0) X-coordinate
#         ybase:    (float)                 # Base (hole 0) Y-coordinate
#         xspan:    (float)                 # Sprocket hole-to-hole X-offset
#         yspan:    (float)                 # Sprocket hole-to-hole Y-offset
#         ...
#       }
#
# Representation of a row of tape data:
#       {
#         params:   row_params              # (See above)
#         regions:  [(closed_region)]       # List of closed regions corresponding to row data (see above)
#       }
#
#
# frnum                 frame number currently processed.  No new regions may end before this frame number.
# row_params            row parameters from recently detected rows
# trace_data            is a list of closed region traces, ordered by ending frame number, which have not yet
#                       been definitively determined as belonging to a completed row, or no row at all.
#
# Returns:
#
# new_rows              is a list of new completed rows, or empty if none can be determined from
#                       the supplied trace data.
# updated_row_params    updated row parameters
# updated_trace_data    is the supplied trace data excluding any region traces that have been detected 
#                       as belonging to a completed row, or determined to be not belonging to any row.
#
def trace_row_detect(frnum, trace_data):
    #@@@@
    return (new_row, updated_trace_data)


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

    open_traces   = []
    closed_traces = []

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
            # frame_show_coords = show_video_frame_mask_centroids(
            #     "Centroids", frame_number, frame_highlights, 
            #     filtered_coord_list
            #     )

            # print("Frame ", frame_number)
            # for c in filtered_coord_list:
            #     print("  ", format_region_coords(c))
            # # paused = True

            # Coalesce regions from successive frames into region traces
            new_traces, open_traces = region_trace_detect(frame_number, filtered_coord_list, open_traces)
            new_traces = region_trace_summary(new_traces)
            closed_traces = region_trace_add(frame_number, new_traces, closed_traces)

            # Show closed region traces
            log_region_traces(frame_number, new_traces)
            frame_show_traces = show_video_frame_mask_region_traces(
                "Traces", frame_number, frame_highlights, 
                filtered_coord_list, closed_traces
                )

            # Write the frame into the file 'output.avi'
            write_video_frame_pair(video_writer, frame_number, frame_show_orig, frame_show_traces)

        # Check for keyboard interrupt, or move to next frame after 20ms
        paused   = paused or step
        keyboard = cv.waitKey(20)
        if keyboard != -1: 
        	print(keyboard)
        if keyboard == ord('p'):
            # Pause
            paused = True
            step   = False
        if keyboard == ord('r'):
            # Resume
            paused = False
            step   = False
        if keyboard == ord('s'):
            # Single-step frame
            step   = True
            paused = False
        if keyboard == ord('q') or keyboard == 27:
            # Quit
            break

    # Shut down on exit

    print("closing down")
    close_video_input(video_capture)
    close_video_output(video_writer)
    print("video streams closed")
    close_video_windows()
    print("end of main")

# Top level script

paused = False
step   = False
main()
# test_row_detect()

# End.
