# video-06-row-detect.py
#
# This version will build highlighted region traces over multiple video frames, then 
# group them into rows corresponding to holes in a single row on the Monotype system tape.

import cv2 as cv
import numpy as np
import argparse
import math
import statistics
import functools
import sys

# Log/debug functions

def log_error(*args):
    print("ERROR", *args)

def log_warning(*args):
    print("WARNING", *args)

def log_info(*args):
    print(*args)

def log_debug(*args):
    print(*args)
    pass


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
    for a in centroid_data:
        print(str(a))
        a.draw(frame_show)
    return

def show_video_frame_mask_centroids(frame_label, frame_number, frame_data, centroid_data):
    # Display mask frame with centroid data, and return displayed frame value
    frame_show = cv.cvtColor(frame_data, cv.COLOR_GRAY2RGB)
    draw_region_centroids(frame_show, centroid_data)
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


# Video processing pipeline classes and functions

# ----------------------------------------- #
# -- region_frame_pixels ------------------ #
# ----------------------------------------- #

class region_frame_pixels(object):
    """
    A collection of video frame pixels corresponding to a highlighted region in the frame.
    Some (most?) such regions correspond to a hole in the tape being read.
    """

    __slots__ = "fnum", "xmin", "ymin", "xmax", "ymax", "pixelcoords"   # Optimization

    def __init__(self, fnum, xmin, ymin, xmax, ymax, pixelcoords):
        self.fnum = fnum
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.pixelcoords = pixelcoords  
        return

    def __str__(self):
        return (
            f"Pixels frame {self.fnum:4d}"
            f", X ({self.xmin:4d}..{self.xmax:<4d}), "
            f", Y ({self.ymin:4d}..{self.ymax:<4d}), "
            )


def convert_frame_to_greyscale(frame):
    """
    Return video frame converted to greyscale
    """
    return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

def select_highlight_regions(frame):
    """
    Select bright highlight regions
    
    Returns frame of B/W pixels coded as 0 or 255
    
    Is this helpful?  Assuming not
    ??? frame_blur = cv.GaussianBlur(frame_grey, (7, 7), 0)
    """
    threshold = 200
    high_out  = 255 
    T, frame_thresholded = cv.threshold(
        frame, threshold, high_out, 
        # cv.THRESH_BINARY_INV
        cv.THRESH_BINARY
        )
    return frame_thresholded

def find_frame_regions(frame_number, frame):
    """
    Finds distinct highlight regions in a video frame.
    
    frame_number      is the video frame number being processed (saved with each region)
    frame             is frame of binary pixels of highlights (see `select_highlight_regions`),
                      in the form of a OpenCV frame (NumPy array) of zero or 255 pixels.
    
    Returns:          a list of `region_frame_pixels` values, one for each region detected.
    """

    global paused, step     # May be used to pause video scan for debugging

    #  Get coords of non-zero pixels
    pixel_coords  = np.column_stack(np.where(frame))
    # log_debug("@@@1", pixel_coords)

    # Merge non-zero pixels into regions.
    region_pixels_list = []              # [region_frame_pixels]
    for ipy,ipx in pixel_coords:

        region_numbers = []         # List of region numbers adjoining next pixel
        # log_debug("@@@1", ipx, ipy)
        # log_debug("@@@1", ipx, ipy, region_pixels_list)
        for nr, r in enumerate(region_pixels_list):
            # log_debug("@@@2", nr, r)

            # Find regions that adjoin new pixel
            merge = False
            if ( r and
                 (ipx >= r.xmin-2) and (ipx <= r.xmax+2) and
                 (ipy >= r.ymin-1) and (ipy <= r.ymax+1) ):
                # Look for adjacency to actual pixel:
                for rpx,rpy in r.pixelcoords:
                    # log_debug("@@@3", rpx, rpy)
                    if ( (ipx >= rpx-2) and (ipx <= rpx+2) and
                         (ipy >= rpy-1) and (ipy <= rpy+1) ):
                        # New pixel adjacent to existing region: merge in
                        if ipx < r.xmin: r.xmin = ipx
                        if ipx > r.xmax: r.xmax = ipx
                        if ipy < r.ymin: r.ymin = ipy
                        if ipy > r.ymax: r.ymax = ipy
                        merge = True
                if merge:
                    r.pixelcoords.append((ipx,ipy))
                    region_numbers.append(nr)

        # If no adjoining region, create new region
        if len(region_numbers) == 0:
            r = region_frame_pixels(frame_number, ipx, ipy, ipx, ipy, [(ipx, ipy)])
            region_pixels_list.append(r)

        # If multiple adjoining regions, merge them all
        elif len(region_numbers) > 1:
            # Merge newly connected regions
            r = region_pixels_list[region_numbers[0]]
            # log_debug("@@@4 (merge regions) ", region_numbers, r, ipx, ipy) 
            # paused = True
            for n in range(1, len(region_numbers)):
                r1 = region_pixels_list[region_numbers[n]]
                # log_debug("@@@5                 ", n, r1) 
                if r1.xmin < r.xmin: r.xmin = r1.xmin
                if r1.xmax > r.xmax: r.xmax = r1.xmax
                if r1.ymin < r.ymin: r.ymin = r1.ymin
                if r1.ymax > r.ymax: r.ymax = r1.ymax
                r.pixelcoords.extend(r1.pixelcoords)
                region_pixels_list[region_numbers[n]] = None
            region_pixels_list[region_numbers[0]] = r

        # If pixel merged with exactly one existing region - nothing more to do.
        else:
            pass

    return filter(None, region_pixels_list)

# ----------------------------------------- #
# -- region_frame_centroid ---------------- #
# ----------------------------------------- #

class region_frame_centroid(object):
    """
    Represents the calculated centroid of a region in a single frame
    """

    __slots__ = "area", "xcen", "ycen", "rpixels"

    def __init__(self, region_pixels):
        # self.area    = 0
        # self.xcen    = 0.0
        # self.ycen    = 0.0
        # self.rpixels = None      # region_frame_pixels value (see above)
        self.calc_centroid_area(region_pixels)
        return

    def draw(self, frame):
        """
        Draw centroid in supplied video frame buffer
        """
        # NOTE colour channels are B,G,R:
        # see https://docs.opencv.org/4.5.3/d8/d01/group__imgproc__color__conversions.html#ga397ae87e1288a81d2363b61574eb8cab
        colour_red = (0, 0, 255)
        cx = round(self.xcen)
        cy = round(self.ycen)
        cr = math.ceil( math.sqrt(self.area) )
        cv.circle(frame, (cx,cy), cr, colour_red, thickness=2)
        return

    def __str__(self):
        return (
            f"region_frame_centroid ({self.xcen:6.1f}, {self.ycen:6.1f})"
            f", area {self.area:4d}"
            f", X ({self.rpixels.xmin:4d}..{self.rpixels.xmax:<4d})"
            f", Y ({self.rpixels.ymin:4d}..{self.rpixels.ymax:<4d})"
            )

    def calc_centroid_area(self, region_pixels):
        # Calculates centroid and area of a region (marker, hole) in a single frame
        #
        # region_pixels     is a `region_frame_pixels` value with the pixels from which
        #                   the centroid and area are calculated.
        #
        # Returns:          reference to the resulting `region_frame_centroid` object
        #
        pcnt = len(region_pixels.pixelcoords)   # Pixel count
        xsum = 0                                # Sum of X coords
        ysum = 0                                # Sum of Y coords
        # log_debug(
        #     "@@@ region_frame_centroid.calc_centroid_area", 
        #     pcnt, 
        #     region_pixels.xmin, region_pixels.ymin, 
        #     region_pixels.xmax, region_pixels.ymax, 
        #     region_pixels
        #     )
        def vecsum(p1, p2):
            x1, y1 = p1
            x2, y2 = p2
            return (x1+x2, y1+y2)
        xsum, ysum   = functools.reduce(vecsum, region_pixels.pixelcoords)
        self.area    = pcnt
        self.xcen    = xsum/pcnt
        self.ycen    = ysum/pcnt
        self.rpixels = region_pixels
        return self

    def get_area(self):
        return self.area

    def get_centroid(self):
        return (self.xcen, self.ycen)

    def regions_overlap(self, r2):
        """
        Test for supplied region overlaps with current region.

        Overlap occurs when the centroid of one region lies within the range of the other.
        """
        r1x = round(self.xcen)
        r1y = round(self.ycen)
        r2x = round(r2.xcen)
        r2y = round(r2.ycen)
        return ( ( (r2x >= self.rpixels.xmin) and (r2x <= self.rpixels.xmax) and
                   (r2y >= self.rpixels.ymin) and (r2y <= self.rpixels.ymax) ) or
                 ( (r1x >= r2.rpixels.xmin)   and (r1x <= r2.rpixels.xmax) and
                   (r1y >= r2.rpixels.ymin)   and (r1y <= r2.rpixels.ymax) ) 
               )


def find_frame_region_coords(region_pixels_list):
    """
    For each in a list of region pixels, calculate region centroid and area
    """
    region_coords = map(region_frame_centroid, region_pixels_list)
    return region_coords

def filter_region_coordinates(region_centroids_list):
    """
    Filter a list of region centroids to eliminate those that are unsuitable for further
    consideration (e.g., too small to be a detected tape hole).
    """
    def f(c):
        return c and (c.area >= 12) # and (c['area'] <= 150)
    return list(filter(f, region_centroids_list))

# ----------------------------------------- #
# -- region_trace ------------------------- #
# ----------------------------------------- #

class region_trace(object):
    """
    Base class for open- and closed- region traces.

    A region trace represents a single region across multiple video frames.
    """

    def __init__(self, frnum, rcoords):
        self.frnum   = frnum        # Start frame number
        self.rcoords = rcoords      # list of `region_frame_centroid`s for each fame in range
                                    # - index+frnum for video frame number
        return

class region_trace_open(region_trace):
    """
    Represents a (open, hence mutable) region traced through multiple frames
    """

    def __init__(self, frnum, rcoords):
        super(region_trace_open, self).__init__(frnum, rcoords)
        self.closing = False
        return

    def region_overlaps_trace(self, rc):
        """
        Determine if given region 'rc' overlaps with the current region trace

        rc              a `region_frame_centroid` value that is to be tested.

        returns:        True if the supplied region coordinates overlap the
                        final frame of the current region trace.
        """
        # log_debug("@@@ region_overlaps_trace ", self.frnum, len(self.rcoords))
        if len(self.rcoords) == 0:
            return False
        rt  = self.rcoords[-1]          # Last position in trace
        return rt.regions_overlap(rc)

    def start_next_frame(self, frnum):
        """
        Start processing for next frame
        """
        # log_debug("@@@ start_next_frame ", frnum, len(self.rcoords))
        self.closing = True     # Default closing unless extend_trace called
        return

    def extend_trace(self, frnum, rc):
        """
        Extend trace with region from next frame
        """
        # log_debug("@@@ extend_trace ", frnum, len(self.rcoords))
        self.rcoords.append(rc)
        self.closing = False
        return

    def trace_closing(self):
        """
        After next frame has been processed, test if trace is closing
        """
        return self.closing


class region_trace_closed(region_trace):
    """
    Represents a (closed, hence immutable) region traced through multiple frames
    """

    def __init__(self, frend, trace_closing):
        """
        Create a closed trace from a closing open trace.

        frend           is the frame number *after* the last frame of the closing trace
        trace_closing   is an open trace that is being closed off
        """
        super(region_trace_closed, self).__init__(trace_closing.frnum, trace_closing.rcoords)
        self.frend = frend                          # End frame number
        self.frlen = frend - trace_closing.frnum    # Overall frame extent (length) of trace 
        self.xcen  = statistics.fmean( (rc.xcen for rc in trace_closing.rcoords) )
        self.ycen  = statistics.fmean( (rc.ycen for rc in trace_closing.rcoords) )
        self.xmin  = min( (rc.rpixels.xmin for rc in trace_closing.rcoords) )
        self.ymin  = min( (rc.rpixels.ymin for rc in trace_closing.rcoords) )
        self.xmax  = max( (rc.rpixels.xmax for rc in trace_closing.rcoords) )
        self.ymax  = max( (rc.rpixels.ymax for rc in trace_closing.rcoords) )
        self.area  = sum( (rc.area         for rc in trace_closing.rcoords) )
        # log_debug("@@@ region_trace_closed.__init__", str(self))
        return

    def __str__(self):
        return (
            f"region_trace: frnum {self.frnum:4d}, frend {self.frend:4d}"
            f", area {self.area}"
            f", cent ({self.xcen},{self.ycen})"
            )

    def format_frame_coords(self, prefix):
        """
        Generator returns formatted region coordinates for frames in trace
        """
        for rc in self.rcoords:
            yield (prefix+str(rc))
        return

    def draw(self, frame_show, frnum):
        """
        Draw trace in supplied video frame buffer

        The trace is offset by the currenmt frame number so that it is moved to
        the left when drawn in later frames
        """
        # NOTE colour channels are B,G,R:
        # see https://docs.opencv.org/4.5.3/d8/d01/group__imgproc__color__conversions.html#ga397ae87e1288a81d2363b61574eb8cab
        colour_green = (0, 255, 0)
        w    = (self.frend - self.frnum)*5      # Width of displayed trace
        h    = self.area / w                    # Height of displayed trace
        xoff = (self.frend - frnum)*5           # Offset position of displayed trace
        # Display on;ly if fits in frame
        if (w < 1000) and (self.xcen + xoff > 0):
            x1   = round(self.xcen - w + xoff)
            y1   = round(self.ycen - h/2)
            x2   = round(self.xcen + xoff)
            y2   = round(self.ycen + h/2)
            cv.rectangle(frame_show, (x1,y1), (x2, y2), colour_green, thickness=2)
        return


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
        rt.start_next_frame(frnum)
    for rc in frame_coords:
        for rt in open_traces:
            if rt.region_overlaps_trace(rc):
                rt.extend_trace(frnum, rc)
                break # to next rc
        else:
            nt = region_trace_open(frnum, [rc])
            new_traces.append(nt)
    # Separate out ongoing/closed traces for result
    closed_traces  = [ region_trace_closed(frnum, rt) for rt in open_traces if rt.trace_closing() ]
    ongoing_traces = [ rt for rt in open_traces if not rt.trace_closing() ] + new_traces
    return (closed_traces, ongoing_traces)

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
    for rt in traces:
        rt.draw(frame_show, frnum)
    return

# show_video_frame_mask_region_traces(frame_label, frame_number, frame_data, rcoords, rtraces)
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
def log_region_traces(frnum, traces):
    log_info("log_region_traces ", frnum)
    for rt in traces:
        log_info(str(rt))
        for rc_str in rt.format_frame_coords("  "):
            log_info(rc_str)
    return

# Row methods
#
# A row is a set of closed traces that are assessed as representing a row of holes
# in the tape.
#
# Representation for a closed trace
#       { 
#         frnum:   (integer)            # Start frame number
#         frend:   (integer)            # End frame number, or -1 if still open
#         rcoords: [(region_coords)]    # region coords for each region in trace
#                                       # - index +frnum to get actual frame number
#         frlen:    (integer),          # Overall frame extent (length) of trace 
#         xcen:     (float),            # Mean X position
#         ycen:     (float),            # Mean Y position 
#         xmin:     (integer),          # Overall min X coord
#         ymin:     (integer),          # Overall min Y coord
#         xmax:     (integer),          # Overall max X coord
#         ymax:     (integer),          # Overall max Y coord
#         area:     (integer),          # Total area summed over all frames
#         # Added when/after a row is detected?
#         hpos:     (float)             # Hole position ('k' in constraints below)
#       }
#
# A row is a (maximal) set of traces (approximately) satisfying the followjng constraints:
#
# [A]: P[frbeg] < Q[frend] 
#   - for all traces P and Q in a row.  
#     No trace in a row can start after another has ended; all traces in a row overlap 
#     in at least one frame.
#     This is used as a basis for determining that no further traces can be added to
#     to a row, assuming that traces are presented in order of (ending) frame number.
#
# [B1]: P[frlen] < 2*Median(Q[frlen])
#   - for all traces P and Q in a row.  
#     No trace may be longer than twice the median length of other traces in a row.
#     The intent is to eliminate extended traces that don't correspond to tape holes, 
#     based on the assumption that all traces in a row will be of similar duration.
#
# [B2]: P[frlen] < 2*Q[frlen]+FLA
#   - for all traces P and Q in a row.  
#     While row data is incomplete, the median length of traces cannot be known.
#     This constraint provides a very rough estimator for assessing whether there might be
#     more traces to add, using the minimum trace length seen as a lower bound for the median.
#   
#     The "+FLA" is to avoid false row endings when a very short trace is present: the assumption here
#     is that if an exceptional short trace is present, more representative traces will be encountered 
#     within the FL additional frames.  This effectively forces a degree of lookahead in the incoming
#     traces before declaring that a row is complete.
#
# [C]: each row contains least 3 traces.
#     A valid Monotype tape row has at least 4 holes: 2 sprocket holes and 2 data holes.  The minimum
#     of 3 allows for one of the sprocket holes to be missed.  Any putative row with fewer than 3 traces
#     should be logged and disregarded.
#
# [D]: (P[xcen], P[ycen]) = (xbase + k*xspan, ybase + k*yspan); 0 <= k <= 1
#   - for all traces P in a row, where xbase, ybase, xspan, yspan are the same for each such P.
#     This effectively requires that the traces in a row sit on a line, at multiples of some unit 
#     separation distance.  
#     If (xbase,ybase) represents one of the sprocket holes, then (xspan,yspan) represents 
#     the offset to the sprocket hole on the other side of the tape.  
#     NOTE: sprocket holes are approx 5mm from data holes; data hole spacing is approx 3mm.
#
# These constraints are designed to allow a RANSAC-like algorithm [1] to be used to separate real
# data from noise (traces not corresponding to tape holes), without making assumptions about the
# camera orientation (may be skewed) or speed at which the tape is being drawn over the reader bar.
# In this application it is possible that the total number of traces under consideration at any time 
# is small enough to allow an exhaustive search rather than random sampling.
#
# [1] https://en.wikipedia.org/wiki/Random_sample_consensus
#
# ----------

# Find overlapping and non-overlapping traces
#
# trace                 a single trace to be tested
# trace_set             a set of traces with which the supplied trace is compared
#
# Returns:  (trace_inc, trace_exc)
#
# trace_inc             list of elements in `trace_set` that overlaps with `trace`
# trace_exc             list of elements in `trace_set` that does not overlap with `trace`
#
def find_trace_overlaps(trace, trace_set):
    def trace_overlaps(t1, t2):
        # NOTE: end frame is *after* last frame of trace
        return (t1.frnum < t2.frend) or (t2.frnum < t1.frend)
    ts_inc = []
    ts_exc = []
    # print("@@@ find_trace_overlaps for", str(trace))
    for t in trace_set:
        if trace_overlaps(t, trace):
            # print("@@@ overlaps  ", str(t))
            ts_inc.append(t)
        else:
            # print("@@@ no overlap", str(t))
            ts_exc.append(t)
    # print("@@@ ts_inc", ts_inc)
    # print("@@@ ts_exc", ts_exc)
    return (ts_inc, ts_exc)

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
#         # For row parameters carried from row-to-row, the following are derived over recent rows:
#         frlenave: (float)                 # Average region frame length
#         xbase:    (float)                 # Base (hole 0) X-coordinate
#         ybase:    (float)                 # Base (hole 0) Y-coordinate
#         xspan:    (float)                 # Sprocket hole-to-hole X-offset (extent of row holes)
#         yspan:    (float)                 # Sprocket hole-to-hole Y-offset (extent of row holes)
#         ...
#       }
#
# Representation of a row of tape data:
#       {
#         params:   row_params              # (See above)
#         traces:  [(trace)]                # List of (closed) traces corresponding to row data (see above)
#       }
#
#
# frnum                 frame number currently processed.  No new traces may end before this frame number.
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
FLA = 4         # Frame lookahead when determining end of row
def trace_row_detect(frnum, row_params, trace_data):
    # Find all maximal subsets of overlapping traces (sharing at least one frame)
    # (maximal => no other item in trace_data overlaps with any member of set)
    trace_subsets = []
    for t in trace_data:
        overlap_seen = False
        print("@@@ tsoverlaps for ", str(t))
        tsoverlaps = ( find_trace_overlaps(t, ts) for ts in trace_subsets )
        for oi, otss in enumerate(tsoverlaps):
            print("@@@ Trace overlap ", oi, ":")
            for ots in otss:
                print("@@@ next ots")
                for ot in ots:
                    print(str(ot))
        for i, (tsinc, tsexc) in enumerate(tsoverlaps):
            if tsinc: 
                overlap_seen = True
                if tsexc:
                    # Partial overlap: new subset with new trace
                    trace_subsets.append(tsinc.append(t))
                else:
                    # Full overlap: add to existing subset
                    trace_subsets[i] = trace_subsets[i].append(t)
        if not overlap_seen:
                # No overlap - new singleton sub set
                trace_subsets.append([t])

    for i, ts in enumerate(trace_subsets):
        print("@@@ Trace subset ", i, ":")
        for t in ts:
            print(str(t))

    # Find sets of traces that cannot be further extended
    updated_trace_data = trace_data.copy()
    updated_row_params = row_params.copy()
    new_rows = []
    while True:
        tsi = None
        for i, ts in enumerate(trace_subsets):
            maxlen = 2*min( (t.frlen) for t in ts ) + FLA
            maxbeg = min( (t.frend) for t in ts )
            if frnum >= maxbeg+maxlen:
                tsi = i
                break
        # Test possible row candidate
        if tsi != None:
            tsrow = trace_subsets.pop(tsi)
            # Have row candidate: eliminate non-eligible traces per [B1]
            medlen = statistics.median( (t.frlen for t in tsrow) )
            for t in tsrow:
                if t.frlen > 2*medlen:
                    tsrow.remove(t)
                if t in updated_trace_data:
                    updated_trace_data.remove(t)
            if len(tsrow) < 3:
                log_warning("About frame ", frnum, ": insufficient traces for row")
                log_region_traces(frnum, tsrow)
            else:
                new_rows.append( {'params': row_params, 'traces': tsrow} )
        else:
            # No more candidates
            break

    return (new_rows, updated_row_params, updated_trace_data)

def draw_rows(frame_show, frnum, rows, colour_border, colour_fill):
    for row in rows:
        for rt in row['traces']:
            w    = (rt.frend - rt.frnum)*5
            h    = rt.area / w
            xoff = (rt.frend - frnum)*5
            if (w < 1000) and (rt.xcen + xoff > 0):
                x1   = round(rt.xcen - w + xoff)
                y1   = round(rt.ycen - h/2)
                x2   = round(rt.xcen + xoff)
                y2   = round(rt.ycen + h/2)
                cv.rectangle(frame_show, (x1,y1), (x2, y2), colour_fill,   thickness=cv.FILLED)
                cv.rectangle(frame_show, (x1,y1), (x2, y2), colour_border, thickness=2)
    return

def draw_new_rows(frame_show, frnum, rows):
    # NOTE colour channels are B,G,R:
    # see https://docs.opencv.org/4.5.3/d8/d01/group__imgproc__color__conversions.html#ga397ae87e1288a81d2363b61574eb8cab
    colour_trace = (  0, 255,   0)  # Green
    colour_new   = (255, 255,   0)  # Cyan
    colour_old   = (128, 128,   0)  # Teal
    draw_rows(frame_show, frnum, rows, colour_trace, colour_new)
    return

def draw_old_rows(frame_show, frnum, rows):
    # NOTE colour channels are B,G,R:
    # see https://docs.opencv.org/4.5.3/d8/d01/group__imgproc__color__conversions.html#ga397ae87e1288a81d2363b61574eb8cab
    colour_trace = (  0, 255,   0)  # Green
    colour_new   = (255, 255,   0)  # Cyan
    colour_old   = (128, 128,   0)  # Teal
    draw_rows(frame_show, frnum, rows, colour_trace, colour_old)
    return

def log_rows(frnum, rows):
    for i, row in enumerate(rows):
        log_info("Frame {frnum:d}, row {i:d}")
        for rt in row['traces']:
            log_info(f"frnum {rt.frnum:4d}, frend {rt.frend:4d}, cent ({rt.xcen},{rt.ycen}, area {rt.area}")

# show_region_traces_rows(frame_label, frame_number, frame_data, rcoords, rtraces)
#
# frame_label   is a display label for the presented video frame
# frame_number  is a frame number to be displayed
# frame_data    is raw video data of the mask frame to be displayed
# rcoords       is the region coordinates to be displayed for the current frame
# rtraces       is a list of closed region traces to be displayed
# new_rows      is a list of traces to be highlighted as newly detected rows
# old_rows      is a list of traces to be highlighted as previously detected rows
#
# Returns displayed video frame
#
def show_video_frame_mask_region_traces_rows(frame_label, frame_number, frame_data, rcoords, rtraces, new_rows, old_rows):
    # Display mask frame with centroid data, and return displayed frame value
    frame_show = cv.cvtColor(frame_data, cv.COLOR_GRAY2RGB)
    draw_region_centroids(frame_show, rcoords)
    draw_region_traces(frame_show, frame_number, rtraces)
    draw_new_rows(frame_show, frame_number, new_rows)
    draw_old_rows(frame_show, frame_number, old_rows)
    return show_video_frame(frame_label, frame_number, frame_show)

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
        log_error('Unable to open: ' + args.input)
        exit(0)
    log_info("frame_height", frame_height, "frame_width", frame_width)

    video_writer = open_video_output(
            'output.avi',                   # where
            frame_width*2, frame_height     # size
        )

    #  Main frame-processing loop

    paused = False
    step   = False

    open_traces   = []
    closed_traces = []
    row_traces    = []
    new_rows      = []
    old_rows      = []
    row_params    = {}

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

            # Extract coordinates of highlight regions
            region_pixels_list = list(find_frame_regions(frame_number, frame_highlights))
            region_coords_list = list(find_frame_region_coords(region_pixels_list))

            # Filter out small and large highlight regions.  
            # Hopefully, just sprocket and data holes remain...
            filtered_region_coords_list = filter_region_coordinates(region_coords_list)

            # Display coords
            # frame_show_coords = show_video_frame_mask_centroids(
            #     "Centroids", frame_number, frame_highlights, 
            #     filtered_region_coords_list
            #     )

            log_debug("Frame (filtered)", frame_number)
            for c in filtered_region_coords_list:
                log_debug("  ", str(c))
            # paused = True

            # Coalesce regions from successive frames into region traces
            new_traces, open_traces = region_trace_detect(frame_number, filtered_region_coords_list, open_traces)
            closed_traces = region_trace_add(frame_number, new_traces, closed_traces)

            # Show closed region traces
            log_region_traces(frame_number, new_traces)
            frame_show_traces = show_video_frame_mask_region_traces(
                "Traces", frame_number, frame_highlights, 
                filtered_region_coords_list, closed_traces
                )

            # Group traces into rows
            row_traces = region_trace_add(frame_number, new_traces, row_traces)
            new_rows, row_params, row_traces = trace_row_detect(frame_number, row_params, row_traces)

            # log_rows(frame_number, new_rows)

            # Show row traces
            # frame_show_traces = show_video_frame_mask_region_traces_rows(
            #     "Rows", frame_number, frame_highlights, 
            #     filtered_region_coords_list, closed_traces,
            #     new_rows, old_rows
            #     )
            old_rows.extend(new_rows)
            new_rows = []

            # Write the frame into the file 'output.avi'
            # write_video_frame_pair(video_writer, frame_number, frame_show_orig, frame_show_traces)

        # Check for keyboard interrupt, or move to next frame after 20ms
        paused   = paused or step
        keyboard = cv.waitKey(20)
        if keyboard != -1: 
        	log_debug(keyboard)
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

    log_info("closing down")
    close_video_input(video_capture)
    close_video_output(video_writer)
    log_info("video streams closed")
    close_video_windows()
    log_info("end of main")

# Top level script

paused = False
step   = False
main()
# test_row_detect()

# End.
