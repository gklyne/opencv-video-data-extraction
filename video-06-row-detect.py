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


# ===================================================== #
# == Constants ======================================== #
# ===================================================== #

ROW_FRAME_LOOKAHEAD = 4             # Frame lookahead when determining end of row

FRAME_WIDTH_X       = 5             # X pixel length for displaying historical frames

# ===================================================== #
# == Log/debug functions ============================== #
# ===================================================== #

def log_error(*args):
    print("ERROR", *args)

def log_warning(*args):
    print("WARNING", *args)

def log_info(*args):
    print(*args)

def log_debug(*args):
    # print(*args)
    pass

# ===================================================== #
# == Coordinate mapping functions ===================== #
# ===================================================== #

# These functions are used to map frame-derived coordinates into pixel
# coordinates used for the video outputs.

# h = self.area / map_frame_len(self.frend-self.frnum) # Height of displayed trace
# x1, y1 = map_frame_coords(frnow, self.frnum, self.xcen, self.ycen-h/2)

def map_frame_len(frlen):
    """
    Convert display frame range to pixel range

    frlen           (integer) is an frame length (duration) value

    Returns:        (integer) the display width in pixels corresponding to `frlen`
    """
    return frlen * FRAME_WIDTH_X

def map_frame_pos(frnow, frnum, x, y):
    """
    Map x,y position at some historical frame number to x,y pixel coordinates for
    display.  Older frames are shifted to provide a visualization of advancing time
    that very loosely relates to the motion of the tape being scanned.

    frnow           (integer) current frame number.
    frnum           (integer) frame number relating to the coordinate value to be mapped.
    x               (float) X coordinate to be mapped
    y               (float) Y coordinate to be mapped

    Returns:        x,y coordinate pair for the position of the supplied coordinate, 
                    adjusted for the passage of time (video frames)
    """
    x1 = round(x + (frnum-frnow)*FRAME_WIDTH_X)
    y1 = round(y)
    return (x1, y1)

# ===================================================== #
# == Video I/O functions ============================== #
# ===================================================== #

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


# ===================================================== #
# == Video processing pipeline classes and functions == #
# ===================================================== #

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

    # Merge non-zero pixels into regions.
    region_pixels_list = []              # [region_frame_pixels]
    for ipy,ipx in pixel_coords:
        region_numbers = []         # List of region numbers adjoining next pixel
        for nr, r in enumerate(region_pixels_list):
            # Find regions that adjoin new pixel
            merge = False
            if ( r and
                 (ipx >= r.xmin-2) and (ipx <= r.xmax+2) and
                 (ipy >= r.ymin-1) and (ipy <= r.ymax+1) ):
                # Look for adjacency to actual pixel:
                for rpx,rpy in r.pixelcoords:
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
            for n in range(1, len(region_numbers)):
                r1 = region_pixels_list[region_numbers[n]]
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
        self.calc_centroid_area(region_pixels)
        return

    def draw(self, frame):
        """
        Draw centroid in supplied video frame buffer
        """
        # @@@@ accept colour as parameter
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

    def overlaps(self, r2):
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

    def overlaps_region(self, rtry):
        """
        Determine if supplied region overlaps with the current region trace

        rtry              a `region_frame_centroid` value that is to be tested.

        returns:        True if the supplied region coordinates overlap the
                        final frame of the current region trace.
        """
        # log_debug("@@@ overlaps_region ", self.frnum, len(self.rcoords))
        if len(self.rcoords) == 0:
            return False
        rend = self.rcoords[-1]         # Last position in trace
        return rend.overlaps(rtry)

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

    def overlaps_trace_frames(self, rtry):
        """
        Determine if supplied region trace overlaps temporally (shares any frames) 
        with the current region trace

        rtry            a `region_trace_closed` value that is to be tested.

        returns:        True if the supplied region trace coordinates overlap the
                        current region trace.
        """
        # NOTE: end frame is *after* last frame of trace, so using '<' not '<='
        return (self.frnum < rtry.frend) and (rtry.frnum < self.frend)

    def __str__(self):
        return (
            f"region_trace: frnum {self.frnum:4d}, frend {self.frend:4d}"
            f", area {self.area}"
            f", cent ({self.xcen:6.1f},{self.ycen:6.1f})"
            )

    def short_str(self):
        return (
            f"trace: frames {self.frnum:d}::{self.frend:d}, area {self.area}"
            )

    def format_frame_coords(self, prefix):
        """
        Generator returns formatted region coordinates for frames in trace
        """
        for rc in self.rcoords:
            yield (prefix+str(rc))
        return

    def draw(self, frame_show, frnow, colour_border):
        """
        Draw trace in supplied video frame buffer

        The trace is offset by the current frame number so that it is moved to
        the left when drawn in later frames.

        frame_show      OpenCV video output buffer for displayed value
        frnow           (integer) current frame number
        colour_border   Colour value for displaying border of area covered trace 
        """
        h = self.area / map_frame_len(self.frend-self.frnum)    # Height for displayed trace
        x1, y1 = map_frame_pos(frnow, self.frnum, self.xcen, self.ycen-h/2)
        x2, y2 = map_frame_pos(frnow, self.frend, self.xcen, self.ycen+h/2)
        cv.rectangle(frame_show, (x1,y1), (x2, y2), colour_border, thickness=2)
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
            if rt.overlaps_region(rc):
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
    colour_green = (0, 255, 0)
    for rt in traces:
        rt.draw(frame_show, frnum, colour_green)
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

# ===================================================== #
# == Row detection classes and methods ================ #
# ===================================================== #

# A row is a set of closed traces that are assessed as representing a 
# row of holes in the Monotype tape.

# ----------------------------------------- #
# -- region_trace_set --------------------- #
# ----------------------------------------- #

FLA = ROW_FRAME_LOOKAHEAD       # Frame lookahead when determining end of row

class region_trace_set(object):
    """
    Represents a set of region traces that overlap in time (i.e. frame number).   
    This is used for the detection of rows of region traces.

    For region traces P and Q, we have:

    P.frbeg, Q.frbeg        are the first frame number in which the traces appear
    P.frend, Q.frend        are the first frame after the final frame in which the traces appear
    P.frlen, Q.frlen        are the number of consecutive frames in the traces appear

    E.g.    P......>
            <======> P.frlen
            ^       ^ 
            |       | 
            |       + P.frend 
            |
            + P.frbeg

    A row is a (maximal) set of traces (approximately) satisfying the following constraints:


    [A]: P.frbeg < Q.frend 
      - for all traces P and Q in a row.  
        No trace in a row can start after another has ended; all traces in a row overlap 
        in at least one frame.  This is used as a basis for determining that no further 
        traces can be added to a row, assuming that traces are presented in order.

    [B1]: P.frlen < 2*Median(Q.frlen)
      - for all traces P and Q in a row.  
        No trace may be longer than twice the median length of other traces in a row.
        The intent is to eliminate extended traces that don't correspond to tape holes, 
        based on the assumption that all traces in a row will be of similar duration.

    [B2]: P.frlen < 2*Q.frlen+FLA
      - for all traces P and Q in a row.  
        While row data is incomplete, the median length of all traces cannot be known.
        This constraint provides a very rough estimator for assessing whether there might be
        more traces to add, using the minimum trace length seen as a lower bound for the median.
      
        The "+FLA" is to avoid false row endings when a very short trace is present: the assumption here
        is that if an exceptional short trace is present, more representative traces will be encountered 
        within the FL additional frames.  This effectively forces a degree of lookahead in the incoming
        traces before declaring that a row is complete.

    [C]: each row contains least 3 traces.
        A valid Monotype tape row has at least 4 holes: 2 sprocket holes and at least 2 data holes.  
        The minimum of 3 allows for one of the sprocket holes to be missed.  (Any putative row with 
        fewer than 3 traces is logged and disregarded.)

    [D]: (P.xcen, P.ycen) = (xbase + k*xspan, ybase + k*yspan); 0 <= k <= 1
      - for all traces P in a row, where xbase, ybase, xspan, yspan are the same for each such P.

        This effectively requires that the traces in a row sit spatially on a line, at multiples 
        of some unit separation distance.  

        If (xbase,ybase) represents one of the sprocket holes, then (xspan,yspan) represents 
        the offset to the sprocket hole on the other side of the tape.  

        NOTE: sprocket holes are approx 5mm from data holes; data hole spacing is approx 3mm.

    These constraints are designed to allow a RANSAC-like algorithm [1] to be used to separate real
    data from noise (traces not corresponding to tape holes), without making assumptions about the
    camera orientation (may be skewed) or speed at which the tape is being drawn over the reader bar.
    In this application it is possible that the total number of traces under consideration at any time 
    is small enough to allow an exhaustive search rather than random sampling.

    [1] https://en.wikipedia.org/wiki/Random_sample_consensus
    """

    def __init__(self, trace_set=None, trace=None):
        """
        Initialize new region_trace_set.

        trace_set       if present, is a list of `region_trace_closed` values that are
                        added to the newly created `region_trace_set`.  
                        (cf. result from `find_trace_overlaps`.)
        trace           if present, is a `region_trace_closed` valuer that is added to the
                        newly created `region_trace_set`.
        """
        self.rtraces  = []                      # List of region traces in set
        self.maxfrbeg = 0                       # Maximum trace start frame
        self.minfrend = sys.maxsize             # Minimum trace end frame
        self.minfrlen = sys.maxsize             # Minimum frame length of any trace
        self.xmin     = sys.maxsize             # Region covering all traces ..
        self.ymin     = sys.maxsize
        self.xmax     = 0
        self.ymax     = 0
        if trace_set != None:
            for t in trace_set:
                self.add_trace(t)
        if trace:
            self.add_trace(trace)
        return

    def __iter__(self):
        """
        Return iterator over members of the trace set
        """
        return iter(self.rtraces)

    def __len__(self):
        # See https://stackoverflow.com/a/15114062/324122
        return len(self.rtraces)

    def __str__(self):
        return ("trace_subset: " + self.short_str())

    def short_str(self):
        return (
            f"maxfrbeg {self.maxfrbeg:d}"
            f", minfrend {self.minfrend:d}, minfrlen {self.minfrlen:d}"
            f", num traces  {len(self.rtraces):d}"
            )

    def draw(self, frame_show, frnow, colour_border, colour_fill):
        """
        Draw background shading connecting traces in set (row)
        """
        xcen = (self.xmin + self.xmax) / 2
        x1, y1 = map_frame_pos(frnow, self.maxfrbeg, xcen, self.ymin )
        x2, y2 = map_frame_pos(frnow, self.minfrend, xcen, self.ymax)
        cv.rectangle(frame_show, (x1,y1), (x2, y2), colour_fill, thickness=cv.FILLED)
        # cv.rectangle(frame_show, (x1,y1), (x2, y2), colour_border, thickness=2)
        return

    def log_trace_set(self, logmethod=print, prefix="", t_prefix="    "):
        logmethod(prefix, self.short_str())
        for t in self.rtraces:
            logmethod(t_prefix+t.short_str())
        return

    def add_trace(self, trace):
        """
        Adds a new region trace to the current trace set
        """
        self.maxfrbeg = max(self.maxfrbeg, trace.frnum)
        self.minfrend = min(self.minfrend, trace.frend)
        self.minfrlen = min(self.minfrlen, trace.frlen)
        self.xmin     = min(self.xmin, trace.xmin)
        self.ymin     = min(self.ymin, trace.ymin)
        self.xmax     = max(self.xmax, trace.xmax)
        self.ymax     = max(self.ymax, trace.ymax)
        self.rtraces.append(trace)
        return self

    def remove_trace(self, trace):
        """
        Remove a region trace from the current trace set
        """
        if trace in self.rtraces:
            self.rtraces.remove(trace)
        return self

    def find_trace_overlaps(self, trace):
        """
        For a supplied trace, find overlapping and non-overlapping traces in the
        current trace set.
        
        trace                 a trace to be tested
        
        Returns:  (trace_inc, trace_exc)
        
        trace_inc             list of elements in the current set that overlap with `trace`
        trace_exc             list of elements in the current set that do not overlap with `trace`
        """

        # def trace_overlaps(t1, t2):
        #     # NOTE: end frame is *after* last frame of trace
        #     return (t1.frnum < t2.frend) or (t2.frnum < t1.frend)
        ts_inc = []
        ts_exc = []
        if len(self.rtraces) > 20:
            raise ValueError("Too many (>20) unallocated traces")
        for t in self.rtraces:
            if t.overlaps_trace_frames(trace):
                ts_inc.append(t)
            else:
                ts_exc.append(t)
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
def trace_row_detect(frnum, row_params, trace_data):
    # Find all maximal subsets of overlapping traces (sharing at least one frame)
    # (maximal => no other item in trace_data overlaps with any member of set)

    global paused, step

    trace_subsets = []
    for t in trace_data:
        overlap_seen = False
        tsoverlaps = [ ts.find_trace_overlaps(t) for ts in trace_subsets ]
        for i, (tsinc, tsexc) in enumerate(tsoverlaps):
            if tsinc: 
                overlap_seen = True
                if tsexc:
                    # Partial overlap: new subset with new trace
                    trace_subsets.append(region_trace_set(trace_set=tsinc, trace=t))
                else:
                    # Full overlap: add to existing subset
                    trace_subsets[i] = trace_subsets[i].add_trace(t)
        if not overlap_seen:
                # No overlap - new singleton sub set
                trace_subsets.append(region_trace_set(trace=t))

    # Find sets of traces that cannot be further extended
    updated_trace_data = trace_data.copy()
    updated_row_params = row_params.copy()
    new_rows = []
    for limit_counter in range(len(trace_subsets)):     # Each pass should remove one subset, or break out
        tsi = None
        for i, ts in enumerate(trace_subsets):
            maxlen = 2*ts.minfrlen + FLA
            maxbeg = ts.minfrend
            if frnum >= maxbeg+maxlen:
                tsi = i
                break
        # Test possible row candidate
        if tsi != None:
            row_trace_set = trace_subsets.pop(tsi)
            # Have row candidate: eliminate non-eligible traces per [B1]
            # Also remove trace from traces to be carried forward.
            medlen = statistics.median( (t.frlen for t in row_trace_set) )
            for t in row_trace_set:
                if t.frlen > 2*medlen:
                    row_trace_set.remove_trace(t)
                if t in updated_trace_data:
                    updated_trace_data.remove(t)
            if len(row_trace_set) < 3:
                log_warning("About frame ", frnum, ": insufficient traces for row")
                log_region_traces(frnum, row_trace_set)
            else:
                new_rows.append(row_trace_set)
        else:
            # No more candidates
            break

    return (new_rows, updated_row_params, updated_trace_data)

def draw_rows(frame_show, frnum, rows, colour_border, colour_fill1, colour_fill2=None):
    colour_fill, colour_fill_next = (colour_fill1, colour_fill2 or colour_fill1)
    for row in rows:
        row.draw(frame_show, frnum, colour_border, colour_fill)
        colour_fill, colour_fill_next = (colour_fill_next, colour_fill)
    return

def draw_new_rows(frame_show, frnum, rows):
    # NOTE colour channels are B,G,R:
    # see https://docs.opencv.org/4.5.3/d8/d01/group__imgproc__color__conversions.html#ga397ae87e1288a81d2363b61574eb8cab
    colour_trace = (204, 255, 204)  # Light green
    colour_new   = (102, 255, 255)  # Light yellow
    draw_rows(frame_show, frnum, rows, colour_trace, colour_new)
    return

def draw_old_rows(frame_show, frnum, rows):
    # NOTE colour channels are B,G,R:
    # see https://docs.opencv.org/4.5.3/d8/d01/group__imgproc__color__conversions.html#ga397ae87e1288a81d2363b61574eb8cab
    colour_trace = (  0, 102,   0)  # Green
    colour_old1  = (102, 102,   0)  # Yellow/green colour old rows (odd)
    colour_old2  = (  0, 102, 102)  # Blue/green   colour old rows (even)
    draw_rows(frame_show, frnum, rows, colour_trace, colour_old1, colour_old2)
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
    draw_old_rows(frame_show, frame_number, old_rows)
    draw_new_rows(frame_show, frame_number, new_rows)
    draw_region_traces(frame_show, frame_number, rtraces)
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
    draw_new_rows = []
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

            # Convert to greyscale, detect highlights
            frame_grey       = convert_frame_to_greyscale(frame)
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
            # frame_show_traces = show_video_frame_mask_region_traces(
            #     "Traces", frame_number, frame_highlights, 
            #     filtered_region_coords_list, closed_traces
            #     )

            # Group traces into rows
            row_traces = region_trace_add(frame_number, new_traces, row_traces)
            new_rows, row_params, row_traces = trace_row_detect(frame_number, row_params, row_traces)

            # log_rows(frame_number, new_rows)

            # Show row traces
            if len(new_rows) > 0:
                draw_new_rows = new_rows
            frame_show_rows = show_video_frame_mask_region_traces_rows(
                "Rows", frame_number, frame_highlights, 
                filtered_region_coords_list, closed_traces,
                draw_new_rows, old_rows
                )

            # Move new rows to old
            old_rows.extend(new_rows)
            new_rows = []

            # Write the frame into the file 'output.avi'
            write_video_frame_pair(video_writer, frame_number, frame_show_orig, frame_show_rows)

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
