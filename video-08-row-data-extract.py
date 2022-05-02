# video-08-row-data-extract.py
#
# This version takes sorted row data and extracts hole data from each row.
# This is expected to be the final stage of decoding data from a Monotype
# system tape video.
#

import cv2 as cv
import numpy as np
import argparse
import math
import statistics
import functools
import itertools
import sys

# ===================================================== #
# == Constants ======================================== #
# ===================================================== #

FRAME_WIDTH_X       = 5             # X pixel length for displaying historical frames

TRACE_MAXFRAMES     = 50            # Maximum frame length of a trace

TRACE_MAXAGE        = 60            # Maximum frame age of the end of a trace

ROW_FRAME_LOOKAHEAD = 4             # Frame lookahead when determining end of row

ROW_FRAME_MAXGAP    = 15            # Maximum gap between frames comprising a row

ROW_FRAME_ORPHAN    = 70            # Gap after frame to be considered orphaned

# NOTE: MAX_ADD_RESIDUAL is used when adding a non-overlapping trace to a row candidate.
# Overlapping traces are added unconditionally, and MAX_ROW_RESIDUAL used to decide
# if the resulting row candidate is acceptable.  
#
# When the tape moves slowly, residuals tend to be larger, but holes in a row usually 
# all appear in at least one frame.

MAX_ADD_RESIDUAL    = 0.6           # Max squared residual for adding trace to row
                                    # (0.5 too small, 2.0 too large)

MAX_ROW_RESIDUAL    = 2.0           # Max residual for accepting trace residual as row

NO_FIT_RESIDUAL     = 10            # Residual when no degrees of freedom for model fitting

# To assist with debugging...

START_FRAME = 0

PAUSE_FRAMES = (
    { 3216, 3340        # 3217, 3351 adjacent holes merged in traces; crossover
    , 3350              # 3367: hole out of range (y1=0, d1=-0.05, n60=-3)
    # , 3570              # Double trace for single hole?
    , 5350              # (5363) trace outside tape, added to row; correct sprocket missed
    , 5650              # Sprocket holes merged
    , 5950, 5954        # 5954(?) very long trace messes with detection
    , 5980, 5990        # Trace outside tape width added to row
    , 6070, 6080        # Trace outside tape width added to row
    , 6090
    , 6350              # Multiple spurious traces outside width of tape
    , 6460              # Trace outside tape width added to row
    , 6640              # Multiple errors?
    , 6790, 6800, 6808  # Multiple merged holes?
    , 6900              # Merged sprocket holes (multiple?)
    , 6940              # (6948) Trace outside tape width added to row
    , 7080              # (7082) Trace outside tape width added to row
    , 7170              # Many spurious traces outside tape width; big width mismatch
    , 7230, 7236        # Multiple spurious traces outside width of tape
    , 7280              # (7288) Trace outside tape width added to row
    , 7290              # (7298) Merged holes give broken row(s)
    , 7350              # (7362) Trace outside tape width added to row
    , 7410, 7420        # Crossed rows, cause not obvious
    , 7440, 7452        # Crossed rows, merged sprocket holes, delayed detection, other poor data?
    , 7461              # Bad row (merged holes?)
    , 7535, 7540        # Trace outside tape width added to row
    , 7620              # Multiple spurious traces outside width of tape; big error
    , 7760              # Missing sprocket hole?  Truncated row.  7765 missed sprocket
    , 7825              # Merged sprocket holes, big error.
    , 7840              # Trace outside tape width added to row
    , 7845, 7850        # Merged sprocket holes(?) (7847, 7850) Inaccurate and truncated rows
    , 7870, 7880        # (7882) Large trace outside tape width added to row, distorted row(s)?
    , 7890              # Merged sprocket holes
    , 7898              # Crossed rows (trace outside tape)
    , 7930, 7933        # Merged sprocket holes
    , 8030, 8040        # Multiple merged holes? 
    , 12000
    , 14200
    })


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
    frame_height = frame_show.shape[0];
    # Text in rectangle 2 px from top/bottom, with 5px margin below
    #
    #   ----          \
    #      2           |
    #   +-----------   |
    #   | :   text     |
    #   | 18     5     |
    #   +-----------   |
    #     :             > frame_height
    #   +-----------   |
    #   | :   text     |
    #   | 18     5     |
    #   +-----------   |
    #      2           |
    #   ----          /
    cv.rectangle(frame_show, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(frame_show, str(frame_number), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    cv.rectangle(frame_show, (10, frame_height-20), (100,frame_height-2), (255,255,255), -1)
    cv.putText(frame_show, str(frame_number), (15, frame_height-7),
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

    def __str__(self):
        return self.long_str()

    def long_str(self, prefix=""):
        return (prefix+
            f"region_trace: frnum {self.frnum:4d}, frend {self.frend:4d}"
            f", area {self.area}"
            f", cent ({self.xcen:6.1f},{self.ycen:6.1f})"
            )

    def short_str(self, prefix=""):
        return (prefix+
            f"trace: frames {self.frnum:d}::{self.frend:d}, area {self.area}"
            )

    def active(self, frnum):
        """
        Returns True if the current trace is considered active for inclusion in any 
        future rows when processing the indicated video frame number.
        """
        return ( (frnum - self.frend) <= TRACE_MAXAGE )

    def overlaps_trace(self, rtry):
        """
        Determine if supplied region trace overlaps temporally (shares any frames) 
        with the current region trace

        rtry            a `region_trace_closed` value that is to be tested.

        returns:        True if the supplied region trace coordinates overlap the
                        current region trace.
        """
        # NOTE: end frame is *after* last frame of trace, so using '<' not '<='
        return (self.frnum < rtry.frend) and (rtry.frnum < self.frend)

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
        cv.rectangle(frame_show, (x1,y1), (x2,y2), colour_border, thickness=2)
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
    # log_info("region_trace_add: ")
    # log_region_traces(frnum, ending_traces)
    closed_traces.extend(ending_traces)
    return closed_traces

def draw_region_traces(frame_show, frnum, traces, colour=None):
    # NOTE colour channels are B,G,R:
    # see https://docs.opencv.org/4.5.3/d8/d01/group__imgproc__color__conversions.html#ga397ae87e1288a81d2363b61574eb8cab
    colour_green = (0, 255, 0)
    colour       = colour or colour_green
    for rt in traces:
        rt.draw(frame_show, frnum, colour)
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
        trace           if present, is a `region_trace_closed` value that is added to the
                        newly created `region_trace_set`.
        """
        self.rtraces  = []                      # List of region traces in set
        self.minfrbeg = sys.maxsize             # Minimum trace start frame
        self.maxfrend = 0                       # Maximum trace end frame
        self.maxfrlen = 0                       # Maximum frame length of any trace
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

    def __contains__(self, trace):
        """
        Test for trace contained in trace set.
        """
        return trace in self.rtraces

    def __len__(self):
        # See https://stackoverflow.com/a/15114062/324122
        return len(self.rtraces)

    def index_trace(self, i):
        return self.rtraces[i]

    def __str__(self):
        return (self.long_str(prefix="trace_subset: ", t_prefix="  "))

    def short_str(self):
        return (
            f"minfrbeg {self.minfrbeg:d}, maxfrend {self.maxfrend:d}, "
            f"maxfrlen {self.maxfrlen:d}, "
            f"num traces  {len(self.rtraces):d}"
            )

    def long_str(self, prefix="", t_prefix="  "):
        return (
            prefix + self.short_str() + "\n" +
            "\n".join([ t_prefix + t.long_str() for t in self.rtraces ] )
            )

    def log(self, frnum, logmethod=print, prefix="", t_prefix="    "):
        logmethod("log (region_trace_set) " + str(frnum))
        logmethod(self.long_str(prefix=prefix, t_prefix=t_prefix))
        # logmethod(prefix, self.short_str())
        # for t in self.rtraces:
        #     logmethod(t_prefix+t.short_str())
        return

    def draw(self, frame_show, frnow, colour_border, colour_fill):
        """
        Draw background shading connecting traces in set (row)
        """
        # Display as rectangle based on top/bottom corners
        topy     = 0
        topxmin  = 0
        topxmax  = 0
        topfrbeg = 0
        topfrend = 0
        lowy     = sys.maxsize
        lowxmin  = 0
        lowxmax  = 0
        lowfrbeg = 0
        lowfrend = 0
        for t in self.rtraces:
            if (t.ymax > topy):
                topy     = t.ymax
                topx     = (t.xmin+t.xmax) // 2
                topxmin  = t.xmin
                topxmax  = t.xmax
                topfrbeg = t.frnum
                topfrend = t.frend
            if (t.ymin < lowy):
                lowy     = t.ymin
                lowx     = (t.xmin+t.xmax) // 2
                lowxmin  = t.xmin
                lowxmax  = t.xmax
                lowfrbeg = t.frnum
                lowfrend = t.frend
        row_points = np.array(
            [ map_frame_pos(frnow, lowfrbeg, lowx+1, lowy)
            , map_frame_pos(frnow, lowfrend, lowx-1, lowy)
            , map_frame_pos(frnow, topfrend, topx-1, topy)
            , map_frame_pos(frnow, topfrbeg, topx+1, topy)
            , map_frame_pos(frnow, lowfrbeg, lowx+1, lowy)
            ])
        cv.fillConvexPoly(frame_show, row_points, colour_fill) # , cv.FILLED)
        return

    def subsumes_set(self, ts):
        """
        returns True if the current region_trace_set subsumes the supplied `ts`
        (i.e. contains all of the traces that are also in `ts`).
        """
        if ts == None:
            raise ValueError("Trace set is None")
        return all( ((t in self.rtraces) for t in ts) )

    def overlaps_traces(self, rtry):
        """
        Determine if supplied region trace overlaps temporally (shares any frames) 
        with all traces in the current trace set.

            overlap_set, overlap_all = region_trace_set.overlaps.traces(rtry)

        Where:

        rtry            a `region_trace_closed` value that is to be tested.

        overlap_set     returns a list of `region_trace_closed` values that overlap
                        the supplied trace `rtry`.

        overlap_all     returns True if the supplied trace overlaps all traces in
                        the current region_trace_set. 
        """
        overlaps_rtry = [ (t, t.overlaps_trace(rtry)) for t in self.rtraces ]
        overlap_set   = [ ts for ts, overlap in overlaps_rtry if overlap ]
        overlap_all   = all( ( overlap for _, overlap in overlaps_rtry ) )
        return (overlap_set, overlap_all)

    def add_trace(self, trace):
        """
        Adds a new region trace to the current trace set

        trace       is a `region_trace_closed` object that is to be included
                    in the current `region_trace_set` object.
        """
        self.minfrbeg = min(self.minfrbeg, trace.frnum)
        self.maxfrend = max(self.maxfrend, trace.frend)
        self.maxfrlen = max(self.maxfrlen, trace.frlen)
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
            self.minfrbeg = min( (t.frnum for t in self.rtraces) )
            self.maxfrend = max( (t.frend for t in self.rtraces) )
            self.maxfrlen = max( (t.frlen for t in self.rtraces) )
            self.xmin     = min( (t.xmin  for t in self.rtraces) )
            self.ymin     = min( (t.ymin  for t in self.rtraces) )
            self.xmax     = max( (t.xmax  for t in self.rtraces) )
            self.ymax     = max( (t.ymax  for t in self.rtraces) )
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
        ts_inc = []
        ts_exc = []
        # @@@@TODO: this check belongs elsewhere
        # if len(self.rtraces) > 20:
        #     raise ValueError("Too many (>20) unallocated traces")
        for t in self.rtraces:
            if self._traces_overlap(t, trace):
                ts_inc.append(t)
            else:
                ts_exc.append(t)
        return (ts_inc, ts_exc)

    def _traces_overlap(self, t1, t2):
        """
        Returns True if two supplied traces are considered to appear in a common
        tape row, allowing for tape skew calculated on the current trace set.
        """
        # NOTE: end frame is *after* last frame of trace, so using '<' not '<='
        return (t1.frnum < t2.frend) and (t2.frnum < t1.frend)

def trace_sets_subsume(trace_sets, ts):
    """
    Returns true is the supplied region_trace_set `ts` is subsumed by any of the 
    sets in a list of trace sets.
    """
    return any( (ts1.subsumes_set(ts) for ts1 in trace_sets) )


# ----------------------------------------- #
# -- row_candidate ------------------------ #
# ----------------------------------------- #

class row_candidate(object):
    """
    This class is used for assembling candidate row data, where a row is a set
    of traces that are suitably aligned with respect to the frames in which 
    they appear.  What constitutes "suitably aligned" here is a topic of ongoing 
    experimentation - the intent is that such detail and associated logic will 
    be handled within this class.
    """

    def __init__(self, trace_set=None):
        self.traces   = trace_set or region_trace_set()
        self.residual = self._estimate_linear_fit_residual(self.traces)
        return

    def __iter__(self):
        """
        Return iterator over traces in row candidate
        """
        return iter(self.traces)

    def __contains__(self, trace):
        """
        Test for trace contained in row candidate.
        """
        return trace in self.traces

    def __len__(self):
        """
        Returns number of traces in current candidate row.
        """
        # See https://stackoverflow.com/a/15114062/324122
        return len(self.traces)

    def __str__(self):
        return (self.long_str(prefix="row_candidate: ", t_prefix="  "))

    def long_str(self, prefix="", t_prefix="  "):
        return self.traces.long_str(prefix=prefix, t_prefix=t_prefix)

    def short_str(self):
        return (self.traces.short_str())

    def log(self, frnum, logmethod=print, prefix="row_candidate", t_prefix="    "):
        logmethod(f"{prefix}: frnum: {frnum:d}, complete: {self.row_complete(frnum)}, residual: {self.residual:6.3f} ")
        logmethod(self.long_str(prefix="  ", t_prefix=t_prefix))
        # logmethod(prefix, self.short_str())
        # for t in self.rtraces:
        #     logmethod(t_prefix+t.short_str())
        return

    def draw(self, frame_show, frnow, colour_border, colour_fill):
        """
        Draw background shading connecting traces in row candidate
        """
        self.traces.draw(frame_show, frnow, colour_border, colour_fill)
        return

    def row_initial_trace(self, trace):
        """
        Tests a supplied trace to see if it might be a seed candidate in the current
        candidate row.  This is an initial test used to construct an initial set of 
        candidate rows.  

            match, new_row = row_candidate.row_initial_trace(trace)
    
        Where: 

        trace       is a `region_trace_closed` object that is to be tested against
                    the current row candidate.

        match       is returned True if the supplied trace can be added to the 
                    current row_candidate, otherwise False.  If True, the supplied 
                    trace is added to the current row_candidate.

        new_row     is a new row_candidate formed from a subset of the current 
                    candidate traces and the supplied trace.  
                    The current row_candidate is not affected.

        There are three possible outcomes: 

        (1) the supplied trace can be added to the current row candidate, in which case
            match is returned as True and new_row as None.

        (2) the supplied trace along with a subset of traces from the current candidate 
            forms a new candidate, in which case match is returned False and new_row as
            a new row_cabndidate object.

        (3) the supplied trace does not form a row candidate with any traces in the 
            current row_candidate, in which case match returns False and new_row None.

        The test should favour false negatives over false positives;
        i.e. accept any trace that has a high probability of being part of a row with 
        any other traces already present.  Subsequent analysis may accept further traces 
        that are rejected by this initial test.  Acceptance of a trace is not a final 
        determination - if there is another row for which any trace in this row is a 
        better fit, that is selected instead.

        The initial criterion applied is to see if it overlaps all other traces in the 
        current row:  if it does then it is accepted.
        """
        if trace.frlen > TRACE_MAXFRAMES:   
            # Don't consider overlong trace
            # This test might get a bit cleverer, but so far that's not needed.
            return (False, None)
        overlaps_set, overlaps_all = self.traces.overlaps_traces(trace)
        if overlaps_all:
            self.traces.add_trace(trace)
            self.residual = self._estimate_linear_fit_residual(self.traces)
            return (True, None)
        if len(overlaps_set) > 0:
            new_row_candidate = row_candidate(region_trace_set(trace_set=overlaps_set, trace=trace))
            return (False, new_row_candidate)
        return (False, None)

    def row_additional_trace(self, trace):
        """
        Tests a supplied trace to see if it could be included in the current row candidate.

            match = row_candidate.row_additional_trace(trace)

        Where:

        trace       is a `region_trace_closed` object that is to be tested against
                    the current row candidate.

        match       is returned True if the supplied trace can be added to the 
                    current row_candidate, otherwise False.  If True, the supplied 
                    trace is added to the current row_candidate.

        This is used as a second pass of row detection after an initial row_candidate of
        overlapping traces has been created.  This test picks up any additional traces 
        that may not overlap every other trace, but is closely aligned with them.

        The test used is a low residual for a linear fit through all traces including the 
        supplied additional trace.
        """
        if trace.frlen > TRACE_MAXFRAMES:
            # Don't consider overlong trace
            # This test might get a bit cleverer, but so far that's not needed.
            return False
        if trace in self.traces:
            return False        # Trace already in this row candidate
        traces   = region_trace_set(trace_set=self.traces, trace=trace)
        residual = self._estimate_linear_fit_residual(traces)

        if residual <= MAX_ADD_RESIDUAL:
            # Add trace to row
            # log_info(f"row_additional_trace: residual {residual:6.3f}")
            # log_info(trace.long_str(prefix="    trace: "))
            # log_info(self.long_str(prefix="  adding to ", t_prefix="    "))
            self.traces.add_trace(trace)
            self.residual = residual
            return True
        return False

    def _estimate_linear_fit_residual(self, traces):
        """
        Estimates a (mean squared) residual from a linear fit to the row_candidate
        traces and a supplied additional trace

        The value is based upon a (least square deviation) linear regression fit for 
        (frnum, ycen) for each trace.

        Let:
    
            f  be a frame number
            y  be Y centroid of a trace
            f0 be the frame number corresponding to y = 0
            b  is the frame skew factor

        Then a linear model gives:

            f = f0 + b y

        Given sets of n observed values { F : fi } and { Y : yi }, an ordinary least 
        squares estimator for b is

            b = ( n SUM(fi yi) - SUM(fi) SUM(yi) ) / ( n SUM(yi^2) - SUM(yi)^2 )

            f0 = (SUM(yi) - b SUM(xi)) / n

        (Also called `bhat` and `ahat`.)

        (See https://en.wikipedia.org/wiki/Simple_linear_regression)

        Then the sum of squared residuals is:

            ssr = SUM( (fi - (f0 + b yi))^2 )


        @@NOTE: I've tried deriving a direct calculation for ssr, but thus far
        haven't been able to come up with a formula that I can reconcile with
        published formulae that I'm seeing, and I haven't found a published derivation
        for such a formula.  

        The closest I've found to a published formula is Chatsworth p175, but there's 
        no derivation there.  Using variable names from above, and expressing ssr rather 
        than variance, that would be:
        
            ssr = SUM(fi^2) - f0 SUM(fi) - b SUM(yi fi)

        Also tried derivation using regression formula suggested by Chatfield p169:

            yi - ybar = b (x - xbar)

        noting that the means are easily computed from the summed terms.
        No joy there, either.
        """
        if (len(traces) < 2):
            return NO_FIT_RESIDUAL

        if (len(traces) == 2):
            # 2 traces: assume vertical alignment, calculate residual accordingly with 1DF
            t0      = traces.index_trace(0)
            t1      = traces.index_trace(1)
            t0frmid = (t0.frnum + t0.frend)/2
            t1frmid = (t1.frnum + t1.frend)/2
            frmean  = (t0frmid + t1frmid)/2
            err     = (t0frmid-frmean)  # == (t1frmid-frmean)
            ssr     = 2*err*err
            return ssr # Only 1 DF, so n=1 for variance calc

        # Calculate regression parameters
        n   = 0
        sf  = 0.0           # SUM(fi)
        sf2 = 0.0           # SUM(fi^2)
        sy  = 0.0           # SUM(yi)
        sy2 = 0.0           # SUM(yi^2)
        sfy = 0.0           # SUM(fi yi)
        for t in traces:
            n   += 1
            fmid = (t.frnum + t.frend)/2
            sf  += fmid
            # sf2 += fmid*fmid
            sy  += t.ycen
            sy2 += t.ycen*t.ycen
            sfy += fmid*t.ycen
        bhat = ( n*sfy - sf*sy ) / ( n*sy2 - sy*sy )
        ahat = ( sf - bhat *  sy ) / n

        # Calculate sum of squared residuals from fit
        ssr  = 0.0
        for t in traces:
            fmid = (t.frnum + t.frend)/2
            res  = fmid - (ahat + bhat*t.ycen)
            ssr += res*res
        # Return variance of error, based on mean square residual, and
        # allowing for 2 degrees of freedom used for fitting the model.
        return ssr/(n-2)

    def row_gap(self, frnum):
        """
        Returns size of gap from last frame occupied by the current row candidate
        to the supplied frame number.
        """
        return (frnum - self.traces.maxfrend)

    def row_complete(self, frnum):
        """
        Returns `True` if no further traces are expected to become available for
        the current row candidate, in light of the current video frame being processed.
        """
        # Don't use row data if less than 2 traces present.
        # @@TODO: try just using MAXGAP test.
        mingap   = self.traces.maxfrlen if len(self.traces) >= 2 else ROW_FRAME_MAXGAP
        rowgap   = self.row_gap(frnum)
        complete = ( (rowgap > ROW_FRAME_LOOKAHEAD) and 
                     ( (rowgap > mingap) or (rowgap > ROW_FRAME_MAXGAP) )
                   )
        return complete

    def row_orphaned(self, frnum):
        """
        Returns `True` if the current row candidate is considered "orphaned".

        That is, if the current row candidate cannot be used, and if no further 
        traces can be added to it.
        """
        rowgap   = frnum - self.traces.maxfrend
        return (rowgap > ROW_FRAME_ORPHAN)


# ----------------------------------------- #
# -- Row_Data ----------------------------- #
# ----------------------------------------- #


class Row_Data(object):
    """
    This class represents data from a single row of the Monotype system tape.
    """
    def __init__(self, frnum, xpos, ymin, ymax):
        """
        Construct row data object.

        frnum   observed frame number of row (for display)
        xpos    observed mean X-position of row (for display)
        ymin    observed or estimated minimum-Y sprocket hole position
        ymax    observed or estimated maximum-Y sprocket hole position
        """
        self.frnum = frnum
        self.xpos  = xpos
        self.ymin  = ymin
        self.ymax  = ymax
        self.yrng  = ymax-ymin
        # Data is a Boolean for each hole position, indexed 0..30
        self.data  = [ False for i in range(31) ]
        return

    def set_data(self, index, value=True):
        """
        Set True or False value for a single hole position.
        """
        self.data[index] = value
        return

    def __iter__(self):
        """
        Return iterator over holes in row.  

        Each hole is returned as (index, value)
        """
        return ( (i,self.data[i]) for i in range(31) )

    def __str__(self):
        return (self.short_str())

    def long_str(self, prefix=""):
        return f"{prefix}frnum {self.frnum:4d}, xpos {self.xpos:8.2f}, ymin {self.ymin:8.2f}, ymax {self.ymax:8.2f}, data: {self.short_str()}"

        return self.traces.long_str(prefix=prefix, t_prefix=t_prefix)

    def short_str(self):
        return "|" + "".join([ "O" if self.data[i] else "-" for i in range(31) ]) + "|"

    def log(self, frnow, logmethod=print, prefix="row_data"):
        logmethod(f"{prefix}: frnow: {frnow:d}, {self.long_str(prefix='')}")
        return

    def draw(self, frame_show, frnow, colour_sprocket, colour_holes, colour_fills, row_data_accum):
        """
        Draw symbols for data in row

        frame_show      OpenCV video output buffer for displayed value
        frnow           (integer) current frame number
        colour_sprocket Colour value for displaying sprocket holes
        colour_holes    Colour value for displaying data holes
        """
        def p(x,y): 
            return (x, y)
        # Show sprocket holes
        cr     = 7
        vo     = 10
        cx, cy = map_frame_pos(frnow, self.frnum, self.xpos, self.ymin)
        vs     = np.array([ p(cx,cy+vo), p(cx+vo, cy), p(cx, cy-vo), p(cx-vo, cy) ], np.int32)
        # cv.polylines(frame_show, [vs], True, color=colour_sprocket, thickness=2)
        cv.circle(frame_show, (cx, cy), cr, colour_sprocket, thickness=-2)
        cx, cy = map_frame_pos(frnow, self.frnum, self.xpos, self.ymax)
        vs     = np.array([ p(cx,cy+vo), p(cx+vo, cy), p(cx, cy-vo), p(cx-vo, cy) ], np.int32)
        # cv.polylines(frame_show, [vs], True, color=colour_sprocket, thickness=2)
        cv.circle(frame_show, (cx, cy), cr, colour_sprocket, thickness=-2)
        # Show data holes
        vo     = 12
        # showholes = [ i for (i, v) in self if v ]
        for i, v in self:
            # y1     = (5 + i*95/30)/104     # See `detect_data_holes` comments below
            y1 = row_data_accum.map_n_y1(i)
            cx, cy = map_frame_pos(frnow, self.frnum, self.xpos, self.ymin + y1*self.yrng)
            vs     = np.array([ p(cx,cy+vo), p(cx+vo, cy), p(cx, cy-vo), p(cx-vo, cy) ], np.int32)
            colour = colour_holes if v else colour_fills
            cv.polylines(frame_show, [vs], True, color=colour, thickness=2)
        return


# ----------------------------------------- #
# -- Row_Data_Accumulator ----------------- #
# ----------------------------------------- #

SPROCKET_HOLE_WIDTH  = 104
DATA_HOLE_WIDTH      = 95
SPROCKET_DATA_OFFSET = (SPROCKET_HOLE_WIDTH-DATA_HOLE_WIDTH)/2

class Row_Data_Accumulator(object):
    """
    This class accumulates extracted data from rows, and also maintains a running 
    context that is used to estimate the extent of each row.
    """

    def __init__(self):
        """
        Construct row accumulator data object.
        """
        self.ymin_track = None      # track of row ymin values
        self.ymax_track = None      # track of row ymax values
        self.num_track  = 0         # Number of values contributing to each track value
        self.row_data   = []        # [Row_Data]
        return


    def map_n_y1(self, n):
        """
        For a supplied tape hole number n (0..30), returns a y position normalized to 
        sprocket holes at positions 0 and 1.

        NOTE: 
        - sprocket holes are approx 5mm from data holes; 
        - data hole spacing is approx 3mm.

        Actual measurements:  
        - sprocket holes 104mm between centres
        - outermost data holes 95 between centres

        Thus, for hole number N (zero-based), the position in mm relative to the 
        lowest sprocket hole is given by:  

            ymm = 5 + N*95/30

        With distance between sprocket holes is normalized to 1, i.e.:

            y1 = (y-ymin)/(ymax-ymin)

        where y, ymin and ymax are observed values, then also:

            y1  = (4.5 + N*95/30)/104

        Normalization of Y values is handled by calling code.
        """
        return (SPROCKET_DATA_OFFSET + n*DATA_HOLE_WIDTH/30)/SPROCKET_HOLE_WIDTH

    def map_y1_n(self, y1):
        """
        For supplied y position normalized to spocket holes at 0 and 1, 
        returns the corresponding tape hole number.

        Given from above:

            y1  = (4.5 + N*95/30)/104

        then

            N = ((y1*104 - 5)*30)/95
              = (y1*3120 - 150)/95
              = y1*32.84 - 1.58
        """
        global paused, step
        # d1 is mapping to data region being range 0..1
        #
        # Map to 60-point range to detect "ambiguous" markers
        d1  = (y1*SPROCKET_HOLE_WIDTH - SPROCKET_DATA_OFFSET)/DATA_HOLE_WIDTH
        n60 = round(d1*60)  # map to integers 0..60
        if (n60 < 0) or (n60 > 61):
            # Data out of range
            log_warning(f"Detected hole out of range: y1 {y1:8.2f}, d1 {d1:8.2f}, n60 {n60:d}")
            log_warning(f"ymin_track {ymin_track:8.2f}, ymax_track  {ymax_track:8.2f}")
            paused = True
        elif (n60%1 == 1):
            # Data between columns
            log_warning(f"Detected hole between columns: y1 {y1:8.2f}, d1 {d1:8.2f}, n60 {n60:d}")
            log_warning(f"ymin_track {ymin_track:8.2f}, ymax_track  {ymax_track:8.2f}")
            paused = True
        return n60 // 2     # Integer division
        # return round(30*(y1*SPROCKET_HOLE_WIDTH - SPROCKET_DATA_OFFSET)/DATA_HOLE_WIDTH)

    def extract_row_data(self, frnum, row_detected):
        """
        Extract row data from a new detected row.

        frnum           current frame number of video being processed.
        row_detected    Row_Candidate value of detected row.
        """
        ymin, ymax = self.track_sprocket_holes(row_detected)
        row_data   = self.detect_data_holes(frnum, row_detected, ymin, ymax)
        self.row_data.append(row_data)
        return row_data

    # def _unused_track_sprocket_holes(self, row_detected):
    #     """
    #     This method analyses data from a new row of tape data, and updates and 
    #     returns estimates of the sprocket hole Y positions from row to row.

    #     It assumes that in the majority of rows, sprocket holes correspond to
    #     the minimum and maximum Y values seen, but allows that data noise may 
    #     mean that some rows contain data outside that range.

    #     The intent of the code is to provide a tracking value that converges 
    #     towards the observed sprocket hole positions, but allowing for some 
    #     row-to-row drift.  It assumes that in the majority of frames, the 
    #     sprocket holes correspond to the minimum and maximum Y coordinates of 
    #     the observed traces for any row.  At steady state, the tolerances for
    #     deviation from the previous row should be very tight, but in the early 
    #     stage they are looser to allow for recovery from noisy data early in 
    #     the process (see calculation of `ydev` below).

    #     @@NOTES:

    #     Arguably, a better approach might be to keep a history of the most recent
    #     N min- and max- Y values, and do a RANSAC-style linear fit and residual 
    #     calculation to exclude noisy data points, similar to the row detection.

    #     Alternatively, the logic here might be overkill and a simplified tracking 
    #     of filtered min- and max-Y values in a detected row might be enough
    #     (allowing that the row detection will have already eliminated many 
    #     noisy data values).

    #     row_detected    is a `row_candidate` value with all of the traces in a
    #                     new row to be processed.

    #     returns:        observed or estimated ymin and ymax values, which 
    #                     correspond to sprocket hole positions for the row.
    #     """
    #     traces  = row_detected.traces.rtraces
    #     ycoords = [ t.ycen for t in traces ]
    #     if self.num_track == 0:
    #         # Initial estimate is 1st row seen
    #         self.ymin_track = min(ycoords)
    #         self.ymax_track = max(ycoords)
    #         self.num_track  = 1
    #     # Look for traces closest above and below tracked min- and max- Y values
    #     ymint = self.ymin_track
    #     ymaxt = self.ymax_track
    #     yrngt = (ymaxt - ymint)
    #     yminl = ymint - yrngt
    #     yminh = ymint + yrngt
    #     ymaxl = ymaxt - yrngt
    #     ymaxh = ymaxt + yrngt
    #     for y in ycoords:
    #         if y < ymint and y > yminl: yminl = y
    #         if y > ymint and y < yminh: yminh = y
    #         if y < ymaxt and y > ymaxl: ymaxl = y
    #         if y > ymaxt and y < ymaxh: ymaxh = y
    #     # yminl and yminh are data coords closest below and above ymint
    #     # ymaxl and ymaxh are data coords closest below and above ymaxt
    #     #
    #     # Pick row data values that are closest to the tracking estimates
    #     yminr = ( yminh if yminh-ymint < ymint-yminl else yminl)
    #     ymaxr = ( ymaxh if ymaxh-ymaxt < ymaxt-ymaxl else ymaxl)
    #     # Reject data values that are not within (1%?) of the Y-coord range of
    #     # the currently tracked values in the limiting case of a long history.
    #     # For shorter tracking lengths, relax the constraints and favour
    #     # actual data that represents a wider range.
    #     old_weight = self.num_track
    #     new_weight = old_weight + 1
    #     ydev       = yrngt / old_weight # @@@TODO: tune this
    #     # @@TODO: the "*4" below is an attempt to allow the range to be 
    #     #         expanded in the early stages of tracking.  In practice, 
    #     #         this multiplier should converge to 1 as the number of
    #     #         tracked points increases.
    #     if ( (yminr < ymint-ydev*4) or (yminr > ymint+ydev) ):
    #         yminr = ymint
    #     if ( (ymaxr < ymaxt-ydev) or (ymaxr > ymaxt+ydev*4) ):
    #         ymaxr = ymaxt
    #     self.ymin_track = (self.ymin_track*old_weight + yminr)/new_weight
    #     self.ymax_track = (self.ymax_track*old_weight + ymaxr)/new_weight
    #     # New values contribute >= 2% when updating, where
    #     # steady-state 2% of tape width is less than the column pitch
    #     self.num_track  = max(new_weight, 49)
    #     # Return raw data points if selected, otherwise previous tracking value
    #     return yminr, ymaxr


    def track_sprocket_holes(self, row_detected):
        """
        This method analyses data from a new row of tape data, and updates and 
        returns estimates of the sprocket hole Y positions from row to row.

        It assumes that in the majority of rows, sprocket holes correspond to
        the minimum and maximum Y values seen, but allows that data noise may 
        mean that some rows contain data outside that range.

        The intent of the code is to provide a tracking value that converges 
        towards the observed sprocket hole positions, but allowing for some 
        row-to-row drift.  It assumes that in the majority of frames, the 
        sprocket holes correspond to the minimum and maximum Y coordinates of 
        the observed traces for any row.

        row_detected    is a `row_candidate` value with all of the traces in a
                        new row to be processed.

        returns:        observed or estimated ymin and ymax values, which 
                        correspond to sprocket hole positions for the row.
        """
        global paused, step
        traces  = row_detected.traces.rtraces
        ycoords = [ t.ycen for t in traces ]
        if self.num_track == 0:
            # Initial estimate is min, max values from 1st row seen
            self.ymin_track = min(ycoords)
            self.ymax_track = max(ycoords)
            self.num_track  = 1
        old_weight = self.num_track
        new_weight = old_weight + 1
        ymint      = self.ymin_track
        ymaxt      = self.ymax_track
        self.num_track  = min(new_weight, 30)

        # Look for traces closest above and below tracked min- and max- Y values
        yrngt = (ymaxt - ymint)
        yminl = ymint - yrngt
        yminh = ymint + yrngt
        ymaxl = ymaxt - yrngt
        ymaxh = ymaxt + yrngt
        for y in ycoords:
            if y <= ymint and y > yminl: yminl = y
            if y >= ymint and y < yminh: yminh = y
            if y <= ymaxt and y > ymaxl: ymaxl = y
            if y >= ymaxt and y < ymaxh: ymaxh = y

        # yminl and yminh are data coords closest below and above ymint
        # ymaxl and ymaxh are data coords closest below and above ymaxt
        #
        # Pick row data values that are closest to the tracking estimates
        yminr = ( yminh if (yminh-ymint < ymint-yminl) else yminl )
        ymaxr = ( ymaxh if (ymaxh-ymaxt < ymaxt-ymaxl) else ymaxl )

        # Check for large value swings.  In each case of ymin and ymax:
        # (a) if the deviation is too large, use last track value
        # (b) otherwise: updated tracked value
        ydev  = yrngt / old_weight      # Max deviation from tracked value
        if abs(yminr-ymint) > ydev:
            log_warning(f"ymin sprocket deviation: yminr {yminr:8.2f}, ymint {ymint:8.2f}, ydev {ydev:8.2f}")
            paused = True
            yminr = ymint
        else:
            self.ymin_track = (ymint*old_weight + yminr)/new_weight
        if abs(ymaxr-ymaxt) > ydev:
            log_warning(f"ymax sprocket deviation: ymaxr {ymaxr:8.2f}, ymaxt {ymaxt:8.2f}, ydev {ydev:8.2f}")
            paused = True
            ymaxr = ymaxt
        else:
            self.ymax_track = (ymaxt*old_weight + ymaxr)/new_weight

        # Return raw data points if selected, otherwise previous tracking value
        return yminr, ymaxr

    def detect_data_holes(self, frnum, row_detected, ymin, ymax):
        """
        Examine data in detected row and return data values corresponding to the
        detected markers.

        frnow           the current frame number - used for data display
        row_detected    is a `row_candidate` value with all of the traces in a
                        new row to be processed.
        ymin            observed or estimated minimum-Y value, used as the 
                        position of one of the sprocket holes for the row.
        ymax            observed or estimated maximum-Y value, used as the 
                        position of the other sprocket hole for the row.

        returns:        A `Row_Data` object indicating the detected holes in 
                        the supplied row.  The detected holes are represented as
                        an array of Boolean values.
        """
        yrng     = ymax - ymin 
        traces   = row_detected.traces.rtraces
        frnum    = round(statistics.fmean( ( (t.frnum + t.frend)/2 for t in traces ) ))
        xpos     = statistics.fmean( ( t.xcen for t in traces ) )
        ycoords  = [ t.ycen for t in traces if (t.ycen > ymin) and (t.ycen < ymax) ]
        row_data = Row_Data(frnum, xpos, ymin, ymax)
        for y in ycoords:
            y1  = (y - ymin)/yrng
            n30 = self.map_y1_n(y1)
            if n30 >= 0 and n30 <= 30:
                row_data.set_data(n30, True)
            #@@@@@
            # #p30 = y1*32.84 - 1.58
            # p60 = y1*65.68 - 3.16
            # n60 = round(p60)
            # if (n60 < 0) or (n60 > 60):
            #     # Data out of sprocket range
            #     log_warning(f"Detected hole out of range: {y:8.2f} [{ymin:8.2f}..{ymax:8.2f}]")
            # elif (n60%1 == 1):
            #     # Data between columns
            #     log_warning(f"Detected hole between columns: {y:8.2f} [{ymin:8.2f}..{ymax:8.2f}]")
            # else:
            #     n30 = n60 // 2     # Integer division
            #     row_data.set_data(n30, True)
            #@@@@@
        return row_data


# ----------------------------------------- #
# -- Analysis pipeline functions ---------- #
# ----------------------------------------- #

# Look for completed row of tape data in trace data.
#
#
#
# frnum                 frame number currently processed.  No new traces may end before this frame number.
# trace_data            is a list of closed region traces, ordered by ending frame number, which have not yet
#                       been definitively determined as belonging to a completed row, or no row at all.
#
# Returns:
#
# new_rows              is a list of new completed rows, or empty if none can be determined from
#                       the supplied trace data.
# updated_trace_data    is the supplied trace data excluding any region traces that have been detected 
#                       as belonging to a completed row, or determined to be not belonging to any row.
#

"""
Overview of row detection...

1. sort into overlapping trace-sets

2. if any trace appears in more than one set, decide which is the best fit

3. if any trace is unassigned, look to see if there is a good skew-fit

4. For longer traces, there should be less allowance for non-overlap 
   - how to characterize this?
"""

def find_initial_row_candidates(frnum, closed_traces):
    """
    Find an initial set of row candidates among the supplied traces.

        row_candidates = find_initial_row_candidates(frnum, traces)

    frnum       is the current frame number
    traces      is the current set of traces that have not yet been determined 
                as belonging to any row, or to no row at all.

    Returns a list of `row_candidate` objects, each of which contains a
    `region_trace_set` value.
    """
    row_candidates = []
    for t in closed_traces:
        new_candidates  = []
        used_candidate  = False
        extend_existing = False
        for rc in row_candidates:
            match_candidate, new_candidate = rc.row_initial_trace(t)
            if match_candidate:
                used_candidate  = True
                extend_existing = True
            if new_candidate:
                new_candidates.append(new_candidate)
                used_candidate = True
        # Add new candidate(s) if trace didn't fully overlap any existing candidate
        if not extend_existing:
            row_candidates.extend(new_candidates)
        if not used_candidate:
            row_candidates.append(row_candidate(region_trace_set(trace=t)))
    return row_candidates

def find_extended_row_candidates(frnum, row_candidates, traces):
    """
    Attempt to add further traces to each of the row candidates.

        row_candidates = find_extended_row_candidates(frnum, row_candidates, traces)

    At this stage, a single trace may appear in multiple row candidates.
    Only one of the resulting row candidates will be selected for further
    processing (see `find_preferred_row_candidate`)
    """
    for t in traces:
        for r in row_candidates:
            if r.row_additional_trace(t):
                # t added to r
                pass
    return row_candidates

def trace_overlap(row_candidate, row_candidates):
    """
    Does any trace in `row_candidate` appear in any of `row_candidates`?
    """
    # NOTE: multiple iterations over `row_candidates` - don't use generator.
    for t in row_candidate:
        for c in row_candidates:
            if t in c:
                return True
    return False

def find_preferred_row_candidate(frnum, row_candidates):
    """
    Select a preferred row candidate from all those assembled.
    """
    log_info(f"## find_preferred_row_candidate ## frnum {frnum:d}")
    unavailable_rows = []
    available_rows   = []
    for c in row_candidates:
        if not c.row_complete(frnum):
            unavailable_rows.append(c)
        elif c.residual < MAX_ROW_RESIDUAL:
            c.log(frnum, prefix=f"Row candidate")
            available_rows.append(c)
    preferred_res = MAX_ROW_RESIDUAL*2
    preferred     = None

    # The following logic looks for a pair or row candidates, the sum of whose
    # residuals is less than the sum of all other candidate pairs.
    # The first of these to appear is selected.
    #
    # This logic is an attempt to take account of additional, better candidate 
    # rows that might otherwise be blocked by selecting the best single candidate.
    for c1 in available_rows:
        for c2 in available_rows:
            if not trace_overlap(c2, [c1]):
                pair_residual = c1.residual + c2.residual
                if pair_residual < preferred_res:
                    # NOTE: values that appear to be equal can pass the above test due to floating
                    #   point rounding errors.  While not ideal, it doesn't invalidate the logic.
                    if not ( trace_overlap(c1, unavailable_rows) or 
                             trace_overlap(c2, unavailable_rows)
                           ):
                        # Select this candidate pair if no overlap with any incomplete row
                        c1.log(frnum, prefix="Select1")
                        c2.log(frnum, prefix="Select2")
                        preferred_res = pair_residual
                        preferred     = c1 if c1.traces.minfrbeg <= c2.traces.minfrbeg else c2
                    elif preferred and trace_overlap(preferred, [c1,c2]):
                        # Deselect current preferred row as there may be a better one we cannot use yet
                        preferred.log(frnum, prefix="Deselect")
                        preferred     = None


    # for c in row_candidates:
    #     if c.row_complete(frnum):
    #         if (c.residual < preferred_res):
    #             c.log(frnum, prefix=f"Try candidate (res < {preferred_res:6.3f})")
    #             if not trace_overlap(c, unavailable_rows):
    #                 # Select this candidate if no overlap with any incomplete row
    #                 c.log(frnum, prefix="Select")
    #                 preferred_res = c.residual
    #                 preferred     = c
    #             elif preferred and trace_overlap(c, [preferred]):
    #                 # Deselect current preferred option as there is a better one we cannot use yet
    #                 preferred.log(frnum, prefix="Deselect")
    #                 preferred     = None

    return preferred

def remove_spurious_traces(frnum, traces):
    """
    Returns list of traces that may still be considered "active".

        active_traces = remove_spurious_traces(frnum, traces)
    """
    return [ t for t in traces if t.active(frnum) ]

    # active_traces = set()
    # for c in row_candidates:
    #     if not c.row_orphaned(frnum):
    #         for t in c:
    #             active_traces.add(t)
    #     else:
    #         c.log(frnum, prefix=f"Orphaned row")
    # return list(active_traces)

def draw_rows(
        frame_show, frnum, rows, colour_border, 
        colour_fill1, colour_fill2=None, 
        colour_trace1=None, colour_trace2=None
        ):
    colour_fill,  colour_fill_next  = (colour_fill1,  colour_fill2  or colour_fill1)
    colour_trace, colour_trace_next = (colour_trace1, colour_trace2 or colour_trace1)
    for row in rows:
        row.draw(frame_show, frnum, colour_border, colour_fill)
        for rt in row:
            rt.draw(frame_show, frnum, colour_trace)
        colour_fill,  colour_fill_next  = (colour_fill_next,  colour_fill) 
        colour_trace, colour_trace_next = (colour_trace_next, colour_trace)
    return

def draw_new_rows(frame_show, frnum, rows):
    # NOTE colour channels are B,G,R:
    # see https://docs.opencv.org/4.5.3/d8/d01/group__imgproc__color__conversions.html#ga397ae87e1288a81d2363b61574eb8cab
    colour_border = (204, 255, 204)     # Light green
    colour_new1   = (102, 255, 255)     # Light yellow
    colour_new2   = (255, 255, 102)     # Light blue
    colour_trace1 = (  0,   0, 255)     # Red  (Outline traces in odd new rows)
    colour_trace2 = (255,   0,   0)     # Blue (Outline traces in even new rows)
    draw_rows(
        frame_show, frnum, rows, colour_border, colour_new1, colour_new2, colour_trace1, colour_trace2)
    return

def draw_old_rows(frame_show, frnum, rows):
    # NOTE colour channels are B,G,R:
    # see https://docs.opencv.org/4.5.3/d8/d01/group__imgproc__color__conversions.html#ga397ae87e1288a81d2363b61574eb8cab
    colour_border = (  0, 102,   0)     # Green
    colour_old1   = (102, 102,   0)     # Yellow/green colour old rows (odd)
    colour_old2   = (  0, 102, 102)     # Blue/green   colour old rows (even)
    colour_trace1 = (255, 255,   0)     # Outline traces in odd rows
    colour_trace2 = (  0, 255, 255)     # Outline traces in even rows
    draw_rows(frame_show, frnum, rows, colour_border, colour_old1, colour_old2, colour_trace1, colour_trace2)
    return

# @@TODO - remove all the "show_*" methods and use draw_* in the main program.

# Render video frame of highlight mask, region centroids, detected rows, and unused traces
# Does not display the video frame.
#
# frame_number  is a frame number to be displayed
# frame_data    is raw video data of the mask frame to be displayed
# rcoords       is the region coordinates to be displayed for the current frame
# unused_traces is a list of unassigned region traces to be displayed
# row_traces    is a list of assigned (to row) region traces to be displayed
# new_rows      is a list of traces to be highlighted as newly detected rows
# old_rows      is a list of traces to be highlighted as previously detected rows
#
# Returns displayed video frame
#
def draw_video_frame_mask_region_traces_rows(
        frame_number, frame_data, rcoords, unused_traces, row_traces, new_rows, old_rows
    ):
    # Display mask frame with centroid data, and return displayed frame value
    frame_show = cv.cvtColor(frame_data, cv.COLOR_GRAY2RGB)
    draw_region_centroids(frame_show, rcoords)
    draw_old_rows(frame_show, frame_number, old_rows)
    draw_new_rows(frame_show, frame_number, new_rows)
    colour_green  = (0, 255, 0)
    colour_yellow = (0, 255, 255)
    draw_region_traces(frame_show, frame_number, unused_traces, colour_green)
    # draw_region_traces(frame_show, frame_number, row_traces, colour_yellow)
    return frame_show

def draw_row_data(frame_show, frame_number, row_data_accum):
    colour_sprocket = (0, 128, 255)       # Orange - colour for sprocket holes
    colour_holes    = (0, 128, 255)       # Orange - colour for data
    colour_fills    = (0, 128, 128)       # Dull orange? - colour for data
    for rd in row_data_accum.row_data:
        rd.draw(frame_show, frame_number, colour_sprocket, colour_holes, colour_fills, row_data_accum)
    return frame_show


# ----------------------------------------- #
# -- Main program ------------------------- #
# ----------------------------------------- #

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

    open_traces    = []     # Buffer of traces still being assembled
    closed_traces  = []     # Buffer of traces assembled but not sorted
    used_traces    = []     # Buffer of traces that have been assigned to rows
    old_rows       = []     # Rows detected in previous frames (cumulative)
    draw_new_rows  = []     # Recently detected rows to be highlighted

    row_data_accum = Row_Data_Accumulator()     # Data extracted from detected rows

    while True:

        if not paused:
            # Read frame
            frame_number, frame = read_video_frame(video_capture)
            if frame is None:
                break

            if frame_number < START_FRAME:
                continue

            if frame_number in PAUSE_FRAMES:
                paused = True

            log_info(f"\n***** Frame number {frame_number:d} *****")

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

            # When new traces are detected...
            new_rows = []
            if len(new_traces) > 0:
                # Update buffer of traces waiting to be sorted into rows
                closed_traces = region_trace_add(frame_number, new_traces, closed_traces)
                closed_traces = remove_spurious_traces(frame_number, closed_traces)
                while (True):
                    log_info(f"## Assemble and test candidates ## frame_number {frame_number:d}, num traces {len(closed_traces):d}")
                    for t in closed_traces:
                        log_info(t.long_str(prefix="  "))

                    # Adding just one new row for each pass through this loop ensures that
                    # no trace is assigned to more than one row/
                    #
                    # Build an initial set of row candidates
                    row_candidates = find_initial_row_candidates(frame_number, closed_traces)
                    # for c in row_candidates:
                    #     c.log(frame_number, prefix=f"Initial candidate")
                    # Add extra traces to row candidates
                    row_candidates = find_extended_row_candidates(frame_number, row_candidates, closed_traces)
                    # Select strongest candidate
                    row_candidate = find_preferred_row_candidate(frame_number, row_candidates)
                    if row_candidate == None:
                        break   # No more for now
                    # Use preferred candidate and remove traces from unallocated traces
                    log_info(f"Preferred: residual {row_candidate.residual:6.3f}")
                    log_info(row_candidate.long_str(prefix="  ", t_prefix="    "))
                    new_rows.append(row_candidate)
                    for t in row_candidate:
                        closed_traces.remove(t)
                        used_traces.append(t)

            # Show closed region traces
            # log_region_traces(frame_number, new_traces)
            # frame_show_traces = show_video_frame_mask_region_traces(
            #     "Traces", frame_number, frame_highlights, 
            #     filtered_region_coords_list, closed_traces
            #     )

            # Show row traces
            if len(new_rows) > 0:
                for r in new_rows:
                    r.log(frame_number, logmethod=print)
                draw_new_rows = new_rows
            frame_rows = draw_video_frame_mask_region_traces_rows(
                frame_number, frame_highlights, 
                filtered_region_coords_list, closed_traces, used_traces,
                draw_new_rows, old_rows
                )

            # Move new rows to old
            old_rows.extend(new_rows)

            # Extract data from rows
            # NOTE: new_rows is a list of `row_candidate` values
            for r in new_rows:
                rd = row_data_accum.extract_row_data(frame_number, r)
                rd.log(frame_number, logmethod=print)
            draw_row_data(frame_rows, frame_number, row_data_accum)

            # Display video frame of visualized data
            frame_row_data = show_video_frame("row_data", frame_number, frame_rows)

            # Write the frame into the file 'output.avi'
            write_video_frame_pair(video_writer, frame_number, frame_show_orig, frame_row_data)

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

    # Extract and write final row data to a file.




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
