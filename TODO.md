TODO.md

## 2022-03-02: video-07-row-detect-redux.py

[ ] Eliminate over-long traces

[ ] Eliminate over-aged traces

[ ] Display frame numbers top and bottom of video (so they remain visible when player shows menu bar over the frame).

[ ] Eliminate single-trace row candidates

[ ] Tune row-fitting parameters

[ ] Estimate and track sprocket hole Y-coordinates; use this to filter out spurious traces from row candidates.

[ ] ??? Track linear fit parameter estimates (bhat, ahat), and use these to eliminate "wild" fits.
    - maybe incorporate some kind of skew estimate into the residual for the 2-point case, 
      so that more nearly orthogonal sets are preferred?

[x] Have preferred selection consider multiple rows, so that spurious good linear fit doesn't force better combinations of row assignments to be rejected.  (e.g. look for best sum of residuals for a pair of non-overlapping row candidates, and choose earliest of those?) 

[x] Improve presentation of row bars to reflect varying width and skew

[x] If a preferred row candidate does not pass the completeness test, look for another that does.  This is to prevent new candidates from blocking or delaying recognition of older ones that should be accepted.

[x] Improve preferred match selection, giving greater weight to the value of the residual


