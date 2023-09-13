TODO.md

## Future

[ ] Eliminate over-long traces (cf., error about frame 5950)

[ ] Eliminate repeated row candidates when assembling from active traces (optimization)

[ ] Tune row-fitting parameters

[ ] ??? Track linear fit parameter estimates (bhat, ahat), and use these to eliminate "wild" fits.
    - maybe incorporate some kind of skew estimate into the residual for the 2-point case, 
      so that more nearly orthogonal sets are preferred?


## 2023-06-15: video-09-row-data-decode.py  

[x] Configure default IP address for RPi, or way to read it on boot, for stepper motor control on new networks

[ ] Make recordings of actual diary tapes/spools

[ ] Look into word separation

[ ] Look word "outline" shapes (per Dawn's wrapping word shapes onto canister)

[ ] Investigate creating data-driven objects using FreeCAD?

[ ] Blog post and telling the MC story...


## 2023-06-15: video-08-row-data-extract.py

[x] Output column codes in decoder output (see photo of Monotype decoder ruler)

[x] Using matrix array configuration, extract character data from decoded data

[x] Set up definition for Bulmer matrix array (see photo)

[x] Redesign reader device to be more uniform with less wobble


## 2022-05-03: video-08-row-data-extract.py

[x] Estimate and track sprocket hole Y-coordinates; use this to filter out spurious traces from row candidates.

[x] Overlay detection data on original video; display and display single frame

[x] Extract hole data from each row, and write to data file


## 2022-03-02: video-07-row-detect-redux.py

[x] Display frame numbers top and bottom of video (so they remain visible when player shows menu bar over the frame).

[x] Eliminate over-aged traces

[x] Have preferred selection consider multiple rows, so that spurious good linear fit doesn't force better combinations of row assignments to be rejected.  (e.g. look for best sum of residuals for a pair of non-overlapping row candidates, and choose earliest of those?) 

[x] Improve presentation of row bars to reflect varying width and skew

[x] If a preferred row candidate does not pass the completeness test, look for another that does.  This is to prevent new candidates from blocking or delaying recognition of older ones that should be accepted.

[x] Improve preferred match selection, giving greater weight to the value of the residual


