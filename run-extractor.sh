# python video-04-region-coords.py --input=20210914-monotype-tape-reader.mov
# python video-05-region-trace.py --input=20210914-monotype-tape-reader.mov
# python video-06-row-detect.py --input=20210914-monotype-tape-reader.mov
# python video-07-row-detect-redux.py --input=20210914-monotype-tape-reader.mov
# python video-08-row-data-extract.py --input=20210914-monotype-tape-reader.mov
# python video-08-row-data-extract.py --input=/Users/graham/Work/Monotype-Compositions/20221024-IMG_2690.mov
# python video-08-row-data-extract.py --input=20230226-input-IMG_2977.MOV



TIME_START=$(date)
SECONDS=0
python video-08-row-data-extract.py --input=20230308-IMG_2997-input.MOV
TIME_END=$(date +"%T")
echo "Started $TIME_START, completed $TIME_END, duration ${SECONDS}s"

# https://stackoverflow.com/questions/8903239/how-can-i-calculate-time-elapsed-in-a-bash-script





# Finished processing: writing output.txt (4836 rows from 20590 frames)
# Written 4836 rows of data to file output.txt
# closing down
# video streams closed
# end of main
# Started Sat  4 Mar 2023 17:26:55 GMT, completed 19:23:45, duration 7010s
