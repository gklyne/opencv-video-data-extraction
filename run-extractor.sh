# python video-04-region-coords.py --input=20210914-monotype-tape-reader.mov
# python video-05-region-trace.py --input=20210914-monotype-tape-reader.mov
# python video-06-row-detect.py --input=20210914-monotype-tape-reader.mov
# python video-07-row-detect-redux.py --input=20210914-monotype-tape-reader.mov
# python video-08-row-data-extract.py --input=20210914-monotype-tape-reader.mov
# python video-08-row-data-extract.py --input=/Users/graham/Work/Monotype-Compositions/20221024-IMG_2690.mov
# python video-08-row-data-extract.py --input=20230226-input-IMG_2977.MOV
# python video-08-row-data-extract.py --input=20230226-input-IMG_2977.MOV
# python video-08-row-data-extract.py \
#     --input=videos/20230308-IMG_2997-input.MOV \
#     --output=videos/20230308-IMG_2997-output.avi
# python video-08-row-data-extract.py \
#     --input=data/20230415-IMG_3071-colophon-input.MOV \
#     --output=data/20230415-IMG_3071-colophon-output.avi

# FILE_STEM="20230415-IMG_3071-colophon"
# FILE_STEM="20230416-IMG_3082-intro"
# FILE_STEM="20230924-VAD-spool-1-IMG_7192"
# FILE_STEM="20240127-VAD-1-in-time-of-war-IMG_7761"
# FILE_STEM="20240127-VAD-2-happenings-of-interest-IMG_7762"
FILE_STEM="20240127-VAD-3-when-on-duty-IMG_7776"

FILE_DIR="."

TIME_START=$(date)
SECONDS=0
python video-09-row-data-extract.py \
 --input=data/${FILE_STEM}-input.MOV \
 --output=${FILE_DIR}/${FILE_STEM}-output.avi \

TIME_END=$(date +"%T")
echo "Started $TIME_START, completed $TIME_END, duration ${SECONDS}s"

if [[ -f output.txt ]]; then
	mv output.txt ${FILE_DIR}/${FILE_STEM}-output.txt
	echo "Created ${FILE_DIR}/${FILE_STEM}-output.txt"
fi
if [[ -f output_errors.txt ]]; then
	mv output_errors.txt ${FILE_DIR}/${FILE_STEM}-output_errors.txt
	echo "Created ${FILE_DIR}/${FILE_STEM}-output_errors.txt"
fi

# https://stackoverflow.com/questions/8903239/how-can-i-calculate-time-elapsed-in-a-bash-script
