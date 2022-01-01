# opencv-video-data-extraction

Experiments with using OpenCV with Python to read data from a video recording


## Setup

- installed xcode (can skip this?)

- installed command line tools (`xcode-select --install`)

- installed python-3.9 (from .pkg file at https://www.python.org/ftp/python/3.9.7/python-3.9.7-macosx10.9.pkg)


Then:


    $(which python3.9) -m venv cvenv
    . cvenv/bin/activate

    pip install --upgrade pip
    pip install numpy
    pip install opencv-python


## Run the video extraction

    python video-04-region-coords.py --input=20210914-monotype-tape-reader.mov

