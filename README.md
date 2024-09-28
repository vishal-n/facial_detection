To run the scripts:

1. Run the command: ***git clone git@github.com:vishal-n/facial_detection.git***
2. Install the following libraries for python3: ***OpenCV, Numpy, Argparse***

3. To detect faces in images, run the command-
***python3 detect_faces.py --image rooster.jpg --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel***

4. To detect faces in videos, run the command-
***python3 detect_faces_video.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel***
