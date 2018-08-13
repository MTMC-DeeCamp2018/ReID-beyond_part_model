import cv2
import numpy as np
import os

video_path = "../../Dataset/epfl/terrace2-c0.avi"
bbox_path = "../../Dataset/epfl/bbox.txt"
store_dir = "./people/s2c0"

# read video
cap = cv2.VideoCapture(video_path)
fid, frame = 0, None

pid = 0
with open(bbox_path, 'r') as f:
    line = f.readline()
    while line:
        id, x, y, w, h, conf = line.strip().split()
        id, x, y, w, h, conf = int(id), float(x), float(y), float(w), float(h), float(conf)
        x1, y1 = int(np.round(max(0, x))), int(np.round(max(0, y)))
        x2, y2 = int(np.round(max(0, x + w))), int(np.round(max(0, y + h)))

        while fid < id:
            succ, frame = cap.read()
            assert(succ)
            fid += 1

        p = frame[y1:y2, x1:x2]

        cv2.imwrite(os.path.join(store_dir, str(pid).zfill(10)+'.png'), p)
        pid += 1
        line = f.readline()
        print(fid, end='\r')