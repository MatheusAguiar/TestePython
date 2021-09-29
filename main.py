import cv2
import sys
from random import randint

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
#print(major_ver, minor_ver, subminor_ver)

tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT']

tracker_type = tracker_types[1]
#print(tracker_type)

if int(minor_ver) < 3:
    tracker = tracker_type
else:
    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    if tracker_type == 'CSRT':
        tracker = cv2.TrackerCSRT_create()

    #print(tracker)

    video = cv2.VideoCapture('videos/race.mp4')
    if not video.isOpened():
        print('Não foi possivel carregar o video')
        sys.exit()

    ok, frame = video.read()
    if not ok:
        print('Não foi possivel ler o arquivo de video')
        sys.exit()
    #print(ok)

    bbox = cv2.selectROI(frame, False)
    #print(bbox)

    ok = tracker.init(frame, bbox)
    #print(ok)

    colors = (randint(0, 255), randint(0, 255), randint(0, 255))
    print(colors)
