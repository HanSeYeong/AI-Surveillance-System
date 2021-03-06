from threading import Thread
from module.tracker import Tracking

import cv2

from pymongo import MongoClient

from module.darknet_model.yolo import YOLO
from module.darknet_model.yolo_fashion import YOLO_fashion


client = MongoClient('localhost', 27017)
print(client)

db = client["CNU_AI_Surveillance"]
collection = db['0324']

yolo = YOLO()
yolo_fashion = YOLO_fashion()

# urls = ['rtsp://admin:ImageLab@@@imagelab602.iptime.org:3001/h264/ch33/main/av_stream',
#         'rtsp://admin:ImageLab@@@imagelab602.iptime.org:3002/h264/ch33/main/av_stream',
#         'rtsp://admin:ImageLab@@@imagelab602.iptime.org:3003/h264/ch33/main/av_stream',
#         'rtsp://admin:ImageLab@@@imagelab602.iptime.org:3004/h264/ch33/main/av_stream']
urls = ['videos/cam0.mp4']

saveOUT = True
tracker = [Tracking(camID, url, saveOUT, yolo, yolo_fashion, collection) for camID, url in enumerate(urls)]

frames = []


cap = cv2.VideoCapture(urls[0])
start_frame = 32 * 30
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
ID = 1
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        ID, frame = tracker[0].detecting(frame)
        cv2.imshow(f'frame {ID}', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
for track in tracker:
    track.destroy()

