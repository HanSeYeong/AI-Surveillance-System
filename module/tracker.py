from module.deep_sort import preprocessing
from module.deep_sort import nn_matching
from module.deep_sort.detection import Detection
from module.deep_sort.tracker import Tracker
from module.tools import generate_detections as gdet

from module.intersectingLine import calcParams, areLinesIntersecting

# from module.face_mtcnn import FACEDetector
# from module.tf_insightface import FACEDetector
from module.FaceRecognition import FACEDetector
from module.fashion import FASHIONDetector

import cv2
import numpy
import time


class Tracking:
    def __init__(self, camID, url, saveOUT, yolo, yolo_fashion, collection):
        # self.cap = FileVideoStream(url)
        # self.cap.start()
        self.collection = collection
        self.cap = cv2.VideoCapture(url)

        self.ID = camID
        self.saveOUT = saveOUT
        self.setVideoOut()

        # line = [[250, 400], [1200, 400]]
        # self.startLine = calcParams([int(250/3), int(400/3)], [int(1200/3), int(400/3)])
        # self.startLinePoint = [(int(250/3), int(400/3)), (int(1200/3), int(400/3))]

        self.startLine = calcParams([250, 400], [1200, 400])
        self.startLinePoint = [(250, 400), (1200, 400)]

        max_cosine_distance = 0.8
        nn_budget = 5
        self.nms_max_overlap = 0.45
        # model_filename = resource_path('module/deep_sort/mars-small128.pb')
        model_filename = 'module/deep_sort/mars-small128.pb'
        self.encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric("euclidean", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric)

        self.yolo = yolo

        self.faceDetector = FACEDetector(collection)
        self.fashionDetector = FASHIONDetector(yolo_fashion, collection)

        print('Start Recognition!')

        self.trackYOLO = []
        self.counted_ids = []
        self.countedPerson = []
        self.frames = []
        self.face_boxes = {}
        self.fashion_boxes = {}
        self.frameCount = 0
        self.maxAge = 5
        self.maxCount = 6

        self.personCounted = 0
        self.faceCounted = 0

    def cropAndsave(self, boxes, image, frameNumber, isSave):
        fashion_cropped_img = []
        face_cropped_img = []
        for i, box in enumerate(boxes):
            (xmin, ymin), (xmax, ymax) = box
            cropped = image[ymin:ymax, xmin:xmax]
            fashion_cropped_img.append(cropped)
            face_cropped_img.append(image[ymin:int(ymax/2), xmin:xmax])
            if isSave:
                cv2.imwrite(f'crop/face/{frameNumber}_{i + 1}.jpg', cropped)
                print(f'{frameNumber}_{i + 1}.jpg   save success!')
        return fashion_cropped_img, face_cropped_img

    def setVideoOut(self):
        if self.saveOUT:
            name = f'camID_{self.ID}.mp4'
            # files = glob.glob('*.mp4')
            # existed_files = [os.path.basename(file) for file in files]
            # if name in files:
            self.out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*"h264"), 30, (1920, 1080))

    # def detecting(self, frames):
    def detecting(self, image):
        ret = True
        if ret:
            Boxes, names, accuracies = self.yolo.detect_image(image)
            # frame = cv2.resize(image.copy(), (640, 360))
            frame = image.copy()
            if Boxes != []:
                centers = []
                trackIds = []
                detectedBoxes = []
                Features = self.encoder(frame, Boxes)
                Detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(Boxes, Features)]
                boxes = numpy.array([d.tlwh for d in Detections])
                scores = numpy.array([d.confidence for d in Detections])
                indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
                detections = [Detections[i] for i in indices]

                self.tracker.predict()
                self.tracker.update(detections)

                for track in self.tracker.tracks:
                    if track.is_confirmed() and track.time_since_update > 1:
                        continue
                    xmin, ymin, xmax, ymax = [int(point) for point in (track.to_tlbr())]

                    detectedBoxes.append([(int(xmin), int(ymin)), (int(xmax), int(ymax))])
                    # fashion_crops, face_crops = self.cropAndsave(detectedBoxes, image, self.frameCount, isSave=False)
                    # fashion_cropped = self.fashionDetector.cropAndsave(detectedBoxes, image, self.frameCount, isSve=False)

                    center = (int((xmax + xmin) / 2), int((ymax + ymin) / 2))
                    centers.append(center)
                    trackIds.append(track.track_id)

                    isSame = False
                    for (i, trackYOLO) in enumerate(self.trackYOLO):
                        if trackYOLO[0] == track.track_id:
                            self.trackYOLO[i].append(center)
                            isSame = True
                            break
                    if isSame is False:
                        self.trackYOLO.append([track.track_id, 0, [], detectedBoxes[-1], center])

                for (i, tracked) in enumerate(self.trackYOLO):
                    if tracked[0] in trackIds:
                        lastCenter = (tracked[4][0], tracked[4][1])
                        center = tracked[-1]
                        newLine = calcParams(lastCenter, center)
                        if tracked[0] not in [self.countedPerson[x][0] for x in range(len(self.countedPerson))]:
                            if areLinesIntersecting(self.startLine, newLine,
                                                    lastCenter, center, self.startLinePoint):
                                print('success to cross in ', tracked[0])
                                self.personCounted += 1

                                # lane = self.determineLane(center)
                                self.counted_ids.append(tracked[0])
                                self.countedPerson.append([tracked[0], 0, self.frameCount])
                                self.frames.append(frame[tracked[3][0][1]:tracked[3][1][1],
                                                   tracked[3][0][0]:tracked[3][1][0]])
                                # area = round(tracked[2], -3)
                                # if area == 0:
                                #     area = round(tracked[2], -2)

                if self.frameCount % 2 == 0:
                    fashion_crops, face_crops = self.cropAndsave(detectedBoxes, image, self.frameCount, isSave=False)
                    face_boxes = {}
                    fashion_boxes = {}
                    # fashion_boxes = []
                    for (people_box, fashion_crop, face_crop, track_id) \
                            in zip(detectedBoxes, fashion_crops, face_crops, trackIds):
                        if track_id in self.counted_ids:
                            if face_crop != []:
                                face_name = self.faceDetector.process(face_crop, track_id)
                                if face_name != '':
                                    face_boxes[track_id] = face_name
                            fashion_name = self.fashionDetector.process(fashion_crop, track_id)
                            if len(fashion_name) != 0:
                                fashion_boxes[track_id] = fashion_name

                    if self.face_boxes == {}:
                        self.face_boxes = face_boxes
                    if self.fashion_boxes == {}:
                        self.fashion_boxes = fashion_boxes

                    self.face_boxes = face_boxes
                    self.fashion_boxes = fashion_boxes

                if len(self.countedPerson) > 15:
                    del self.countedPerson[:-15]
                    del self.frames[:-15]

                # if hide is False:
                frame = self.drawRectangle(detectedBoxes, trackIds, frame)

                for (i, track) in enumerate(self.trackYOLO):
                    if track[0] not in trackIds:
                        self.trackYOLO[i][1] += 1
                        self.trackYOLO[i].append(track[-1])

                    if self.trackYOLO[i][1] > self.maxAge:
                        del self.trackYOLO[i]

                for (i, track) in enumerate(self.trackYOLO):
                    if len(track) > self.maxCount:
                        del self.trackYOLO[i][4:-(self.maxCount - 1)]

                    # if hide is False:
                    try:
                        # if self.hide is False:
                        if len(track) > 5:
                            cv2.polylines(frame, [numpy.int32(track[4:])], False, (0, 255, 0), 2)
                        cv2.putText(frame, str(track[0]), track[-1], cv2.FONT_HERSHEY_COMPLEX, 1,
                                    (0, 0, 255))
                        if track[2] != []:
                            cv2.putText(frame, "[" + str(track[2][0]) + ", " + str(int(track[2][1] * 100)) + "%]",
                                        (track[-1][0], track[-1][1] - 30), cv2.FONT_HERSHEY_TRIPLEX, 0.8,
                                        (0, 0, 255))
                    except:
                        print('drawing')

                # self.drawFace(frame, faceBoxes)

                if len(self.trackYOLO) > 20:
                    del self.trackYOLO[:-20]

            self.frameCount += 1
            cv2.putText(frame, f'count : {self.personCounted}', (400, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            cv2.line(frame, self.startLinePoint[0], self.startLinePoint[1], (0, 255, 0), 4)

            if self.saveOUT:
                self.out.write(frame)
            return self.ID, frame
        else:
            return self.ID, None

    def drawRectangle(self, detectedBoxes, trackIds, frame):
        for track_id, box in zip(trackIds, detectedBoxes):
            cv2.rectangle(frame, box[0], box[1], (255, 0, 0), 2)
            if track_id in self.face_boxes.keys():
                cv2.putText(frame, f'{self.face_boxes[track_id]}', (box[1][0] + 10, box[0][1]),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            if track_id in self.fashion_boxes.keys():
                cv2.putText(frame, f'{self.fashion_boxes[track_id]}', (box[1][0] + 10, box[0][1] + 30),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        return frame

    def updateFPS(self):
        fps = 0
        temp = time.time() - self.startTime
        if temp > 1:
            fps = self.frameCounter / temp
            self.frameCounter = 0
            self.startTime = time.time()
        self.frameCounter += 1
        return fps

    def destroy(self):
        # self.cap.destroy()
        self.out.release()
