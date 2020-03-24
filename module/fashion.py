import cv2
import operator
from datetime import datetime


class FASHIONDetector(object):
    def __init__(self, yolo_fashion, collection):
        self.yolo_detector = yolo_fashion
        self.coll = collection
        self.accuracy_thresh = 0.6
        self.category_count = {}
        self.fashion_name = {}

    def detectFashion(self, frame):
        return self.yolo_detector.detect_image(frame)

    def process(self, frame, track_id):
        fashion_name = []

        do_detect = True
        if track_id in self.category_count.keys():
            id_names = self.category_count[track_id]
            fashion_frequency = sorted(id_names.items(), key=operator.itemgetter(1))
            # if len(id_names) == 1:
            #     if fashion_frequency[-1][1] > 5:
            #         do_detect = True
            if len(id_names) > 1:
                if fashion_frequency[-1][1] > 5 and fashion_frequency[-2][1] > 5:
                    do_detect = False

        if do_detect:
            bounding_boxes, names, accuracies = self.detectFashion(frame)

            for bounding_box, name, accuracy in zip(bounding_boxes, names, accuracies):
                if accuracy > self.accuracy_thresh:
                    if track_id not in self.category_count.keys():
                        self.category_count[track_id] = {}
                        self.category_count[track_id][name] = 1
                    else:
                        if name not in self.category_count[track_id].keys():
                            self.category_count[track_id][name] = 1
                        else:
                            self.category_count[track_id][name] += 1

            if track_id in self.category_count.keys():
                id_names = self.category_count[track_id]
                print(f'id_names : {id_names}')
                fashion_frequency = sorted(id_names.items(), key=operator.itemgetter(1))
                print(f'fashion_frequency : {fashion_frequency}')
                if len(id_names) == 1:
                    fashion_name = [fashion_frequency[-1][0]]
                if len(id_names) > 1:
                    fashion_name = [fashion_frequency[-1][0],
                                    fashion_frequency[-2][0]]
            self.fashion_name[track_id] = fashion_name
            self.process_db(track_id, fashion_name)
        else:
            fashion_name = self.fashion_name[track_id]

        return fashion_name

    def process_db(self, track_id, fashion_name):
        if self.coll.find_one({"id": track_id}) is None:
            now_time = datetime.now().strftime('%Y-%m-%d/%H:%M:%S').split('/')
            post = {
                "id": track_id,
                "date": now_time[0],
                "time": now_time[1],
                'Clothes': fashion_name,
            }
            self.coll.insert(post)
            print(f'DB Inserted fashion - {post}')
        else:
            post = {
                'Clothes': fashion_name,
            }
            self.coll.find_one_and_update({"id": track_id}, {"$set": post})
            print(f'DB updated fashion - {post}')

    def fashionDraw(self, frame, fashion_boxes, isDrawBoxes=True):
        for i, box in enumerate(fashion_boxes):
            track_id, people_box, fashion_box = box
            (xmin, ymin) = people_box[0]
            if fashion_box[0] != []:
                bounding_boxes, names, accuracies = fashion_box
                for bounding_box, name, accuracy in zip(bounding_boxes, names, accuracies):
                    if isDrawBoxes:
                        cv2.rectangle(frame,
                                      (xmin + bounding_box[0], ymin + bounding_box[1]),
                                      (xmin + bounding_box[0] + bounding_box[2], ymin + bounding_box[1] + bounding_box[3]),
                                      (255, 125, 0), 2)
                        cv2.putText(frame, f"[{name}_{accuracy}",
                                    (xmin + bounding_box[0], ymin + bounding_box[1] - 30),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 125, 0))
        return frame

    def cropAndsave(self, boxes, image, frameNumber, isSave):
        cropped_img = []
        for i, box in enumerate(boxes):
            (xmin, ymin), (xmax, ymax) = box
            cropped = image[ymin:ymax, xmin:xmax]
            cropped_img.append(cropped)
            if isSave:
                cv2.imwrite(f'crop/fashion/{frameNumber}_{i + 1}.jpg', cropped)
                print(f'{frameNumber}_{i + 1}.jpg   save success!')
        return cropped_img

    def identifyfashion(self, frame, fashionBoxes):
        for fashion in fashionBoxes:
            for box in fashion:
                name = box[0]
                left, top, right, bottom = box[1]
                y = top - 15 if top - 15 > 15 else top + 15
                if name == 'unknown':
                    color = (0, 0, 255)
                else:
                    color = (255, 0, 0)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
                # cv2.putText(frame, 'identified:' + str(self.fashionCounted), (int(len(frame[0]) - 170), 50),
                #             cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), thickness=1, lineType=2)
