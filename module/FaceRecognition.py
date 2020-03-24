import os
from datetime import datetime
import pickle
import cv2
import face_recognition
from module.FaceRecord import FACE


class FACEDetector:

    def __init__(self, collection):
        self.coll = collection
        self.classifier_name = 'module/face/models/face.pkl'
        self.face_detected = []
        self.classifier_data = pickle.loads(open(self.classifier_name, 'rb').read())

        self.save_dir = 'module/face/database'
        # self.last_face_number = self.get_last_face()
        self.last_face_number = 0
        self.scale_factor = 2
        self.tolerance = 0.9

        self.recog_count = 0

        # self.new_encodings = {}
        self.face = {}
        self.face["encodings"] = []
        self.face["names"] = []
        self.face["face_id"] = []
        self.face_encodings = []
        self.faces = []

    def get_last_face(self):
        list_dir = os.listdir(self.save_dir)
        if list_dir == []:
            return 0
        else:
            return int(float(sorted(list_dir, reverse=True)[0]))

    # def multi_process(self, frames, crop_boxes):
    #     for frame, crop_box in zip(frames, crop_boxes):
    #         self.process(frame, crop_box)

    def init_face(self):
        self.face = []
        self.recog_count = 0

    def process_db(self, face_id, encoding, name):
        if self.coll.find_one({"id": face_id}) is None:
            now_time = datetime.now().strftime('%Y-%m-%d/%H:%M:%S').split('/')
            post = {
                "id": face_id,
                "date": now_time[0],
                "time": now_time[1],
                'face_encoded': list(encoding),
                "name": name
            }
            self.coll.insert(post)
            print(f'DB Inserted face - {post}')
        else:
            post = {
                'face_encoded': list(encoding),
                "name": name
            }
            self.coll.find_one_and_update({"id": face_id}, {"$set": post})
            print(f'DB updated face - {post}')

    def process(self, crop_frame, face_id):
        if face_id not in self.face["face_id"]:
            frame_resize = cv2.resize(crop_frame, None, fx=self.scale_factor, fy=self.scale_factor)

            frame_resize = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(frame_resize, model='cnn')
            name = ''
            if len(boxes) > 0:
                encoding, name = self.identify_face(frame_resize, boxes)
                if encoding is not None:
                    self.face["face_id"].append(face_id)
                    self.face["encodings"].append(encoding)
                    self.face["names"].append(name)

                    self.process_db(face_id, encoding, name)

                    # cv2.imwrite(f'{self.save_dir}/{self.face["face_id"][-1]}.jpg', frame_resize)
        else:
            matchedIdx = self.face['face_id'].index(face_id)
            name = self.face['names'][matchedIdx]
        return name

    def process_point(self, crop_frame, crop_box, face_id):
        # try:
        frame_resize = cv2.resize(crop_frame, None, fx=self.scale_factor, fy=self.scale_factor)
        # except:
        #     print(crop_frame)

        frame_resize = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(frame_resize, model='cnn')

        self.recog_count += 1

        top, right, bottom, left = boxes[0]
        top = int(top / self.scale_factor) + crop_box[0][1]
        right = int(right / self.scale_factor) + crop_box[0][0]
        bottom = int(bottom / self.scale_factor) + crop_box[0][1]
        left = int(left / self.scale_factor) + crop_box[0][0]

        # draw the predicted face name on the image
        y = top - 15 if top - 15 > 15 else top + 15

        if face_id not in self.face["face_id"]:
            encoding, name, color = self.identify_face(frame_resize, boxes)
            if encoding is not None:
                self.face["encodings"].append(encoding)
                self.face["names"].append(name)
                self.face["face_id"].append(face_id)
                # self.face_ids[face_id].append({
                #     'encodings': encoding,
                #     'names': name
                # })
                cv2.imwrite(f'{self.save_dir}/{self.face["face_id"][-1]}.jpg', frame_resize)
        else:
            matchedIdx = self.face['face_id'].index(face_id)
            name = self.face['name'][matchedIdx]
            color = (255, 0, 0)

        return [(left, top), (right, bottom), y, color, name]

    # def register_face(self, encoding, face_id):
    #     self.new_encodings[face_id] = encoding
    #     self.face_ids[face_id] = 0
    #     self.last_face_number += 1

    def identify_face(self, frame_resize, boxes):
        encodings = face_recognition.face_encodings(frame_resize, boxes, num_jitters=10)

        encoding = None
        name = f'p_{self.last_face_number}'
        if encodings != []:
            encoding = encodings[0]
            # matches = face_recognition.compare_faces(self.classifier_data["encodings"], encoding,
            #                                          tolerance=self.tolerance)
            # if True in matches:
            #     matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            #     name = self.classifier_data['names'][matchedIdxs[0]]
            #
            # if True not in matches:
            matches = face_recognition.compare_faces(self.face["encodings"], encoding, tolerance=self.tolerance)

            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                name = self.face["names"][matchedIdxs[0]]

        if name == f'p_{self.last_face_number}':
            self.last_face_number += 1
            name = f'p_{self.last_face_number}'
        if name == 'han':
            name = 'chaewon'
        return encoding, name

    def face_draw(self, frame, face_boxes):
        # box : [(left, top), (right, bottom), y, color, name]
        for box in face_boxes:
            cv2.rectangle(frame, box[0], box[1], box[3], 2)
            cv2.putText(frame, box[4], (box[0][0], box[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, box[3], 2)
        return frame

    def train(self):
        print("[INFO] writing face encodings")
        face_data = {"encodings": self.face["encodings"], "names": self.face["names"]}
        f = open(self.classifier_name, 'wb')
        f.write(pickle.dumps(face_data))
        f.close()


    # def compare_encodings(self):
    #     if True not in matches:
    #         if self.new_encodings != []:
    #             new_matches = face_recognition.compare_faces(self.new_encodings, encoding)
    #             if True in new_matches:
    #                 self.last_face_number
    #
    #         else:
    #             self.new_encodings.append(encoding)
    #             os.mkdir(f'{self.save_dir}/{self.last_face_number}')
    #
    #             self.last_face_number += 1

    # def draw_status(self, frame):
    #     cv2.putText(frame, 'identified:' + str(self.recog_count), (int(len(frame[0]) - 170), 50),
    #                 cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), thickness=1, lineType=2)
    #     cv2.putText(frame, 'unknown:' + str(len(boxes) - self.recog_count), (int(len(frame[0]) - 170), 70),
    #                 cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=1, lineType=2)
