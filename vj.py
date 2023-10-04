#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

import socket
import time
# 172.168.0.189
# 172.168.0.189
rover_host = '172.168.3.113'  # Replace with the IP address or hostname of the rover
rover_port = 7  # Replace with the port number the rover is listening on
rover_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
rover_socket.connect((rover_host, rover_port))
print("Connection successful")
lastrequesttime=time.time()

# client1_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# host1 = "172.168.3.157"
# port1 = 3000
# client1_socket.connect((host1, port1))
# print("Connection success")
# client1_socket.send(b'd')





def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=1200)
    parser.add_argument("--height", help='cap height', type=int, default=1000)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",        
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


def main():

    display_device = 1  # Index of the back camera
    display_cap = cv.VideoCapture(display_device)
   
    # Argument parsing #################################################################
    print(" 1. Rover -> Right \n Arm -> Left \n\n 2. Rover -> Left \n Arm -> Right \n")
    inp=int(input(("Enter your choice ")))
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True
    # web cam
    display_device = 1 
    display_cap = cv.VideoCapture(display_device)
    # 
    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    point_history_classifier = PointHistoryClassifier()

    # Read labels ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=history_length)

    #  ########################################################################
    mode = 0

    while True:
        fps = cvFpsCalc.get()

        ret1, frame1 = display_cap.read()
        if not ret1:
            break


        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # webcam
        image1 = cv.flip(frame1, 1)  # Mirror display
        debug_image1 = copy.deepcopy(frame1) #flip removed
        cv.rectangle(debug_image1, (224, 120), (416, 360), (255,255, 255), 2) 
        # additional rover not armed RED
        # while True: 
        #     # conn, addr = .accept()
        #     # print(f"Connected to gesture recognition at {addr[0]}:{addr[1]}")

        #     while True:
        #         data = rover_socket.recv(1024)
        #         if not data:
        #             break
        #         command = data.decode()
        #         if(command=='unarmed'):
        #             cv.rectangle(debug_image1, (224, 120), (416, 360), (0,0, 255), 2) 
                
        #  
        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Bounding box calculation
                # brect = calc_bounding_rect(debug_image,debug_image1, hand_landmarks, handedness,inp)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)
                # Write to the dataset file
                logging_csv(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 2:  # Point gesture
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                # Calculates the gesture IDs in the latest detection
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()

                brect = calc_bounding_rect(debug_image,debug_image1, hand_landmarks, handedness,inp,keypoint_classifier_labels[hand_sign_id])
                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image,debug_image1, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )
        else:
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)

        # Screen reflection #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image1)

    cap.release()
    cv.destroyAllWindows()


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calc_bounding_rect(image,image1, landmarks,handedness,inp,sig):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    # For the no movement boundry
    image_height, image_width = image.shape[:2]
    box_width = int(image_width * 0.3)
    box_height = int(image_height * 0.5)
    left = int((image_width - box_width) / 2)
    top = int((image_height - box_height) / 2)
    right = left + box_width
    bottom = top + box_height
    
    cv.rectangle(image, (left, top), (right, bottom), (255,0, 0), 2)
    # lastrequesttime=time.time()
    # Movement chcks
    if (w>250 and h>350):
        cv.rectangle(image1, (left, top), (right, bottom), (255,0, 0), 2)
        # cv.putText(image1, 'Move Back', (left, top - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255,0, 0), 2)
    elif(w<100 and h<150):
        
        cv.rectangle(image1, (left, top), (right, bottom), (255,0, 0), 2)
        # cv.putText(image1, 'Come closer', (left, top - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255,0, 0), 2)



    info_text = handedness.classification[0].label[0:]
    # print(info_text)
    global lastrequesttime
    if(inp==1):
        # if(info_text=='Right'):
        #     client1_socket.send(b'b')
        # if(info_text=='Left'):
        #     client1_socket.send(b'a')
        
        if x>=150 and x<=340 and y>=80 and y<=260:
            cv.rectangle(image1, (left, top), (right, bottom), (255,0, 0), 2)
            cv.putText(image1, 'No Movement', (left, top - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255,0, 0), 2)
            
            if (time.time()-lastrequesttime)>1.5:
                if(info_text=='Right'):
                    rover_socket.send(b'rstop')
                elif(info_text=='Left'):
                    # rover_socket.send(b'astop')
                    if(sig=='Open' or sig=='OK'):
                        rover_socket.send(b'open')
                    elif(sig=='Close'):
                        rover_socket.send(b'close')
                lastrequesttime=time.time()
            
            # detect_gesture(5)
        
        elif x>=150 and x<=340 and y<80:
            cv.rectangle(image1, (left, top), (right, bottom), (0, 255, 0), 2)
            cv.putText(image1, 'Moving Forward', (left, top - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0,255, 0), 2)
            # global lastrequesttime
            if (time.time()-lastrequesttime)>1.5:
                if(info_text=='Right'):
                    rover_socket.send(b'rforward')
                elif(info_text=='Left'):
                    rover_socket.send(b'aforward')
                lastrequesttime=time.time()

        elif x>=150 and x<=340 and y>260:
            cv.rectangle(image1, (left, top), (right, bottom), (0, 255, 0), 2)
            cv.putText(image1, 'Moving Backward', (left, top - 10),cv.FONT_HERSHEY_SIMPLEX, 0.9, (0,255, 0), 2)
            if (time.time()-lastrequesttime)>1.5:
                if(info_text=='Right'):
                    rover_socket.send(b'rbackward')
                elif(info_text=='Left'):
                    rover_socket.send(b'abackward')
                lastrequesttime=time.time()
            
            # detect_gesture(4)
        elif x<150 and y>=80 and y<=260 :
            cv.rectangle(image1, (left, top), (right, bottom), (0, 255, 0), 2)
            cv.putText(image1, 'Moving Left', (left, top - 10),cv.FONT_HERSHEY_SIMPLEX, 0.9, (0,255, 0), 2)
            if (time.time()-lastrequesttime)>1.5:
                if(info_text=='Right'):
                    rover_socket.send(b'rleft')
                elif(info_text=='Left'):
                    rover_socket.send(b'aleft')
                lastrequesttime=time.time()
            
            # detect_gesture(3)
        elif x>340 and y>=80 and y<=260:
            cv.rectangle(image1, (left, top), (right, bottom), (0, 255, 0), 2)
            cv.putText(image1, 'Moving Right', (left, top - 10),cv.FONT_HERSHEY_SIMPLEX, 0.9, (0,255, 0), 2)
            if (time.time()-lastrequesttime)>1.5:
                if(info_text=='Right'):
                    rover_socket.send(b'rright')
                elif(info_text=='Left'):
                    rover_socket.send(b'aright')
                lastrequesttime=time.time()
                # detect_gesture(2)
        print("Hi")
        return [x, y, x + w, y + h]
    elif(inp==2):
        # if(info_text=='Right'):
        #     client1_socket_socket.send(b'c')
        # if(info_text=='Left'):
        #     client1_socket_socket.send(b'd')
        if x>=150 and x<=340 and y>=80 and y<=260:
           
            cv.rectangle(image1, (left, top), (right, bottom), (255,0, 0), 2)
            cv.putText(image1, 'No Movement', (left, top - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255,0, 0), 2)
            # global lastrequesttime
            if (time.time()-lastrequesttime)>1.5:
                if(info_text=='Right'):
                    # rover_socket.send(b'astop')
                    if(sig=='Open' or sig=='OK'):
                        rover_socket.send(b'open')
                    elif(sig=='Close'):
                        rover_socket.send(b'close')
                elif(info_text=='Left'):
                    rover_socket.send(b'rstop')
                lastrequesttime=time.time()
            
            # detect_gesture(5)

        elif x>=150 and x<=340 and y<80:
            
            cv.rectangle(image1, (left, top), (right, bottom), (0, 255, 0), 2)
            cv.putText(image1, 'Moving Forward', (left, top - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0,255, 0), 2)
            # global lastrequesttime
            if (time.time()-lastrequesttime)>1.5:
                if(info_text=='Right'):
                    rover_socket.send(b'aforward')
                elif(info_text=='Left'):
                    rover_socket.send(b'rforward')
                lastrequesttime=time.time()


            # time.sleep(2)
            # detect_gesture(1)
        elif x>=150 and x<=340 and y>260:
           
            cv.rectangle(image1, (left, top), (right, bottom), (0, 255, 0), 2)
            cv.putText(image1, 'Moving Backward', (left, top - 10),cv.FONT_HERSHEY_SIMPLEX, 0.9, (0,255, 0), 2)
            if (time.time()-lastrequesttime)>1.5:
                if(info_text=='Right'):
                    rover_socket.send(b'abackward')
                elif(info_text=='Left'):
                    rover_socket.send(b'rbackward')
                lastrequesttime=time.time()
            
            # detect_gesture(4)
        elif x<150 and y>=80 and y<=260 :
            
            cv.rectangle(image1, (left, top), (right, bottom), (0, 255, 0), 2)
            cv.putText(image1, 'Moving Left', (left, top - 10),cv.FONT_HERSHEY_SIMPLEX, 0.9, (0,255, 0), 2)
            if (time.time()-lastrequesttime)>1.5:
                if(info_text=='Right'):
                    rover_socket.send(b'aleft')
                elif(info_text=='Left'):
                    rover_socket.send(b'rleft')
                lastrequesttime=time.time()
            
            # detect_gesture(3)
        elif x>340 and y>=80 and y<=260:
             
            cv.rectangle(image1, (left, top), (right, bottom), (0, 255, 0), 2)
            cv.putText(image1, 'Moving Right', (left, top - 10),cv.FONT_HERSHEY_SIMPLEX, 0.9, (0,255, 0), 2)
            if (time.time()-lastrequesttime)>1.5:
                if(info_text=='Right'):
                    rover_socket.send(b'aright')
                elif(info_text=='Left'):
                    rover_socket.send(b'rright')
                lastrequesttime=time.time()
            # detect_gesture(2)
        if(sig=='Open' or sig=='OK'):
            if (time.time()-lastrequesttime)>1.5:
                if(info_text=='Right'):
                    rover_socket.send(b'open')
        elif(sig=='Close'):
            if (time.time()-lastrequesttime)>1.5:
                if(info_text=='Right'):
                    rover_socket.send(b'close')
        return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # 手首1
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # 手首2
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # 親指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # 親指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # 親指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # 人差指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # 人差指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # 人差指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # 人差指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # 中指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # 中指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # 中指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # 中指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # 薬指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # 薬指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # 薬指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # 薬指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # 小指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # 小指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # 小指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # 小指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image1,image ,brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text +':' + str(brect)
        # print(info_text)
        print(hand_sign_text)
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


if _name_ == '_main_':
    main()
