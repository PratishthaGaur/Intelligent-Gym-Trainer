from flask import Flask, render_template, Response, request,jsonify
import cv2
import numpy as np
import mediapipe as mp

app = Flask(__name__)
# app.config['ENV'] = 'development'
# app.config['DEBUG'] = True
# app.config['TESTING'] = True
ClassIDs = []


def gen_frames():
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            net = cv2.dnn.readNet("yolov4-obj_last4000.weights", "yolov4-obj.cfg")

            classes = []
            with open("obj.names.txt", "r") as f:
                read = f.readlines()
            for i in range(len(read)):
                classes.append(read[i].strip("\n"))

            layer_names = net.getLayerNames()
            output_layers = []
            for i in net.getUnconnectedOutLayers():
                output_layers.append(layer_names[i - 1])

            height, width, channels = frame.shape

            blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (416, 416), (0, 0, 0), True, crop=False)

            net.setInput(blob)
            outs = net.forward(output_layers)
            class_ids = []
            thresholds = []
            boxes = []
            for output in outs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    threshold = scores[class_id]

                    if threshold > 0.7:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        thresholds.append(float(threshold))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, thresholds, 0.3, 0.4)

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = classes[class_ids[i]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (185, 185, 0), 2)
                    cv2.rectangle(frame, (x - 1, y), (x + w + 1, y - 25), (185, 185, 0), -1)
                    cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 185), 2)
            for output in outs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    threshold = scores[class_id]
                    if threshold > 0.7:
                        thresholds.append(float(threshold))
                        class_ids.append(class_id)
                        ClassIDs.append(class_ids)
                        print(class_id)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def plank_result():

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    # Curl counter variables
    counter = 0
    stage = None
    stage2 = None
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ## Setup mediapipe instance
            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

                    # Recolor image to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame.flags.writeable = False

                    # Make detection
                    results = pose.process(frame)

                    # Recolor back to BGR
                    frame.flags.writeable = True
                    image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    # Extract landmarks
                    try:
                        landmarks = results.pose_landmarks.landmark

                        # Get coordinates
                        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

                        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                        # Calculate angle
                        back_angle = calculate_angle(shoulder, hip, knee)
                        elbow_angle = calculate_angle(shoulder, elbow, wrist)

                        # Visualize angle
                        cv2.putText(image, str(back_angle),
                                    tuple(np.multiply(hip, [640, 480]).astype(int)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                        cv2.putText(image, str(elbow_angle),
                                    tuple(np.multiply(elbow, [640, 480]).astype(int)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )

                        # Curl counter logic
                        if back_angle < 160:
                            stage = "move your back down"
                        elif back_angle > 200:
                            stage = "move your back up"
                            counter += 1
                            # print(counter)
                        else:
                            stage = 'Perfect'

                        if elbow_angle > 85 and elbow_angle < 105:
                            stage2 = 'Perfect'
                        else:
                            stage2 = "elbow should be at 90"


                    except:
                        pass

                    # Render curl counter
                    # Setup status box
                    # cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)

                    # Rep data
                    # cv2.putText(image, 'REPS', (15, 12),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    # cv2.putText(image, str(counter),(10, 60),cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                    # Stage data
                    cv2.putText(image, 'FEEDBACK FOR BACK ARC', (65, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                                cv2.LINE_AA)
                    cv2.putText(image, stage, (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

                    cv2.putText(image, 'FEEDBACK FOR ELBOW', (65, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, stage2, (60, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

                    # Render detections
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                              )
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def rowing_result():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    counter = 0
    stage = None
    camera = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:

            # Setup mediapipe instance
            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                # Recolor image to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame.flags.writeable = False

                # Make detection
                results = pose.process(frame)

                # Recolor back to BGR
                frame.flags.writeable = True
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark

                    # Get coordinates
                    lankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    lknee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    lhip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    lshoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    lelbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    lwrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                    rankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    rknee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    rhip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    rshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    relbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    rwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                    # Calculate angle
                    lkangle = calculate_angle(lankle, lknee, lhip)
                    rkangle = calculate_angle(rankle, rknee, rhip)
                    lhangle = calculate_angle(lknee, lhip, lshoulder)
                    rhangle = calculate_angle(rknee, rhip, rshoulder)
                    leangle = calculate_angle(lwrist, lelbow, lshoulder)
                    reangle = calculate_angle(rwrist, relbow, rshoulder)

                    # Visualize angle
                    cv2.putText(frame, str(leangle),
                                tuple(np.multiply(lelbow, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )

                    cv2.putText(frame, str(reangle),
                                tuple(np.multiply(relbow, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )  # Visualize angle
                    cv2.putText(frame, str(lhangle),
                                tuple(np.multiply(lelbow, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )

                    cv2.putText(frame, str(rhangle),
                                tuple(np.multiply(relbow, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )  # Visualize angle
                    cv2.putText(frame, str(lkangle),
                                tuple(np.multiply(lelbow, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )

                    cv2.putText(frame, str(rkangle),
                                tuple(np.multiply(relbow, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )

                    # Curl counter logic
                    if rkangle < 90 and rhangle < 45 and reangle > 160:
                        stage = "Compressed"
                    if rkangle > 160 and rhangle > 45 and reangle < 30 and stage == 'Compressed':
                        stage = "Expand"
                        counter += 1
                        print(counter)

                except:
                    pass

                cv2.rectangle(frame, (0, 0), (225, 75), (245, 117, 16), -1)

                # Counter data
                cv2.putText(frame, 'REPS', (15, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, str(counter),
                            (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                # Stage data

                cv2.putText(frame, stage,(60, 120),cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                # Render detections
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2,
                                                                 circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                          )

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def pushup_result():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    counter = 0
    stage = None
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                    # Recolor image to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame.flags.writeable = False

                    # Make detection
                    results = pose.process(frame)

                    # Recolor back to BGR
                    frame.flags.writeable = True
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    # Extract landmarks
                    try:
                        landmarks = results.pose_landmarks.landmark

                        # Get coordinates
                        lshoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        lelbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                        lwrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                        rshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                     landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                        relbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                        rwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                        # Calculate angle
                        langle = calculate_angle(lshoulder, lelbow, lwrist)
                        rangle = calculate_angle(rshoulder, relbow, rwrist)
                        # Visualize angle
                        cv2.putText(frame, str(langle),
                                    tuple(np.multiply(lelbow, [640, 480]).astype(int)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )

                        cv2.putText(frame, str(rangle),
                                    tuple(np.multiply(relbow, [640, 480]).astype(int)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                        # Curl counter logic
                        if langle < 110 and rangle < 110:
                            stage = 'Down'
                        if langle > 160 and rangle > 160 and stage == 'Down':
                            stage = 'Up'
                            counter += 1
                    except:
                        pass

                    cv2.rectangle(frame, (0, 0), (225, 150), (245, 117, 16), -1)

                    # Left Rep data
                    cv2.putText(frame, 'REPS', (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(frame, str(counter),
                                (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                    # Left Stage data

                    cv2.putText(frame, stage,
                                (60, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                    # Render detections
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                              )
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
def bicep_curl_result():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    lcounter = 0
    lstage = None
    rcounter = 0
    rstage = None
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:

            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame.flags.writeable = False

                    # Make detection
                    results = pose.process(frame)

                    # Recolor back to BGR
                    frame.flags.writeable = True
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    # Extract landmarks
                    try:
                        landmarks = results.pose_landmarks.landmark

                        # Get coordinates
                        lshoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        lelbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                        lwrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                        rshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                     landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                        relbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                        rwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                        # Calculate angle
                        langle = calculate_angle(lshoulder, lelbow, lwrist)
                        rangle = calculate_angle(rshoulder, relbow, rwrist)
                        # Visualize angle
                        cv2.putText(frame, str(langle),
                                    tuple(np.multiply(lelbow, [640, 480]).astype(int)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )

                        cv2.putText(frame, str(rangle),
                                    tuple(np.multiply(relbow, [640, 480]).astype(int)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                        # Curl counter logic
                        if langle > 160:
                            lstage = "down"
                        if langle < 30 and lstage == 'down':
                            lstage = "up"
                            lcounter += 1
                            print(lcounter)
                        if rangle > 160:
                            rstage = "down"
                        if rangle < 30 and rstage == 'down':
                            rstage = "up"
                            rcounter += 1
                            print(rcounter)


                    except:
                        pass

                    # Render curl counter
                    # Setup status box
                    cv2.rectangle(frame, (0, 0), (225, 150), (245, 117, 16), -1)

                    # Left Rep data
                    cv2.putText(frame, 'LEFT REPS', (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(frame, str(lcounter),
                                (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                    # Left Stage data

                    cv2.putText(frame, lstage,
                                (60, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                    # Right Rep data
                    cv2.putText(frame, 'RIGHT REPS', (15, 48),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(frame, str(rcounter),
                                (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                    # Right Stage data

                    cv2.putText(frame, rstage,
                                (60, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                    # Render detections
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                              )
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def squat_result():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    # Curl counter variables
    counter = 0
    stage = None
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
    ## Setup mediapipe instance
            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                    # Recolor image to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame.flags.writeable = False

                    # Make detection
                    results = pose.process(frame)

                    # Recolor back to BGR
                    frame.flags.writeable = True
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    # Extract landmarks
                    try:
                        landmarks = results.pose_landmarks.landmark

                        # Get coordinates
                        lhip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                        lknee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                        lankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                        rhip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                        rknee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                        rankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                        # Calculate angle
                        rkneeangle = calculate_angle(rhip, rknee, rankle)
                        lkneeangle = calculate_angle(lhip, lknee, lankle)

                        # Visualize angle
                        cv2.putText(frame, str(rkneeangle),
                                    tuple(np.multiply(rknee, [640, 480]).astype(int)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                        cv2.putText(frame, str(lkneeangle),
                                    tuple(np.multiply(lknee, [640, 480]).astype(int)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )

                        # Curl counter logic
                        if rkneeangle < 110 and lkneeangle < 110:
                            stage = "down"
                        if rkneeangle > 160 and lkneeangle > 160 and stage == 'down':
                            stage = "up"
                            counter += 1
                            print(counter)

                    except:
                        pass

                    # Render curl counter
                    # Setup status box
                    cv2.rectangle(frame, (0, 0), (225, 73), (245, 117, 16), -1)

                    # Rep data
                    cv2.putText(frame, 'REPS', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(frame, str(counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                    # Stage data
                    cv2.putText(frame, 'STAGE', (65, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(frame, stage, (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                    # Render detections
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                              )
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/ID', methods=['get'])
def ID():
    ID= str(ClassIDs[0][0])
    return render_template('exercise.html',ID=ID)
@ app.route('/video_feed_rowing')
def video_feed_rowing():
    return Response(rowing_result(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_squat')
def video_feed_squat():
    return Response(squat_result(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_pushup')
def video_feed_pushup():
    return Response(pushup_result(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_plank')
def video_feed_plank():
    return Response(plank_result(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_bicep_curl')
def video_feed_bicep_curl():
    return Response(bicep_curl_result(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/Superman')
def Superman():
    return render_template('Rowing.html')
@app.route('/Crunches')
def Crunches():
    return render_template('Rowing.html')
@app.route('/Jumping_jack')
def Jumping_jack():
    return render_template('Rowing.html')
@app.route('/Bridge')
def Bridge():
    return render_template('Rowing.html')
@app.route('/Side_plank')
def Side_plank():
    return render_template('Rowing.html')
@app.route('/Burpees')
def Burpees():
    return render_template('Rowing.html')
@app.route('/Lunges')
def Lunges():
    return render_template('Rowing.html')
@app.route('/Rowing')
def Rowing():
    return render_template('Rowing.html')


@app.route('/Pushup')
def Pushup():
    return render_template('pushup.html')


@app.route('/Plank')
def Plank():
    return render_template('plank.html')

@app.route('/Bicep_curl')
def Bicep_curl():
    return render_template('bicep_curl.html')
@app.route('/Squat')
def Squat():
    return render_template('squat.html')


if __name__ == "__main__":
    app.run(debug=True)
