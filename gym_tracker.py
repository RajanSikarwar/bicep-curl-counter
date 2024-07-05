import cv2
import mediapipe as md
import numpy as np

md_drawing = md.solutions.drawing_utils
md_pose = md.solutions.pose

# curl counter variables
count = 0
stage = None

# video feed
cap = cv2.VideoCapture(0)

# mediapipe instance
with md_pose.Pose(min_detection_confidence = 0.7, min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("empty camera")
            break

        # recolor image to RGB for mediapipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # make detection
        results = pose.process(image)

        # back to BGR for cv2
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        def calculate_angle(a,b,c):
            a = np.array(a) #First point
            b = np.array(b) #Mid point
            c = np.array(c) #End point

            radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(radians*180.0/np.pi)

            if angle >180.0:
                angle = 360 - angle

            return angle
        
        # extracting landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # get coordinated
            shoulder= [landmarks[md_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[md_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[md_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[md_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[md_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[md_pose.PoseLandmark.LEFT_WRIST.value].y]

            # calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)
            h,w,_ = image.shape

            # visualize angle
            cv2.putText(image, str(angle),
                        tuple(np.multiply(elbow, [w, h]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )
            
            # curl counter logic
            if angle > 160:
                stage = "down"
            if angle < 30 and stage == "down":
                stage = "up"
                count += 1
                #print(count)

        except:
            pass
        
        # render curl counter
        # setup status box
        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)# start point, end point, color, line thickness

        # rep data
        cv2.putText(image, 'REPS', (15,12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(count), (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        
        #stage data
        cv2.putText(image, 'STAGE', (65,12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, stage , (60,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        md_drawing.draw_landmarks(image, results.pose_landmarks, md_pose.POSE_CONNECTIONS,
                                  md_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), #joint color
                                  md_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2), #connection color
                                  )
        
                    
        cv2.imshow("Mediapipe feed", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()