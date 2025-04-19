import cv2
import dlib
import numpy as np
import joblib
import pygame

# Load the trained model
model = joblib.load("drowsiness_svm_model.pkl")

# Load shape predictor and face detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()

# Initialize pygame mixer
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("C:/Users/hp/Desktop/Drowsiness using svm/alert.wav")

# Function to calculate EAR
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# Start webcam
cap = cv2.VideoCapture(0)

drowsy_event = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])

        ear_left = eye_aspect_ratio(left_eye)
        ear_right = eye_aspect_ratio(right_eye)
        ear = (ear_left + ear_right) / 2.0

        prediction = model.predict([[ear]])[0]

        if prediction == 1:  # Drowsy
            label = "Drowsy"
            color = (0, 0, 255)

            # Play sound only once per drowsy event
            if not drowsy_event and alert_sound.get_num_channels() == 0:
                alert_sound.play()
                drowsy_event = True
        else:  # Awake
            label = "Awake"
            color = (0, 255, 0)
            drowsy_event = False  # Reset flag

        cv2.putText(frame, f"{label}", (face.left(), face.top() - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), color, 2)

    cv2.imshow("Real-Time Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#this code is best if pc takes the laod
# import cv2
# import dlib
# import numpy as np
# import joblib
# import simpleaudio as sa

# # Load the trained model
# model = joblib.load("drowsiness_svm_model.pkl")

# # Load shape predictor
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# detector = dlib.get_frontal_face_detector()

# # Load the alert sound
# wave_obj = sa.WaveObject.from_wave_file("C:/Users/hp/Desktop/Drowsiness using svm/alert.wav")
# play_obj = None
# drowsy_alert_active = False

# # Function to calculate EAR
# def eye_aspect_ratio(eye):
#     A = np.linalg.norm(eye[1] - eye[5])
#     B = np.linalg.norm(eye[2] - eye[4])
#     C = np.linalg.norm(eye[0] - eye[3])
#     return (A + B) / (2.0 * C)

# # Start webcam
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = cv2.flip(frame, 1)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = detector(gray)

#     for face in faces:
#         landmarks = predictor(gray, face)
#         left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
#         right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])

#         ear_left = eye_aspect_ratio(left_eye)
#         ear_right = eye_aspect_ratio(right_eye)
#         ear = (ear_left + ear_right) / 2.0

#         prediction = model.predict([[ear]])[0]

#         if prediction == 1:  # Drowsy
#             label = "Drowsy"
#             color = (0, 0, 255)

#             if not drowsy_alert_active:
#                 play_obj = wave_obj.play()
#                 drowsy_alert_active = True
#         else:  # Awake
#             label = "Awake"
#             color = (0, 255, 0)

#             if drowsy_alert_active and play_obj is not None:
#                 play_obj.stop()
#                 drowsy_alert_active = False

#         cv2.putText(frame, f"{label}", (face.left(), face.top() - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
#         cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), color, 2)

#     cv2.imshow("Real-Time Drowsiness Detection", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()