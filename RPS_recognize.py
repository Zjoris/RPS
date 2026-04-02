import cv2
import numpy as np 
# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision

import argparse


def main(server_url):
    cap = cv2.VideoCapture(server_url)

    if not cap.isOpened():
        print(f"Failed to connect to stream at {server_url}")
        return

    t = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))
        frame = cv2.flip(frame, 1)
            
        cv2.imshow("Webcam Stream", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        t += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default=0,
                        help='URL of the webcam stream server')
    args = parser.parse_args()

    main(args.url + "/video" )