import cv2
import numpy as np 
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import argparse

''' 
Function to print what is being recognized
 Input
   res    : GestureRecognizerResult
   out_im : mediapipe image
   t      : timestep as Int
 '''
def prnt_res(res : vision.GestureRecognizerResult, out_im : mp.Image, t : int ) :
    if not t%10 :
        print(f'Recognizing: {res.handedness} , {t}')


def main(server_url):
    base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
    options = vision.GestureRecognizerOptions(base_options=base_options,
                                              running_mode = vision.RunningMode.LIVE_STREAM,
                                              result_callback = prnt_res)
    
    with vision.GestureRecognizer.create_from_options(options) as recog :
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

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data= frame)
            recog.recognize_async(mp_image, t)
             
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

    main(args.url)