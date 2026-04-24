import cv2
import numpy as np 
import torch

import argparse

from RPS_Dataset_Models import RPS_Classifier_torch


def main(server_url):
    cap = cv2.VideoCapture(server_url)

    if not cap.isOpened():
        print(f"Failed to connect to stream at {server_url}")
        return
    
    model = RPS_Classifier_torch(4, dims = (600, 600))
    model.load_state_dict(torch.load("model.pt",
                                    weights_only=True,
                                    map_location=torch.device("cpu")))
    model.eval()

    print("model loaded")

    t = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (600, 600))
        frame = cv2.flip(frame, 1)
            
        cv2.imshow("Webcam Stream", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        # Ubuntu crashes when trying to use this model. Which makes sense.
        # CHeck : https://www.slingacademy.com/article/efficient-pytorch-inference-for-real-time-neural-network-classification/ 
        # with torch.no_grad() :
        #     outputs = model(frame[None, :, :, :])
        #     print(torch.max(outputs, 1))
        t += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default="0",
                        help='URL of the webcam stream server')
    args = parser.parse_args()

    main(args.url + "/video" )