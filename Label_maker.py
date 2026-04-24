import os
import csv
import subprocess

''' 
File to download RPS image dataset and create a csv file containing 
(PATH_TO_IMAGE, IMAGE_LABEL) pairs for torch dataset. 
'''
def main(): 
    if not os.path.isdir("rps_data_sample") :
        # get dataset
        subprocess.run(["wget", "https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/rps_data_sample.zip"])
        subprocess.run(["unzip", "rps_data_sample.zip"])
        subprocess.run(["rm", "rps_data_sample.zip"])

        # create labels csv: for each image write "PATH, label"
        with open("labels.csv", "w") as labels_csv :
            writer = csv.writer(labels_csv)
            for i, d in enumerate(os.listdir("rps_data_sample")) :
                for f in os.listdir(f"rps_data_sample/{d}") :
                    writer.writerow([f"{d}/{f}", i])

if __name__ == "__main__" :
    main()