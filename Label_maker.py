import os
import csv
import shutil
import subprocess

def main(): 
    if not os.path.isdir("rps_data_sample") :
        #wget https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/rps_data_sample.zip
        subprocess.run(["wget", "https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/rps_data_sample.zip"])
        #unzip rps_data_sample.zip
        subprocess.run(["unzip", "rps_data_sample.zip"])
        #rm rps_data_sample.zip
        subprocess.run(["rm", "rps_data_sample.zip"])
        # File restructure
        # os.makedirs("rps_data")
        with open("labels.csv", "w") as labels_csv :
            writer = csv.writer(labels_csv)
            for i, d in enumerate(os.listdir("rps_data_sample")) :
                for f in os.listdir(f"rps_data_sample/{d}") :
                    writer.writerow([f"{d}/{f}", i])

if __name__ == "__main__" :
    main()