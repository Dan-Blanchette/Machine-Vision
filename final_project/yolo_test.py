# Author: Dan Blanchette
# Date: November 2024
# Project Description: Machine Vision Final Project YOLOv8 to detect PPE 
# via motion capture livestream.

'''
Sources and Datasets:
https://universe.roboflow.com/roboflow-100/construction-safety-gsnvb/dataset/2#
'''
'''
Tutorials and References:
https://medium.com/@jaykumaran2217/personal-protective-equipment-tracking-in-construction-site-using-yolov8-f4775a4f1fe3

'''

# IMPORT THE DATASET FROM ROBOFLOW

# from roboflow import Roboflow
# rf = Roboflow(api_key="R5Z4WKO5hT8sQJGj9Ht0")
# project = rf.workspace("roboflow-100").project("construction-safety-gsnvb")
# version = project.version(2)
# dataset = version.download("yolov8")

# Create a and train a YOLO model
from ultralytics import YOLO
model = YOLO('yolov8n.pt')

model.info()
# # train using .yaml file format to roll through the construction data set
# # NOTE: be sure to update file paths in .yaml to the the directory for the construction data
# # for test, train, and validation sets.
results  = model.train(data='/home/tarnished-dan22/Machine-Vision/final_project/construction-safety-2/data.yaml', 
                       epochs=10, imgsz=640)
# tese the model by providing it "unseen" data to evaluate
results = model('test_imgs/',  save=True)
