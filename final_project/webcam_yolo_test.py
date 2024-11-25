import cv2 as cv
import argparse
import supervision as sv

from ultralytics import YOLO

def parse_arguments() -> argparse.Namespace:
   parser = argparse.ArgumentParser(description="YOLOv8 live")
   parser.add_argument("--webcam-resolution",
                       default=[1280, 720],
                       nargs=2,
                       type=int
   )
   args = parser.parse_args()
   return args

def main():
   args = parse_arguments()
   frame_width, frame_height = args.webcam_resolution

   cap = cv.VideoCapture(0)
   cap.set(cv.CAP_PROP_FRAME_WIDTH, frame_width)
   cap.set(cv.CAP_PROP_FRAME_HEIGHT, frame_height)

   model = YOLO("yolov8l.pt")

   bounding_box_annotator = sv.BoundingBoxAnnotator()

   while True: 
      ret, frame = cap.read()
      cv.imshow("yolov8", frame)

      result = model(frame)[0]
      detections = sv.Detections.from_ultralytics(result)

      frame = bounding_box_annotator.annotate(scene=frame, detections=detections)

      if cv.waitKey(30) == 27:
         break

if __name__ == "__main__":
   main()