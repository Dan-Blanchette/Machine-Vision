import cv2
import platform


def main():
   print("Python version: " + platform.python_version())

   print("opencv version: " + cv2.__version__)

if __name__=="__main__":
   main()