try:
    import cv2
    import mediapipe
    import numpy
    print("Imports successful")
except ImportError as e:
    print(f"Import failed: {e}")
