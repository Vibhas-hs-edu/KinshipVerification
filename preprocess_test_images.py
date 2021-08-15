#create a new virtual environment and install following dependencies
#activate the virtual environment
#!pip install retina-face opencv-pythonD scikit-image

from retinaface import RetinaFace
from pathlib import Path
import os
from tqdm import tqdm
import cv2
import numpy as np
from skimage import transform as trans
from norm_crop import norm_crop
import sys

def create_path_if_not_exists(cur: Path) -> Path:
    if not cur.exists():
        os.makedirs(str(cur))
    return cur

def get_landmarks(face):
    if 'face_1' not in face:
        return []
    if len(face) == 0:
        return []
    tmp = []
    for item in face['face_1']['landmarks']:
        tmp.append(face['face_1']['landmarks'][item])
    a = np.array(tmp)
    return a

def preprocess_test_images(root_dir: Path, output_dir: Path):
  output_dir = create_path_if_not_exists(output_dir)
  for img_path in root_dir.iterdir():
    img = cv2.imread(str(img_path))
    resp = RetinaFace.detect_faces(img_path = str(img_path))
    landmarks = get_landmarks(resp)
    output_path = output_dir / img_path.name
    if not landmarks.any():
      print(f'smth wrong with {img_path}')
      warped_img = cv2.resize(img, (112, 112))
    else:
      warped_img = norm_crop(img, landmarks)
    cv2.imwrite(str(output_path), warped_img)

if __name__ == '__main__':
    print('Hello world')
    #assert len(sys.argv) == 3
    test_dir = sys.argv[1]
    output_dir = sys.argv[2]
    preprocess_test_images(Path(test_dir), Path(output_dir))


