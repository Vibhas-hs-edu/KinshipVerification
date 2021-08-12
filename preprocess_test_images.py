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

def preprocess_train_images(train_dir :Path, output_dir : Path):
    create_path_if_not_exists(output_dir)
    #This script expects the folder to have the following structure
    #FamilyFolder -> Persons Folder -> Images of a person
    for family_path in tqdm(train_dir.iterdir()):
        for person_path in family_path.iterdir():
            if not person_path.is_dir():
                continue
            output_path = create_path_if_not_exists(output_dir / person_path.relative_to(train_dir))
            print(output_path)
            for img_path in person_path.iterdir():
                output_path_img = output_path / img_path.name
                print('output_path_img', output_path_img)
                if os.path.exists(output_path_img):
                    print(f'{output_path_img} already exists. Ignoring!')
                    continue
                img = cv2.imread(str(img_path))
                faces = RetinaFace.detect_faces(img_path = str(img_path), threshold = 0.5)
                landmarks = get_landmarks(faces)
                if len(landmarks) == 0:
                    print(f'smth wrong with {img_path}')
                    warped_img = cv2.resize(img, (112, 112))
                else:
                    print(f'in norm_crop')
                    warped_img = norm_crop(img, landmarks)
                cv2.imwrite(str(output_path_img), warped_img)

if __name__ == '__main__':
    print('Hello world')
    #assert len(sys.argv) == 3
    train_dir = sys.argv[1]
    output_dir = sys.argv[2]
    preprocess_train_images(Path(train_dir), Path(output_dir))


