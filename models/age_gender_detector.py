import torch
import cv2
from facenet_pytorch import MTCNN
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import torchvision
from torch import nn
import time

device = "cuda:0" if torch.cuda.is_available() else "cpu"
################################# loading models #################################
start = time.time()
print("Loading models into memory and gpu....")
face_detector = MTCNN(image_size=112, keep_all=True, device=device)
gender_model = torch.load(
    "./checkpoints/bigger_face_checkpoint_all.pt", 
    map_location="cpu"
    )["model"]
gender_model["backbone"].to(device).eval()
gender_model["head"]["gender"].to(device).eval()

age_model = torch.load(
    "./checkpoints/vgg_checkpoint.pt", 
    map_location="cpu"
    )["model"]
age_model["backbone"].to(device).eval()
age_model["head"]["age"].to(device).eval()
print("Model loaded in {:.4f} seconds".format(time.time() - start))
############################# pre-process transforms #############################
trans = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])


def detect_face(
        frame: np.ndarray, 
        ratio: float=1.0,
        face_detector: nn.Module=face_detector,
        ):
    bboxs, _, landmarks = face_detector.detect(frame, landmarks = True)
    faces = []
    if bboxs is not None: 
        for bbox in bboxs:
            bbox = bbox.reshape(2, 2)
            cbox = np.mean(bbox, axis = 0)
            # a bigger bounding box for face detection
            width = np.max(bbox[1, :] - bbox[0, :]) * ratio
            # determine whether the face is too small
            if width < 60: continue
            crop = np.vstack([cbox - width / 2., cbox + width / 2])
    
            crop[:, 0] = crop[:, 0].clip(0, frame.shape[1])
            crop[:, 1] = crop[:, 1].clip(0, frame.shape[0])
            # crop face to perform prediction
            ux, uy, lx, ly = crop.astype(np.int).reshape(-1)
            face = frame[uy: ly, ux: lx]
            face = cv2.resize(face, (112, 112), interpolation=cv2.INTER_NEAREST)
            faces.append((cbox, face))
            # the maximum face detected should be 2
            if len(faces) == 2: break
    return sorted(faces, key=lambda x: x[0][0] if isinstance(x, tuple) else 1)


def predict_gender(
        faces: list, 
        gender_model: OrderedDict=gender_model, 
        device: str=device,
        trans: nn.Module=trans
        ):
    backbone = gender_model["backbone"]
    cls_head = gender_model["head"]["gender"]
    # make predictions
    gen_pred = cls_head(backbone(faces))
    _, gender = torch.max(gen_pred, dim=1)
    gender = gender + 1
    return gender.tolist()


def predict_age(
        faces: list, 
        age_model: OrderedDict=age_model, 
        device: str=device,
        trans: nn.Module=trans
        ):
    backbone = age_model["backbone"]
    reg_head = age_model["head"]["age"]
    # make predictions
    age_pred = torch.sigmoid(reg_head(backbone(faces))).squeeze(1)
    age = age_pred * 35 + 15
    return age.tolist()


def age_gender_detection_kernel(frame: np.ndarray):
    genders, ages = [], []
    #########################################################
    ## predict gender
    faces = detect_face(frame, ratio=1.5)
    for idx, (center, face) in enumerate(faces):
        # reshape and preprocess the data for gender classification
        face = trans(face).unsqueeze(0).to(device)
        gender = predict_gender(face)
        genders += gender
        if len(genders) >= 2: break
    if len(genders) == 1: genders = genders + genders
    #########################################################
    ## predict age
    
    faces = detect_face(frame, ratio=1.0)
    for idx, (center, face) in enumerate(faces):
        # reshape and preprocess the data for gender classification
        face = trans(face).unsqueeze(0).to(device)
        age = predict_age(face)
        
        ages += age
        if len(ages) >= 2: break
    if len(ages) == 1: ages = ages + ages
    return genders, ages


def run_age_gender_detector(cap: cv2.VideoCapture):
    start = time.time()
    genders, ages = [], []
    while True:
        _, frame = cap.read()
        frame = frame[:, :, ::-1]

        gender, age = age_gender_detection_kernel(frame)
        genders += gender
        ages += age
        if time.time() - start > 5: break
    genders = np.median(np.array(genders).reshape(-1, 2), axis=0)
    ages = (np.mean(np.array(ages).reshape(-1, 2), axis=0) > 32) + 1
    return genders.astype(np.int32), ages.astype(np.int32)

"""
videofile = "../../watch-edited/subject-4.mp4"
cap = cv2.VideoCapture(videofile)
faces = []
while cap.isOpened():
    ret, frame = cap.read()
    frame = frame[:,:,::-1]
    print(age_gender_detection_kernel(frame))
    break
"""