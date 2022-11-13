from facenet_pytorch import MTCNN
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

start = time.time()
print("Initailizing eye gaze model and facial expression model...")
from .eye_gaze_and_facial_expression_model import FacialExpression, EyeGaze

EYE_RATIO = 0.3
device = "cuda:0" if torch.cuda.is_available() else "cpu"

FacialExpression.to(device).eval()
EyeGaze.to(device).eval()
print("Model loaded in {:.4f} seconds".format(time.time() - start))

Transform = torchvision.transforms.Compose([
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                 std=[0.229, 0.224, 0.225])
                ])
# pretrained face detector
start = time.time()
print("Initailizing face detector...")
mtcnn = MTCNN(device)
mtcnn.detect([np.random.randn(112, 112, 3)])
print("Model loaded in {:.4f} seconds".format(time.time() - start))


def UnsqueezeToDevice(*args):
    return tuple(Transform(x).unsqueeze(0).to(device) for x in args)


# Helper function for eye gaze input extraction
def ResizeToTensor(x: np.ndarray, size: tuple):
    x = cv2.resize(x, size, cv2.INTER_CUBIC)
    x = x.astype(np.float32) / 255.
    x = torch.from_numpy(x).permute(2, 0, 1)
    return x


def MakeMask(image_size: tuple, grid: tuple, ymin: int, xmin: int, ymax: int, xmax: int):
    (h, w, _), (m, n) = image_size, grid
    ch, cw = h / 2, w / 2.
    rate = 1.15
    xmin, xmax = map(lambda x: (ch - (ch - x)) * rate, [xmin, xmax])
    ymin, ymax = map(lambda x: (cw - (cw - x)) * rate, [ymin, ymax])
    mask = np.zeros( (m, n), dtype=np.float32 )
    xmin, xmax = xmin/h*m, xmax/h*m
    ymin, ymax = ymin/w*n, ymax/w*n
    mask[int(xmin-0.5):int(xmax+0.5), int(ymin-0.5):int(ymax+0.5)] = 1.
    return torch.from_numpy(mask).view(-1)


# Main function for face detection
def DetectFace(image: np.ndarray, grid: tuple=(36, 64)):
    # Crop Image for faster inference
    shiftx, shifty = 550, 400
    # shiftx, shifty = 0, 0
    image_crop = image[shiftx:850, shifty:1200, :]
    # plt.imshow(image_crop); plt.show()
    bbox, _, lmks = mtcnn.detect(image_crop, landmarks=True)
    # Return a string if no valid facehave been detected
    if bbox is not None: 
        faces = []
        for b, lm in zip(bbox, lmks):
            ymin, xmin, ymax, xmax = b.astype(np.int32)
            if min(xmax - xmin, ymax - ymin) < 60: continue
            ymin, xmin, ymax, xmax = ymin + shifty, xmin + shiftx, ymax + shifty, xmax + shiftx
            lm = lm + np.array([shifty, shiftx])
            landmark = lm
            ##################### Face Rotation Check ##########################
            lm = lm - lm[2, :]
            eye = lm[1, :] - lm[0, :]
    
            phi = np.arctan(eye[1]/eye[0])
            rotate = np.array([[ np.cos(phi), np.sin(phi),           0.],
                               [-np.sin(phi), np.cos(phi),           0.],
                               [           0,           0,           1.]])
            lm = rotate @ np.hstack([lm, np.ones((5, 1))]).T
    
            l, r = lm[0, 0], lm[0, 1]
            NoseMouth = np.mean(lm[1, 3:5], axis=0)
            MouthLength = np.linalg.norm(lm[:, 3] - lm[:, 4])
    
            leftright = min(abs(l/r), abs(r/l))
            updown = NoseMouth / MouthLength
            ###################### Crop face from the frame #####################
            face = image[xmin:xmax, ymin:ymax, ...]
    
            height, width = xmax - xmin, ymax - ymin
            if height > width:
                pad = (height - width) // 2
                face = np.pad(face, ((0, 0), (pad, height-width-pad), (0, 0)), 
                              constant_values=0)
            if height < width:
                pad = (width - height) // 2
                face = np.pad(face, ((pad, width-height-pad), (0, 0), (0, 0)), 
                              constant_values=0)
            face = cv2.resize(face, (112, 112), cv2.INTER_CUBIC)
            #################### Crop eyes from the frame #######################
            (lefty, leftx), (righty, rightx) = landmark[0:2].astype(np.int32)
            bias = int((EYE_RATIO*0.5)*(xmax-xmin))
            left_eye = image[ leftx-bias: leftx+bias,  lefty-bias: lefty+bias, ...]
            righteye = image[rightx-bias:rightx+bias, righty-bias:righty+bias, ...]
    
            left_eye = ResizeToTensor(left_eye, (32, 32))
            righteye = ResizeToTensor(righteye, (32, 32))
            face     = ResizeToTensor(face    , (112, 112))
            ################## Absolute location of the face ####################
            mask     = MakeMask(image.shape, grid, ymin, xmin, ymax, xmax)
            
            left_eye, righteye, face = UnsqueezeToDevice(left_eye, righteye, face)
            mask = mask.unsqueeze(0).to(device)

            center = ((ymin + ymax)/2, (xmin+xmax)/2.)
            faces.append(
                (center, left_eye, righteye, face, mask, leftright, updown)
                )
        return faces
    return "No valid face have been detected!"


def detect(frame: np.ndarray):
    output = DetectFace(frame)
    results = []
    # If face is detected
    if isinstance(output, list):
        for center, left_eye, righteye, face, mask, leftright, updown in output:
            # Visualize to check whether the face detector works
            """
            plt.figure()
            plt.subplot(221); plt.imshow(left_eye.cpu().squeeze().permute(1, 2, 0))
            plt.subplot(222); plt.imshow(righteye.cpu().squeeze().permute(1, 2, 0))
            plt.subplot(223); plt.imshow(face.cpu().squeeze().permute(1, 2, 0))
            plt.subplot(224); plt.imshow(mask.cpu().squeeze().reshape(36, 64))
            plt.show()
            """
            ############################# Predictions #################################
            headtilt, headlower = leftright < 0.5, updown < 0.3
            headmove = headtilt or headlower

            gaze = F.softmax(EyeGaze(left_eye, righteye, face, mask), dim=-1)
            expression = F.softmax(FacialExpression(face), dim=-1)

            prob, region = torch.max(gaze, dim=-1)
            # assumes correct prediction if confidence is greater than 0.7 else choose 5
            region = region.tolist()[0] + 1 if prob.tolist()[0] > 0.7 else 5
    
            prob, exprsn = torch.max(expression, dim=-1)
            # predicts negative if the confidence is over 0.9
            if exprsn.tolist()[0] == 0 and prob > 0.9: exprsn = 1
            # predicts positive if the confidence is over 0.7
            elif exprsn.tolist()[0] == 2 and prob > 0.7: exprsn = 3
            # otherwise we predicts a neutral face
            else: exprsn = 2
            
            if headmove is True: region, exprsn = -1, -1
            result = [center, region, exprsn]
            results.append(result)
        if len(results) == 1: results = results + results
    # If no face is detected or head move, region and expression should be -1
    if isinstance(output, str) or len(results) == 0: 
        results = [[(float("inf"), float("inf")), -1, -1],
                   [(float("inf"), float("inf")), -1, -1]]
    return results


def run_personalize_detection(cap: cv2.VideoCapture):
    start = time.time()
    left, right = [], []
    while True:
        _, frame = cap.read()
        frame = frame[:, :, ::-1]
        # plt.imshow(frame); plt.show()
        PROCESS_TIME = time.time() - start
        results = detect(frame)
        # Assign detected results to different person
        results = sorted(results, key=lambda x: x[0])
        left.append([PROCESS_TIME, results[0][1], results[0][2]])
        right.append([PROCESS_TIME, results[1][1], results[1][2]])
        # print(PROCESS_TIME, left[-1], right[-1])
        if time.time() - start > 15: break
    return np.array(left).astype(np.float32), np.array(right).astype(np.float32)


"""
import glob
videos = glob.glob("D:\watch-edited\*.mp4")
for vid in videos:
    c = cv2.VideoCapture(vid)
    cnt = 0
    while True:
        start = time.time()
        ret, frame = c.read()
        frame = frame[:,:,::-1]
        if ret is False: break
        print(detect(frame))
        print("running time per frame: {:.4f} seconds".format(time.time() - start))
        cnt += 1
        if cnt == 100: break
"""