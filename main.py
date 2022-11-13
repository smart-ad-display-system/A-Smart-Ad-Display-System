import os
import re
import cv2
import numpy as np
from typing import Union
from copy import deepcopy
import time
import glob
from collections import OrderedDict
import subprocess
import random

VIDEO_PLAYER = r"C:\Program Files\Windows Media Player\wmplayer.exe"

start = time.time()
# Initializing Video Player
print("Initializing machine ...")
background = subprocess.Popen(
    VIDEO_PLAYER, shell=False, stdout=subprocess.PIPE, close_fds=True
    )

from models.facial_detector import run_personalize_detection
from recommender import recommander_kernel, adcoding, age_gender_recommmander
from models.age_gender_detector import run_age_gender_detector

if "log" not in os.listdir(): os.mkdir("log")
log_filename = int(input("Please Enter the Experiment's Index:"))
logname = "log/experiment-{:d}.txt".format(log_filename)
number_subjects = int(input("How many subjects in this experiment:"))


AD_ALREADY_WATCHED = []
if number_subjects == 2:
    assert os.path.exists(logname), "two subjects' experiment should be carried out after two single one have been proceeded!"
    with open(logname, "r") as f:
        log = f.read()
    matches = re.findall("AD_ALREADY_WATCHED:(.*)\n", log)
    assert len(matches) == 2
    for info in matches:
        AD_ALREADY_WATCHED += eval(info)

AD_ALREADY_WATCHED = list(set(AD_ALREADY_WATCHED))
print("""These ads are already watched during previous experiments:
{}""".format(AD_ALREADY_WATCHED)
)

if os.path.exists(logname): print("log for this experiment exists...")
logger = open(logname, "a+")
logger.write("Experiment: {} with {} subjects.\n".format(log_filename, number_subjects))
logger.write("""These ads are already watched during previous experiments:
{}\n""".format(AD_ALREADY_WATCHED)
)


ADVERTISEMENTS = glob.glob("D:/experiment_codes/ad_bank/*.mp4")
ADVERTISEMENTS = OrderedDict(
    {int(ad.split("-")[-1].split(".")[0]): ad for ad in ADVERTISEMENTS}
    )
AD_IDS = list(ADVERTISEMENTS.keys())
print("All required files are already initialized in {:.4f} seconds".format(time.time() - start))

start = time.time()
print("Initializing camera...")
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
print("Camera initialized in {:.4f} seconds".format(time.time() - start))

EXPERIMENT_ORDER = [
    ["RANDOM", "AGE_GENDER", "PERSONALIZE", "RANDOM"],
    ["RANDOM", "RANDOM", "AGE_GENDER", "PERSONALIZE"],
    ["RANDOM", "PERSONALIZE", "RANDOM", "AGE_GENDER"],
    ["RANDOM", "AGE_GENDER", "PERSONALIZE", "RANDOM"],
    ["RANDOM", "RANDOM", "AGE_GENDER", "PERSONALIZE"],
    ["RANDOM", "PERSONALIZE", "RANDOM", "AGE_GENDER"],

    ["RANDOM", "AGE_GENDER", "PERSONALIZE", "RANDOM"],
    ["RANDOM", "RANDOM", "AGE_GENDER", "PERSONALIZE"],
    ]
ORDER_MAPPING = {"PERSONALIZE": 0, "AGE_GENDER": 1, "RANDOM": 2}


def play_first_ad(
        gender: Union[list, np.ndarray], 
        age: Union[list, np.ndarray], 
        cap: cv2.VideoCapture=cap, 
        ad_watched: list=[],
        EXPERIMENTid: int=None
        ):
    unwatched = list(filter(lambda x: x not in ad_watched, AD_IDS))
    FIRST_AD = random.choice(unwatched)
    AD_ALREADY_WATCHED.append(FIRST_AD)
    AD_THIS_ROUND = [FIRST_AD]

    process = subprocess.Popen(
        "{} {}".format(VIDEO_PLAYER, ADVERTISEMENTS[FIRST_AD]), 
        shell=False, stdout=subprocess.PIPE, close_fds=True
    )
    time.sleep(0.5)
    # running for personalized recommander
    seen = deepcopy(AD_ALREADY_WATCHED) + AD_THIS_ROUND
    if EXPERIMENTid > 6: seen += list(range(1, 254))
    # print(seen)
    detected_results = run_personalize_detection(cap)
    personal_recommands = recommander_kernel(
        detected_results, 
        adcoding[FIRST_AD],
        already_seen = seen,
        logger=logger,
        EXPERIMENTid=EXPERIMENTid
        )
    AD_ALREADY_WATCHED.append(personal_recommands)
    AD_THIS_ROUND.append(personal_recommands)
    # running for age-and-gender recommander
    age_gender_recommands = age_gender_recommmander(
        gender, 
        age, 
        already_seen=ad_watched + AD_THIS_ROUND
        )
    # print(ad_watched + AD_THIS_ROUND)
    AD_ALREADY_WATCHED.append(age_gender_recommands)
    AD_THIS_ROUND.append(age_gender_recommands)
    # running for random recommander
    unwatched = list(filter(lambda x: x not in ad_watched + AD_THIS_ROUND, AD_IDS))
    rec = random.choice(unwatched)
    AD_ALREADY_WATCHED.append(rec)
    AD_THIS_ROUND.append(rec)
    return personal_recommands, age_gender_recommands, rec

# predict ages and genders for each participants
start = time.time()
input("Press Enter whenever you are ready:")
print("Running Age and Gender Detection, should take a few seconds...")
gender, age = run_age_gender_detector(cap)
print("Age and gender detected, age: {}, gender: {}".format(
    age, gender
    ))
logger.write("Age and gender detected, age: {}, gender: {}\n".format(age, gender))
input("Press enter when you are ready to perform this experiment:\n")


try:
    for EXPERIMENTid, ADorder in enumerate(EXPERIMENT_ORDER, 1):
        time.sleep(5)
        # play first advertisement and perform predictions
        MASK = list(range(254, 313)) if EXPERIMENTid <= 6 else list(range(1, 254))
        start = time.time()
        recommand_from = deepcopy(AD_ALREADY_WATCHED) + MASK
        # print(recommand_from)
        recommands = play_first_ad(gender, age, cap, recommand_from, EXPERIMENTid)
        print(
            "\tRound: {}, and the recommands are: {}".format(EXPERIMENTid, recommands)
            )
        logger.write(
            "\tRound: {}, and the recommands are: {}\n".format(EXPERIMENTid, recommands)
            )
        # print("Recommander takes {:.4f} seconds".format(time.time() - start))
        time.sleep(20 - (time.time() - start))
        # play advertisements based on the recommander's order
        for recommander in ADorder[1:]:
            ADRECid = recommands[ORDER_MAPPING[recommander]]
            process = subprocess.Popen(
                "{} {}".format(VIDEO_PLAYER, ADVERTISEMENTS[ADRECid]),
                shell=False, stdout=subprocess.PIPE, close_fds=True
            )
            time.sleep(20)
        # Sanity check: how many ads have been watched
        # print(len(AD_ALREADY_WATCHED))
        # Pause a round of experiments
        if EXPERIMENTid < 8:
            while True:
                key = input("Experiment paused, whether to continue:(y/n)\n")
                if key.lower() == "y": break
        print("The next round will start in 5 seconds")

except KeyboardInterrupt:
    cap.release()
print("Thanks for your cooperation, the experiemnt ends.")
logger.write("AD_ALREADY_WATCHED:{}\n".format(AD_ALREADY_WATCHED))
logger.close()
cap.release()