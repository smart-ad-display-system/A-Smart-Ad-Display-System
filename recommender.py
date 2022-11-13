import time, copy
import numpy as np
import pandas as pd
import random
from typing import Union
from collections import OrderedDict
from math import sqrt

############################### Super Parameters ##############################
TopK = 5
start = time.time()
print("Loading Datasets...")
###################### Read and pre-processing ad-coding ######################
adcoding = OrderedDict()
data = np.array(pd.read_csv("./data/adcoding.csv"))
for adID, frame in zip(data[:, 0], data[:, 2:]):
    adcoding[int(adID)] = adcoding.get(int(adID), []) + [frame]
for adID, values in adcoding.items():
    values = np.array(values).reshape(-1, 9, 64)
    adcoding[int(adID)] = np.concatenate([values, np.zeros((len(values), 1, 64))], axis=1)
############# Read and pre-processing preference counted in exp 1 #############
preference = OrderedDict()
data = np.array(pd.read_csv("./data/exp1_databank.csv"))
for pid, frame, score, prefer in zip(data[:, 0], data[:, 1::3], data[:, 2::3], data[:, 3::3]):
    preference[int(pid)] = (frame, prefer)
############## Read and pre-processing the annotation of videos ###############
adsummary = OrderedDict()
data = np.array(pd.read_csv("./data/adsummary.csv", encoding = 'gb2312'))
for adID, frame in zip(data[:, 0], data[:, 7:].astype(np.float32)):
    adsummary[int(adID)] = frame
############ Read and pre-processing the age-gender recommender ###############
age_gender_database = OrderedDict()
data = pd.read_csv("./data/age_gender_databank.csv", encoding = 'gb2312')
for adID, target_gender, target_age in zip(data["adID"], data["target_gender"], data["target_age"]):
    gender, age = [target_gender], [target_age]
    if target_gender == 0: gender += [1, 2]
    if target_age == 0: age += [1, 2]
    for g in gender:
        for a in age:
            age_gender_database[(g, a)] = age_gender_database.get(
                (target_gender, target_age), []
                ) + [adID]

print("Dataset loaded in {:.4f} seconds".format(time.time() - start))
###############################################################################
########################## Advertisement Recommander ##########################
###############################################################################
def object_of_interest(predict: np.ndarray, adinfo: np.ndarray):
    """
    @params: predict, (Nx3), [time, region, expression]
    @params: adinfo, (Mx9x64), whether an element exists in a frame
    """
    # Assumes 30fps for a single advertisement, and annotated with 6fps
    frameid = (predict[:, 0] * 6).astype(np.int32).clip(0, adinfo.shape[0]-1)

    region = predict[:, 1].astype(np.int32)
    expression = predict[:, 2].astype(np.int32)
    # Locate the corresponding frame from the advertisement
    object_sbj_focus = adinfo[frameid, region]

    # Count the objects in the whole advertisement
    count_per_object = (np.sum(adinfo, axis=1) > 0).astype(np.int32)
    count_per_object = np.sum(count_per_object, axis=0) + 1e-6

    # sanity check and sum up along frames, count the proportion
    check = lambda x: np.zeros_like(count_per_object) if len(x.shape) < 2 else np.sum(x, axis=0)
    neg = check(object_sbj_focus[expression == 1]) / count_per_object
    neu = check(object_sbj_focus[expression == 2]) / count_per_object
    pos = check(object_sbj_focus[expression == 3]) / count_per_object

    no_attn  = 1 - neg - neu - pos
    # Pending preference
    prefer  = np.bitwise_and(pos > 0.3, pos > neg)
    dislike = np.bitwise_and(neg > 0.3, neg > pos)
    
    nopayat = (no_attn > 0.5)
    # filter for order reason
    nopayat[dislike is True] = False
    nopayat[prefer is True] = False

    score_per_object = prefer * 2 - dislike * 2 - nopayat
    return score_per_object, count_per_object.astype(np.int32)


def matching_neighbor(results: tuple, preference: OrderedDict):
    """
    # @params: results, (score_per_object, count_per_object) results from object_of_interest
    # @params: preference, information of all 120 subjects in experiment 1
    """
    score_per_object, count_per_object = results
    matching_score = []
    # calculate the matching score between two subjects
    for pid, (frameinfo, prefer) in preference.items():
        # find out the co-occured objects
        cooccur_objects = np.bitwise_and(frameinfo > 0, count_per_object > 0)
        # l2 norm between two cropped vectors
        dist = np.linalg.norm(
            score_per_object[cooccur_objects] - prefer[cooccur_objects]
            )
        dim = np.sum(cooccur_objects)
        if dim == 0: 
            sim = 0
        elif dist == 0: 
            sim = 10
        else: 
            # sim = sqrt(np.sum(cooccur_objects)) / dist
            sim = 1. / dist
        matching_score.append((sim, pid, dist, dim))
    # find out the closest neighbor
    matching_score = sorted(matching_score, key=lambda x:x[0], reverse=True)
    topk_matches = matching_score[:TopK]
    i = 0
    while matching_score[TopK+i][0] == matching_score[0][0]:
        topk_matches.append(matching_score[TopK+i])
        i += 1

    return topk_matches


def object_of_interest_interpolation(results: tuple, topk_matches: list):
    """
    This function predicts the implied preference for unseen objects
    """
    score_per_object, count_per_object = results
    # interpolation for missing values
    score_interpolated = np.zeros((np.sum(count_per_object == 0), ))
    accumulate_weight = 0
    for sim, pid, dist, dim in topk_matches:
        count, prefer = preference[pid]
        score_interpolated += sim * prefer[count_per_object == 0]
        accumulate_weight += sim
    score_interpolated /= accumulate_weight
    # score for negative, positive, neutral or pay no attention
    pos = score_interpolated > 0
    neg = score_interpolated < -1
    rnd = np.bitwise_and(score_interpolated > -1, score_interpolated < 0)
    # mapping the missing values for perference
    score_interpolated[pos] = 2
    score_interpolated[neg] = -2
    score_interpolated[rnd] = np.round(score_interpolated[rnd], decimals=0)
    # interpolation missing values for perference
    score_per_object[count_per_object == 0] = score_interpolated
    return score_per_object


def advertisement_recommander(
    embeddings: Union[list, tuple, np.ndarray], 
    adsummary: np.ndarray,
    EXPERIMENTid: int,
    logger
):
    """ Advertisement Recommander for multiple persons """
    # num_ads
    rank_per_person = np.zeros((len(adsummary), ))
    adIndex = np.array([adID for adID, frame in adsummary.items()])
    
    for person_id, person_embd in enumerate(embeddings):
        emb = person_embd > 0
        scoreboard = []
        for adID, frame in adsummary.items():
            score = np.dot(emb, frame)
            scoreboard.append(score)
        # num_ads
        scoreboard = np.array(scoreboard); adIndex = np.array(adIndex)
        
        message = f'''
        preference person_id {person_id}, score per ad:
        {np.hstack([adIndex.reshape(-1, 1), scoreboard.reshape(-1, 1)]).T}'''
        print(message)
        logger.write(message + '\n')
        
        sortedScore = sorted(set(scoreboard), reverse=True)
        # sum rank ids
        cache = copy.deepcopy(rank_per_person.copy())
        for score_id, score in enumerate(sortedScore):
            rank_per_person[scoreboard == score] += score_id
        
        message = f'''
        preference person_id {person_id}, rank per ad:
        {np.hstack([adIndex.reshape(-1, 1), (rank_per_person - cache).reshape(-1, 1)]).T}'''
        print(message)
        logger.write(message + '\n')
        
    bestMatches = list(
        adIndex[rank_per_person == rank_per_person.min()]
    )
    recommands = random.choice(bestMatches)
    message = f'''recommanded ad: {recommands}'''
    print(message)
    logger.write(message + '\n')
    
    '''
    # analyze kernel person
    kernel_person = (EXPERIMENTid - 1) % 2
    emb = embeddings[kernel_person] > 0
    scoreboard = []
    for adID, frame in adsummary.items():
        score = np.dot(emb, frame)
        scoreboard.append((score, adID))
    scoreboard = sorted(scoreboard, key=lambda x: x[0], reverse=True)
    max_score = scoreboard[0][0]
    best_matches = list(filter(lambda x: x[0] == max_score, scoreboard))
    # analyze the second person
    emb = embeddings[1 - kernel_person] > 0
    scoreboard = []
    for _, adID in best_matches:
        score = np.dot(emb, adsummary[adID])
        scoreboard.append((score, adID))
    scoreboard = sorted(scoreboard, key=lambda x: x[0], reverse=True)
    max_score = scoreboard[0][0]
    best_matches = list(filter(lambda x: x[0] == max_score, scoreboard))
    recommands = random.choice(best_matches)
    '''
    return recommands


def recommander_kernel(
        facial_analyze: Union[list, tuple, np.ndarray], 
        adinformation: np.ndarray,
        already_seen: Union[list, tuple, np.ndarray] = [], 
        logger=None,
        preference: OrderedDict = preference,
        EXPERIMENTid: int=None,
        adsummary: OrderedDict = adsummary
        ):
    """ recommander kernel for single(or multiple) persons """
    # input refinement
    if isinstance(facial_analyze, np.ndarray):
        facial_analyze = (facial_analyze, )
    # make prediction for all the subjects
    embeddings = []
    for facial in facial_analyze:
        print(facial.shape)
        results = object_of_interest(facial, adinformation)
        topk_matches = matching_neighbor(results, preference)
        embedding = object_of_interest_interpolation(results, topk_matches)
        embeddings.append(embedding)

    for emb in embeddings:
        # emb = (emb > 0).astype(np.float32)
        print("The embeddings for subjects are:\n{}".format(emb))
        logger.write("The embeddings for subjects are:\n{}\n".format(emb))
    # Use embeddings to make predictions
    ad_bank = OrderedDict({
            adID: frame \
            for adID, frame in adsummary.items() \
                if adID not in already_seen
            })
    recommands = advertisement_recommander(embeddings, ad_bank, EXPERIMENTid, logger)
    return recommands


def age_gender_recommmander(
        gender: list, 
        age: list, 
        age_gender_database: OrderedDict=age_gender_database,
        already_seen: list=[]
        ):
    database = OrderedDict({
        key: [adID for adID in value if adID not in already_seen] \
            for key, value in age_gender_database.items()
        })
    age = 0 if age[0] != age[1] else age[0]
    gender = 0 if gender[0] != gender[1] else gender[0]
    recommand_from = database[(gender, age)]
    if len(recommand_from) == 0:
        recommand_from = [
            k for k in adcoding.keys() if k not in already_seen
            ]
    return random.choice(database[(gender, age)])

"""
test = pd.read_csv("./data/test.csv")
test = test[test["pid"] == 4]

for _ in set(test["adID"]):
    _slice = test[test["adID"] == 208]
    _slice = np.array(_slice).astype(np.float32)[:, [4, 2, 1]]
    _slice[:, 0] = _slice[:, 0] / 30.

    results = recommander_kernel(_slice, adcoding[174])
    print(results)
    break
"""