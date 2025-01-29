import basic_utils
from retinaface import RetinaFace
import os
from pathlib import Path
import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("expID", type=int)
parser.add_argument("subID", type=str)
parser.add_argument("--CHILD", type=bool)
parser.add_argument("--PARENT", type=bool)
parser.add_argument("--RM", type=bool)


args = parser.parse_args()

model = RetinaFace.detect_faces

def save_log(expid, subject):
     with open (r"C:\Users\multimaster\Documents\RetinaFace_doc\metadata\prediction-log.txt", "a") as file:
            file.write(f"{expid} {subject} {datetime.datetime.now()}\n")

def predict_exp(INPATH, expid, PARENT=False, CHILD=True):
    INPATH = Path(INPATH)
    subjects = os.listdir(INPATH)
    subjects = [subject for subject in subjects if subject[:2] == "__"]
    count = 0

    for subject in subjects:
        predict_dyad(INPATH, subject, CHILD, PARENT)
        save_log(expid, subject)
        count+=1

def predict_dyad(INPATH, subject, CHILD, PARENT, RM):
    REL_INPATH = INPATH / subject 
    
    print(f'Predicting subject: {subject}')
    if PARENT:
        result = predict_subject(REL_INPATH,  "parent", RM)
     
    if CHILD:
        result =predict_subject(REL_INPATH, "child", RM)

    if result:
        save_log(351, subject)
        print(f"SUCCESS {subject} {datetime.datetime.now()}\n")
   

def predict_subject(REL_INPATH, agent = "child", RM=False):
    if agent == "child":
        INPATH_FM = REL_INPATH / "cam07_frames_p"
    elif agent == "parent":
        INPATH_FM = REL_INPATH / "cam08_frames_p"
    
    if not os.path.exists(INPATH_FM):
        print("subject has no FOV frame folder")
        return False

    OUTPATH = REL_INPATH / "supporting_files" / f"bbox_annotations_{agent}_face"
    os.makedirs(OUTPATH, exist_ok=True)

    if RM:
        basic_utils.remove_dir(OUTPATH)
    
    try:
        basic_utils.predict_dir(INPATH_FM, OUTPATH, model)
    except Exception as e:
            print(e)

    return True

def main():
    IN_PATH = Path(f"M:\\experiment_{args.expID}\\included")

    if args.CHILD is not None: 
        CHILD = args.CHILD
    else:
        CHILD = True
    if args.PARENT is not None:
        PARENT = args.PARENT
    else:
        PARENT = False

    if args.RM is not None:
        RM = args.RM
    else:
        RM = False
    predict_dyad(IN_PATH, args.subID, CHILD, PARENT, RM)
    
main()