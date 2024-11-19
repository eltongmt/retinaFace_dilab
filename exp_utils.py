import basic_utils
from retinaface import RetinaFace
import os
from pathlib import Path
import datetime


PARENT = False
CHILD = True
model = RetinaFace.detect_faces

def predict_exp(INPATH, expid):
    INPATH = Path(INPATH)
    subjects = os.listdir(INPATH)
    subjects = [subject for subject in subjects if subject[:2] == "__"]
    count = 0

    for subject in subjects:
        count+=1
    
        REL_INPATH = INPATH / subject 

        if PARENT:
            agent = "parent"
            INPATH_FM = REL_INPATH / "cam08_frames_p"
            OUTPATH = REL_INPATH / "supporting_files" / f"bbox_annotations_{agent}_face"
            
            print(f"subject:{subject}, {count}/{len(subjects)}")
            os.makedirs(OUTPATH, exist_ok=True)

            try:
                basic_utils.predict_dir(INPATH_FM, OUTPATH, model)
            except Exception as e:
                print(e)
                
        
        if CHILD:
            agent = "child"
            INPATH_FM = REL_INPATH / "cam07_frames_p"
            OUTPATH = REL_INPATH / "supporting_files" / f"bbox_annotations_{agent}_face"

            print(f"subject:{subject}, {count}/{len(subjects)}")
            os.makedirs(OUTPATH, exist_ok=True)

            try:
                basic_utils.predict_dir(INPATH_FM, OUTPATH, model)
            except Exception as e:
                print(e)
    
    with open (r"C:\Users\multimaster\Documents\RetinaFace_doc\metadata\prediction-log.txt", "a") as file:
        file.write(f"{expid} {datetime.datetime.now()}")

for i in ["351"]:    
    predict_exp(f"M:\\experiment_{i}\\included", i) 