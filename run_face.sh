
#path constants
conda_path="C:\\Users\\multimaster\\anaconda3\\etc\\profile.d\\conda.sh"
yolovm_path="C:\\Users\\multimaster\\documents\\YOLO-Object-Detection-Project\\venv\\Scripts\\activate"
retinaPython="C:\\Users\\multimaster\\Documents\\retinaFace_dilab\\utils\\exp_utils.py"
yoloPython="C:\\Users\\multimaster\\Documents\\GitHub\\YOLO-face_detection\\scripts\\drawing_scripts\\post_processing_utils.py"
source "$conda_path"


## PREDICTION MODES ## 

# This script has two prediction modes: singular subject and list of subjects
#   to predict just one subject set the many argument to false and set subID to the target subject
#   to predict multiple subjects set the many argument to true and provide the path to a .txt file with a the subjects
#   an example can be found under metadata/distributed_processing/comp1.txt
#                        
expID=351
many=true
CHILD=1
PARENT=0

subID="__20221112_10041"
subList="metadata/test_subs.txt"

## PREDICTION ARGUMENTS  ##

# There are two scripts used for face processing. 
# one predicts and or removes and normalizes the predictions, to use this script set the post argument to true
# The other smooths the predictions and creates a video with the predictions, to use this set the 
# pred flag to true
# to use both set both flags to true, pred will be exectuted before post 
pre=1
post=1

# To create face predictions set the PRED argument to 1
PRED=0
# To normalize existing face predictions set the NORM argument to 1
NORM=0
# In case a subjects face predictions need be re run one must remove the existing predictions
# this can be done by setting RM to 1
RM=0
# To smooth predictions set SMOOTH to 1
SMOOTH=0
# To create a video with predictions set VIDEO to 1
VIDEO=0

# The actions above can all be performed during one run, so for a new subject all the flags except 
# RM will be true. The actions are performed in the following order RM -> PRED -> NORM -> SMOOTH -> VIDEO 

if ($many); then
    while read line;  do
        if [ "$pre" -eq 1 ]; then
            conda activate tf
            python "$retinaPython" "$expID" "$line" --CHILD "$CHILD" --PARENT "$PARENT" --PRED "$PRED" --NORM "$NORM" --RM "$RM" --pre "$pre" 
            conda deactivate
        fi
        if [ "$post" -eq 1 ] ; then
            source "$yolovm_path"
            python "$retinaPython" "$expID" "$line" --CHILD "$CHILD" --PARENT "$PARENT" --SMOOTH "$SMOOTH" --VIDEO "$VIDEO" --post "$post"
            deactivate 
        fi
        #FOR DEGUGGING echo "$line" >> metadata/final_list.txt
    done < "$subList"
else
    if [ "$pre" -eq 1 ]; then
        conda activate tf
        python "$retinaPython" "$expID" "$subID" --CHILD "$CHILD" --PARENT "$PARENT" --PRED "$PRED" --NORM "$NORM" --RM "$RM" --pre "$pre"
        conda deactivate
    fi
    if [ "$post" -eq 1 ]; then
        source "$yolovm_path"
        python "$retinaPython" "$expID" "$subID" --CHILD "$CHILD" --PARENT "$PARENT" --SMOOTH "$SMOOTH" --VIDEO "$VIDEO" --post "$post"
        deactivate 
        
    fi
    #FOR DEGUGGING echo "$subID" >> metadata/final_list.txt
fi

## fyi we call the script twice because they have to be exectued
# in different environments, utralitytics can be installed and executed in
# tf vm(python3.9) but obj_detector uses some python features not in 3.9
# so technically if someone wants to refactor that code we only have to 
# call the script once 