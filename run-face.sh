expID=351
subID="__20221218_10047"
one=false

#cuda path
source C:\\Users\\multimaster\\anaconda3\\etc\\profile.d\\conda.sh
conda activate tf 

if ($one); then
    python C:\\Users\\multimaster\\Documents\\RetinaFace_doc\\exp_utils.py "$expID" "$subID" --RM true
else
    while read line; do 
        python C:\\Users\\multimaster\\Documents\\RetinaFace_doc\\exp_utils.py "$expID" "$line" --RM true
    done < metadata/comp1.txt
fi

conda deactivate 

## IGNORE THIS IF YOLO-face_detection is not set up ## 
# path to face detection repo
#source C:\\Users\\multimaster\\Documents\\GitHub\\YOLO-face_detection\\.venv\\Scripts\\activate
#python C:\\Users\\multimaster\\Documents\\GitHub\\YOLO-face_detection\\scripts\\drawing_scripts\\post_processing_utils.py "$expID" "$subID" --SMOOTH True --VIDEO False
#deactivate 