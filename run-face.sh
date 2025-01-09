expID=351
subID="__20221112_10041"

#cuda path
source C:\\Users\\multimaster\\anaconda3\\etc\\profile.d\\conda.sh

#path to retinaface
conda activate tf 
python C:\\Users\\multimaster\\Documents\\RetinaFace_doc\\exp_utils.py "$expID" "$subID"
conda deactivate 

# path to face detection repo
#source C:\\Users\\multimaster\\Documents\\GitHub\\YOLO-face_detection\\.venv\\Scripts\\activate
#python C:\\Users\\multimaster\\Documents\\GitHub\\YOLO-face_detection\\scripts\\drawing_scripts\\post_processing_utils.py "$expID" "$subID" --SMOOTH True --VIDEO False
#deactivate 

