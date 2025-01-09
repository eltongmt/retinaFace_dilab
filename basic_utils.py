# Headers
import cv2
import pathlib as Path
import os 
from tqdm import tqdm 

# get_bb
def get_bb(model, img):
    preds = model(img)
    try:
        bbox = [val for val in preds["face_1"]["facial_area"]]
        bbox.append(preds["face_1"]["score"])

        return list(bbox)
    except:
        pass

    return 0

# write annotation
def write_txt(bbox, OUTDIR, error=False):

    with open(OUTDIR, "w") as file:
        if bbox:
            bbox.insert(0, 0)
            file.write(" ".join(map(str, bbox)))
        elif error:
            file.write("An error occured durring prediction")
        else:
            file.write("")

# draw bbox 
def draw_bbox(bbox, img, color=(255,0,0),font_size=1,line_width=1 ):
    img = cv2.imread(img)

    conf = bbox[-1]
    bbox = [int(point) for point in bbox[:4]]
    x1,y1,x2,y2 = bbox
    label = f"face:{conf: .2f}"

    (w, h), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

    bbox_img = cv2.rectangle(img, (x1,y1), (x2,y2),color, line_width)
    textimg = cv2.rectangle(bbox_img, (x1, y1 - 20), (x1 + w, y1), color, -1)
    img = cv2.putText(img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    return img

# predict img
def predict_img(OUTDIR, model, img):
    bbox = get_bb(model, img)
    
    write_txt(bbox, OUTDIR)
    return draw_bbox(bbox, img)

# predict dir 
def predict_dir(INDIR, OUTDIR, model):
    INF = os.listdir(INDIR)

    for file in tqdm(INF, desc="Predicting frames"):
        # ignore hidden files such as the terrible .db mac files 
        if file.split(".")[-1] != "jpg":
            continue 
        
        R_INF= os.path.join(INDIR, file)
        R_OUTF = os.path.join(OUTDIR,f"{file[:-4]}.txt")
        try:
            bbox = get_bb(model, R_INF)
            write_txt(bbox, R_OUTF)
        except:
            write_txt([], R_OUTF, True)


# utility to remove files in order to predict 
def remove_dir(INDIR):
    files = os.listdir(INDIR)

    for f in tqdm(files, desc="deleting files"):
        relative_frame = INDIR / f
        os.remove(relative_frame)
        
