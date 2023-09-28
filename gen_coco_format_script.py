import json
import cv2
import os
import csv
from glob import glob


def resolve_coco_json():
    filepath = "/home/ubuntu/datasets/coco/annotations/coco_karpathy_train.json"
    # filepath = "/home/ubuntu/datasets/coco/annotations/coco_karpathy_test.json"
    # filepath = "/home/ubuntu/datasets/coco/annotations/coco_karpathy_val.json"
    # filepath = "/home/ubuntu/datasets/sample20230919/annotations/sample048_train.json"
    with open(filepath) as f:
        data = json.load(f)
    print(type(data), len(data))
    idx = 10
    print(data[idx])
    print(data[idx].keys())
    print(data[idx]["image"])
    print(data[idx]["caption"], type(data[idx]["caption"]))


def resolve_coco_gt_json():
    filepath = "/export/home/.cache/lavis/coco_gt/coco_karpathy_val_gt.json"
    filepath = "/content/LAVIS/lavis/output/BLIP2/Caption_sample20230919/20230920011/result/val_epoch0.json"
    with open(filepath) as f:
        data = json.load(f)
    print(type(data), len(data))
    print(data.keys())
    data = data["annotations"]
    idx = 0
    print(data[idx])
    print(data[idx].keys())
    


def vid2images():
    vid_root = "/home/ubuntu/datasets/sample20230911/videos"
    save_root = "/home/ubuntu/datasets/sample20230919/images/train"
    os.makedirs(save_root, exist_ok=True)
    vid_paths = glob("%s/*.mp4" % vid_root)
    for vid_path in vid_paths:
        count = 0
        reader = cv2.VideoCapture(vid_path)
        while True:
            ret, frame = reader.read()
            if not ret:
                break
            if count == 3:
                cv2.imwrite("%s/%s.jpg" % (save_root, os.path.basename(vid_path).split(".")[0]), frame)
            count += 1
        reader.release()


def gen_annos():
    csv_anno_path = "/home/ubuntu/datasets/sample20230911/filter_annotation/sample20230911.csv"
    coco_format_annos_train = []
    coco_format_annos_test = []
    with open(csv_anno_path) as f:
        reader = csv.reader(f)
        img_id = 0
        for line in reader:
            vid_name, tag = line[0], line[-1]
            if vid_name == "videoid":
                continue
            # print(vid_name, tag)
            img_path = "train/%s.jpg" % vid_name
            coco_format_annos_train.append({"image": img_path, "caption": tag, "image_id": img_id})
            coco_format_annos_test.append({"image": img_path, "caption": tag, "image_id": img_id})
            img_id += 1
    train_anno_path = "/home/ubuntu/datasets/sample20230919/annotations/sample048_train.json"
    test_anno_path = "/home/ubuntu/datasets/sample20230919/annotations/sample048_test.json"
    os.makedirs(os.path.dirname(train_anno_path), exist_ok=True)
    with open(train_anno_path, "w") as f:
        json.dump(coco_format_annos_train, f)
    with open(test_anno_path, "w") as f:
        json.dump(coco_format_annos_test, f)


def gen_gt_annos():
    json_anno_path = "/content/LAVIS/lavis/output/BLIP2/Caption_sample20230919/20230920024/result/val_epoch0.json"
    sampled_img_ids = []
    with open(json_anno_path) as f:
        data = json.load(f)
        
        for line in data:
            sampled_img_ids.append(line["image_id"])
    
  
    train_anno_path = "/content/sample20230919/annotations/sample048_train.json"
    annos = []
    imgs = []
    with open(train_anno_path) as f:
        data = json.load(f)
        
        for line in data:
            tag = line["caption"]
            img_id = line["image_id"]
            
            if img_id in sampled_img_ids:
              annos.append({"image_id": img_id, "caption": tag, "id": img_id})
              imgs.append({"id": img_id})
            
    coco_format_annos_train = {"annotations": annos, "images": imgs}
    train_anno_path = "/export/home/.cache/lavis/coco_gt/coco_karpathy_val_gt.json"
    
    with open(train_anno_path, "w") as f:
        json.dump(coco_format_annos_train, f)
    

# resolve_coco_json()
# vid2images()
gen_annos()
# resolve_coco_json()
# resolve_coco_gt_json()
# gen_gt_annos()