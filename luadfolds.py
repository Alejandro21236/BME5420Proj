import pandas as pd
import csv
import numpy as np
import pathlib as path

path1 = "..." #anonymized location of univ2 embeddings on supercluster
Slides = []
case_id = []
for file in path.Path(path1).iterdir():
    if file.suffix == ".npy":
        Slides.append(file.stem)
        case_id.append(file.stem[:12])
#reduce each file.stem to the first 12 characters to ge the case id

#make stratified train/val/testsplt of the data
#remove duplicates from case_id and Slides
dedup_case_id = list(set(case_id))
#keep only the first slide for each case_id
dedup_slides = []
for case in dedup_case_id:
    for i in range(len(case_id)):
        if case_id[i] == case:
            dedup_slides.append(Slides[i])
            break


fold1 = []
fold2 =[]
fold3 = []
fold4 = []
fold5 =[]


fold_size = int(len(dedup_slides)/5)
fold1 = dedup_slides[:fold_size]
fold2 = dedup_slides[fold_size:fold_size*2]
fold3 = dedup_slides[fold_size*2:fold_size*3]
fold4 = dedup_slides[fold_size*3:fold_size*4]
fold5 = dedup_slides[fold_size*4:]
split_by_slide = {}

for fold in [fold1,fold2,fold3,fold4,fold5]:
    train = fold[:int(len(fold)*0.6)]
    val = fold[int(len(fold)*0.6):int(len(fold)*0.75)]
    test = fold[int(len(fold)*0.75):]
    split_by_slide.update({slide: "train" for slide in train})
    split_by_slide.update({slide: "val" for slide in val})
    split_by_slide.update({slide: "test" for slide in test})
    
labels = [[slide,1] for slide in train] + [[slide,2] for slide in val] + [[slide,3] for slide in test]
    
fold_labels = [[slide,1] for slide in fold1] + [[slide,2] for slide in fold2] + [[slide,3] for slide in fold3] + [[slide,4] for slide in fold4] + [[slide,5] for slide in fold5]

# all columns must be in the final csv
FINAL = pd.DataFrame(fold_labels, columns=["slide_id", "fold"])
FINAL["case_id"] = FINAL["slide_id"].map(dict(zip(Slides, case_id)))
FINAL["split"] = FINAL["slide_id"].map(split_by_slide)

FINAL.to_csv("slide_fold_assignments.csv", index=False)

#add them ECMscores 
jung = pd.read_csv("ssgseanomrscore.csv",index_col = 0)
#merge the two on case_id
FINAL = FINAL.merge(jung, left_on = "case_id", right_index = True)
FINAL.to_csv("groundtruth.csv", index = False)
