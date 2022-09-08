from math import nan
import os
import pandas as pd
import zipfile
from zipfile import ZipFile
from PIL import Image
import json
import numpy as np

AcinoSet=os.getcwd()

# annotation id
# Unzip Acino data 
for file in os.listdir(AcinoSet):
    if zipfile.is_zipfile(file):
        new_folder=os.path.splitext(file)[0]
        if not os.path.exists(new_folder):
            os.mkdir(new_folder)
            with zipfile.ZipFile(file,'r') as zip_ref:
                zip_ref.extractall(new_folder)



with open('ap10k_as_train.json','r+') as f:
    f_data=json.load(f)
    img_ids=[int(os.path.splitext(img)[0]) for img in os.listdir('C:\\Users\\julia\\mmpose\\images\\data')]
    if not len(img_ids)==0:
        img_no=max(img_ids)+1
    else:
        img_no=0
    for dir in os.listdir(AcinoSet):
        if os.path.isdir(dir): # file = 05_03_2019LilyFlick1CAM6
            for fi in os.listdir(dir): # fi = collected_data.csv, as_0.png, as_1.png
                if os.path.splitext(fi)[1]==".png":
                    im=Image.open(dir+'/'+fi)
                    img_name='as_'+str(img_no)
                    im.save('AcinoImages'+'/'+img_name+'.jpg') #os.path.splitext(f)[0]
                    img_no+=1
                    os.remove(dir+'/'+os.path.splitext(fi)[0]+'.png')
                
                if fi=="CollectedData_UCT.csv":
                    df=pd.read_csv(dir+'/'+"CollectedData_UCT.csv")
                    #Extract keypoints to match AP10K
                    df=df.reindex(columns=(['scorer','UCT.2','UCT.3','UCT','UCT.1','UCT.46','UCT.47','UCT.48','UCT.49','UCT.44','UCT.45','UCT.26','UCT.27','UCT.28','UCT.29','UCT.32','UCT.33','UCT.4','UCT.5','UCT.6','UCT.7','UCT.10','UCT.11','UCT.34','UCT.35','UCT.36','UCT.37','UCT.40','UCT.41','UCT.14','UCT.15','UCT.16','UCT.17','UCT.20','UCT.21']))
                    img_no_1=img_no
                    for i in range(2,len(df)): # Iterate over rows
                        # Add image details
                        img_id=img_no_1+i-2
                        img_name_1='as_'+str(img_id)+'.jpg' #df.loc[i,'scorer'].split('/')[2].split('.')[0]+'.jpg'
                        
                        img_instance=dict(width=2704,height=1520,file_name=img_name_1,background=1,id=img_id)
                        f_data['images'].append(img_instance)

                        ann_id=img_id
                        
                        #Add annotations
                        kp=[]
                        num_kp=0

                        # Initial bbox:
                        x_min=2703
                        x_max=1
                        y_min=1519
                        y_max=1

                        for j in range(1,len(df.columns),2):
                            if pd.isnull(df.iloc[i,j]):
                                kp.append(0)
                                kp.append(0)
                                kp.append(0)
                            else:
                                x=int(np.round(pd.to_numeric(df.iloc[i,j])))
                                y=int(np.round(pd.to_numeric(df.iloc[i,j+1])))
                                #Add annotations
                                kp.append(x)
                                kp.append(y)
                                kp.append(2)
                                num_kp+=1

                                #Bbox
                                if x<x_min:
                                    x_min=x
                                if x>x_max:
                                    x_max=x
                                if y>y_max:
                                    y_max=y
                                if y<y_min:
                                    y_min=y

                        w=max(1,(x_max-x_min)) # BBox width
                        h=max(1,(y_max-y_min)) # BBox height
                        a=w*h #Area
                        
                        bbox=[x_min,y_min,w,h] # xywh format NB! y_min==top left corner
                        ann_instance=dict(image_id=img_id,iscrowd=0,category_id=25,num_keypoints=num_kp,keypoints=kp,bbox=bbox,id=ann_id,area=a)
                        f_data['annotations'].append(ann_instance)
                        
                        f.seek(0)
                        json.dump(f_data,f,indent=4)
                    os.rename(dir+"/CollectedData_UCT.csv",dir+"/done_collected.csv")
f.close()
                