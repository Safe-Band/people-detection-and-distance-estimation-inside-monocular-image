"""handler for JHU DB

change JHU_PATH to the path of the JHU dataset on your computer
"""

import pandas as pd
import random as rd
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

JHU_PATH = r"/mnt/c/Users/gaeta/OneDrive - CentraleSupelec/3A/Safeband/Projet - InfoNum/Datasets/BigDSimages/jhu_crowd_v2.0/jhu_crowd_v2.0"


def plot_rectangle(ax:plt.Axes, x, y, w, h, color="r"):
    ax.add_patch(Rectangle((x-w/2, y-h/2), w, h, linewidth=1, edgecolor=color, facecolor="none"))



class JHU_handler:
    def __init__(self,kind="train"):
        self.path = JHU_PATH
        self.name = "JCHU_handler"
        self.description = "JCHU_handler is a class that handles the JCHU data"
        self.label_filename = "image_labels.txt"
        self.version = "1.0"
        
        self.kind = kind
        self.labels = self.get_labels()
        
        
    def change_kind(self,kind):
        self.kind = kind
        self.labels = self.get_labels()
        
    def get_labels(self):
        """return the labels of images in the dataset with informations about the scene-type, weather-condition and distractor
        
        kind = "train", "test" or "val"
        """
        df = pd.read_csv(os.path.join(self.path,self.kind,self.label_filename), sep=",", header=None,index_col=None,dtype=str)
        df.rename(columns={0:"filename", 1:"total-count", 2:"scene-type", 3:"weather-condition", 4:"distractor"}, inplace=True)
        return df
    
    def get_label(self,img_id):
        """return the label of an image
        
        img_id = the id of the image
        """
        return self.labels[self.labels['filename'] == img_id].to_dict(orient="records")[0]       
    
    def get_image(self, image_id):
        """return an image from the dataset
        
        kind = "train", "test" or "val"
        """
        infos = self.get_label(image_id)
        file_name = infos['filename']
        heads = pd.read_csv(
            os.path.join(self.path ,self.kind,"gt",f"{file_name}.txt"),
            sep=" ",
            header=None,
            )
        heads.rename(columns={0:"x", 1:"y",2:"w",3:"h",4:"o",5:"b"}, inplace=True)
        # img = cv2.imread(
        #     os.path.join(self.path,self.kind,"images",f"{file_name}.jpg"),
        #     )
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = Image.open(os.path.join(self.path,self.kind,"images",f"{file_name}.jpg"))
        return img, heads, infos
    
    def get_random_image(self):
        """return a random image from the dataset
        
        kind = "train", "test" or "val"
        """
        img_id = rd.choice(self.labels['filename'])
        return self.get_image(img_id) 
    
    def show_img(self, img, ax:plt.Axes):
        ax.imshow(img)

    def show_headboxes(self,heads,ax:plt.Axes):
        ax.scatter(heads["x"], heads["y"], color="g", marker=".", s=1)
        #headbox
        occ_color = {1:"g", 2:"orange", 3:"r"}
        for i in range(len(heads)):
            occlusion = heads.iloc[i]["o"]
            # print("occlusion",occlusion)
            plot_rectangle(
                ax,
                heads.iloc[i]["x"], 
                heads.iloc[i]["y"], 
                heads.iloc[i]["w"], 
                heads.iloc[i]["h"], 
                color=occ_color[occlusion]
                )
        

        
    
        