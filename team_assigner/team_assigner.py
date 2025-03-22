import cv2
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {} # {player_id: team_id} {42:1,21:1,22:2}
    
    def get_clustering_model(self,image):
        image_2d = image.reshape(-1,3)
        kmeans = KMeans(n_clusters=2,random_state=0,init="k-means++",n_init=1)
        kmeans.fit(image_2d)
        return kmeans
    
    def get_player_color(self,frame,bbox):
        #croping the image
        image  = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
        
        #selecting the top half
        top_half_image = image[0: int(image.shape[0]/2),:]
        
        #applying kmean
        kmeans = self.get_clustering_model(top_half_image)
        
        #to get labels 
        labels = kmeans.labels_ 
        
        #reshaping labels back to to same size as the picture 
        clustered_image = labels.reshape(top_half_image.shape[0],top_half_image.shape[1])
        
        #to determine kaunsa label is for which cluster
        corner_clustes  = [clustered_image[0,0], clustered_image[0,-1],clustered_image[-1,0],clustered_image[-1,-1]]
        non_player_cluster_label = max(set(corner_clustes),key=corner_clustes.count)
        player_cluster_label = 1- non_player_cluster_label
        player_color = kmeans.cluster_centers_[player_cluster_label]
        #to return the rgb
        return player_color
    
    def assign_team_color(self,frame,player_detections):
        players_colors = []
        for _ , player_detection in player_detections.items(): #i think yahan ham player_dict hi bhejnege and     basically again we dont need track id reminder that player_dict is something like {0:{bbox:[01,23,1,12]},1:{bbox:[22,21,21,2]}} <- player_dict
            bbox = player_detection['bbox']
            player_color = self.get_player_color(frame,bbox)
            players_colors.append(player_color)
        
        #ab players_colors me dono bahut saare colors add ho gye honge , half are very simiilar to green , the other half is very similar to white in our example. To unke cluster bnane ke liye we again use kmeans
        
        kmeans = KMeans(n_clusters=2,init="k-means++",n_init=1,random_state=0)
        kmeans.fit(players_colors)
        #store kra for further use 
        self.kmeans = kmeans
        
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]
    
    def get_player_team(self,frame,player_bbox,player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color = self.get_player_color(frame,player_bbox)
        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0] # kmeans expects a 2d array. reshape 1,-1 will conver a 1D array into 2D array with 1 row and auto number of columns(-1)      [171.11330698, 235.40316206, 142.52700922] -> [[171.11330698, 235.40316206, 142.52700922]] and [0] because wo output will be in an array , uski value will be whats inside the array 
        
        team_id+=1 #instead of 0,1 we want 1,2 
        
        if player_id == 93: #Goalkeeper ke liye hardcode krna pada 
            team_id=1
        
        self.player_team_dict[player_id] = team_id
        
        return team_id