from ultralytics import YOLO
import supervision as sv # tracker
import pickle
import os
import cv2 
from utils import get_center,get_width,get_foot_position
import numpy as np
import pandas as pd

class Tracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
    
    def interpolate_ball_position(self,ball_positions):
        ball_positions = [x.get(-1,{}).get("bbox",[]) for x in ball_positions] # convertinng ball positions into a list becuase to turn it into a pd dataframe we need a list. #x.get(-11) becuase ball ki track_id is -1 hamne hi hard code kri thi peeches
        
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])
        
        #interpolate missing values
        
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill() # agar pehle hi frame empty aa gye to unme kaise lagegi interpolation , to unke liye ye edge case
        
        ball_positions = [{-1:{"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions
    
    def detect_frames(self,frames):
        detections=[]
        batch_size=20
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size])
            #not using track here directly because it would also track the goalkeeper
            #we want to treat the goal keeper as a normal player
            detections+=detections_batch

        return detections
    
    def add_position_to_tracks(sekf,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position
    
    def get_object_tracks(self,frames,read_from_stub=False,stub_path=None):
        # stub path is added taki baar baar na run krna pde while developing the model, ek baar krke save krlo pickle me then wahan se read krte rho
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return(tracks)
        
        detections = self.detect_frames(frames)
        
        # a dictionary of list of dictionaries
        tracks={
            "players":[], #inside of the list will be multiple dictionaries.
                        # {0:{bbox:[01,23,1,12]},1:{bbox:[22,21,21,2]}} this if just for one frame say frame 0
                        # {0:{bbox:[01,23,1,12]},1:{bbox:[22,21,21,2]}} frame 1
            "referees":[], 
            "ball":[]
        }
        
        for frame_num,detection in enumerate(detections):
            cls_names = detection.names #will return something like
            #{0:person,1:goakeeper}
            cls_name_inv =  {v:k for k,v in cls_names.items()}
            # will convert it to {person:0, goalkeeper:1,ball:2}
            
            #convert to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            #converting Goalkeeper to player
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id]=="goalkeeper":
                    detection_supervision.class_id[object_ind]= cls_name_inv["player"]
            # to basically detection_supervision.class_id returns something like 
            #[2,2,2,2,2,2,2,2,2,1,0,2,2,2,2,2] where 2 is a person , 1 is a goalkeeper and 0 is a ball
            #we want to convert that 1 into 2. Directly kar skte hai but we dont want to hard code it hence the code above
            
            #Track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision) 
            
            #print(detection_supervision)
            
            #tracker_id=array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
            #tracks each object , as the object moves throught the frames the postions of the object will change through the array
            
            tracks["players"].append({}),
            tracks["ball"].append({}),
            tracks["referees"].append({})
            
            #tracking for the player and the refree not the ball
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4] 
                #refer to the journal file for the project for why we did this
                
                #just appending players to tracks["players"] and same for referees
                if cls_id == cls_name_inv['player']:
                    tracks['players'][frame_num][track_id] = {'bbox':bbox}
                    
                if cls_id == cls_name_inv['referee']:
                    tracks['referees'][frame_num][track_id] = {'bbox':bbox}
            
            #for ball
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                if cls_id == cls_name_inv['ball']:
                    tracks['ball'][frame_num][-1] = {'bbox':bbox} # yahan track_id ki jagah we'll just pass in 1 . To ball ki track_id hamesha ke liye ab 1 ho gyi
        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)
        
        return tracks

    def draw_team_ball_control(self,frame,frame_num,team_ball_control):
        #drawing the rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay,(1350,850),(1900,970),(255,255,255),cv2.FILLED)
        alpha = 0.4
        cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)
        
        #team_ball_control_calculations
        team_ball_control_till_frame = team_ball_control[:frame_num+1] 
        
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0] # shape[0] extrats the total number of rows giving us the count ki kitni jagah 1 aya hai
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        
        team_1 = (team_1_num_frames/(team_1_num_frames + team_2_num_frames))
        team_2 = (team_2_num_frames/(team_1_num_frames + team_2_num_frames))
        
        cv2.putText(frame,f"Team 1 Possesion : {team_1*100:.2f}%" , (1400,900),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
        cv2.putText(frame,f"Team 2 Possesion : {team_2*100:.2f}%" , (1400,950),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
        
        return frame
    
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_center , _ = get_center(bbox)
        width = get_width(bbox)
        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width),int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            lineType=cv2.LINE_4,
            color=color,
            thickness=2
        )
        
        #for the text box on the ellipse 
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2 - rectangle_height//2) + 15
        y2_rect = (y2 + rectangle_height//2) + 15
        
        if track_id is not None:
            cv2.rectangle(
                frame,
                (int(x1_rect),int(y1_rect)),
                (int(x2_rect),int(y2_rect)),
                color,
                cv2.FILLED
            )

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )
        
        return frame

    def draw_triangle(self,frame,bbox,color):
        y = int(bbox[1])
        x , _ = get_center(bbox)
        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20]
        ])
        
        cv2.drawContours(
            frame,
            [triangle_points],
            0,
            color,
            cv2.FILLED
        )
        # for the border
        cv2.drawContours(
            frame,
            [triangle_points],
            0,
            (0,0,0),
            2
        )
        return frame
    
    def draw_annotations(self,video_frames,tracks,team_ball_control):
        output_video_frames =[]
        for frame_num,frame in enumerate(video_frames):
            frame = frame.copy()
            
            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            
            #Draw Players
            for track_id,player in player_dict.items():
                #{0:{bbox:[01,23,1,12]},1:{bbox:[22,21,21,2]}} <- player_dict , 
                #track_id will be [0,1] , player will be [{bbox:[01,23,1,12]},{bbox:[22,21,21,2]}}]
                color = player.get("team_color",(0,0,255))
                frame = self.draw_ellipse(frame,player["bbox"],color,track_id)
                
                if player.get("has_ball",False):
                    frame = self.draw_triangle(frame,player["bbox"],(255,0,0)) 
            
            #Draw Referee
            for _ ,referee in referee_dict.items():
                frame = self.draw_ellipse(frame,referee["bbox"],(0,225,255))
            
            #Draw ball 
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame,ball["bbox"],(0,225,0))
            
            #Draw Ball Contorl
            frame = self.draw_team_ball_control(frame,frame_num,team_ball_control)
            
            output_video_frames.append(frame)
            
        return output_video_frames