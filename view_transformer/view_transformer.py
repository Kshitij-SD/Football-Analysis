
import numpy as np
import cv2
class ViewTransformer():
    def __init__(self):
        court_width = 68 #actual court width 68 meters
        court_length = 23.32 #how we got this value -> journal
        
        #vertices of wo points jinki madad se ham transform k rrhe hai prespective
        self.pixel_vertices = np.array([
            [110,1035],
            [265,275],
            [910,260],
            [1640,915]
        ])
        
        self.target_vetices = np.array([
            [0,court_width],
            [0,0],
            [court_length,0],
            [court_length,court_width]
        ]) 
        
        self.pixel_vertices  = self.pixel_vertices.astype(np.float32)
        self.target_vetices  = self.target_vetices.astype(np.float32)
        
        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices,self.target_vetices)
    
    def transform_point(self,point):
        p = (int(point[0]),int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_vertices,p,False) >= 0 #It returns: +1 if the point is inside 0 if the point is exactly on the edge -1 if the point is outside
        if not is_inside:
            return None
        
        reshaped_point =  point.reshape(-1,1,2).astype(np.float32) #-1,1,2 me rehshape kiya because cv2.perspectiveTransform issi format me input leta hai
        transform_point = cv2.perspectiveTransform(reshaped_point, self.perspective_transformer)
        
        return transform_point.reshape(-1,2) # then wapis reshape kiya 
        
    def add_transformed_positions_to_tracks(self,tracks):
        for object,object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info["position_adjusted"] # position_adjusted -> [x,y] 
                    position = np.array(position)
                    position_trannsformed = self.transform_point(position)
                    if position_trannsformed is not None:
                        position_trannsformed = position_trannsformed.squeeze().tolist()
                    tracks[object][frame_num][track_id]["position_transformed"] = position_trannsformed