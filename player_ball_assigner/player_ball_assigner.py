import sys
from utils import get_center,measure_distance

class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 70
    
    def assign_ball_to_player(self,players,ball_bbox): #players = tracks["players"]
        ball_position = get_center(ball_bbox) #returns 2 values for x and for y
        
        minimum_distance = 99999
        assigned_player = -1
        
        for player_id,player in players.items():
            player_bbox = player["bbox"]
            
            distance_left = measure_distance((player_bbox[0],player_bbox[-1]),ball_position) #bottom left of player_bbox se ball of distance, of left foot se ball ka distance
            distance_right = measure_distance((player_bbox[2],player_bbox[-1]),ball_position) #bottom right
            distance = min(distance_left,distance_right)
            
            if distance < self.max_player_ball_distance:
                if distance < minimum_distance:
                    assigned_player = player_id
                    minimum_distance = distance
        
        return assigned_player