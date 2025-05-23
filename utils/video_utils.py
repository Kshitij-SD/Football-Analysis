import cv2

def read_video(video_path):
    frames=[]
    cap=cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return frames  # Return an empty list
    
    while True:
        ret, frame = cap.read()
        #returns 2 things frame a and indicator
        # return 1 if frame is captured and 0 if not captured basically the end of a video
        if not ret:
            break
        frames.append(frame)
    return frames

def save_video(output_video_frames,output_video_path):
    fourcc=cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path,fourcc,24,(output_video_frames[0].shape[1],output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()