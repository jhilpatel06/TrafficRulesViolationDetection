# demo/video_to_gif.py


import cv2
import imageio.v2 as imageio

video_path = "demo/output.mp4"
gif_path = "demo/output_gif.gif"

cap = cv2.VideoCapture(video_path)

frames = []

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Resize to reduce GIF size
    frame = cv2.resize(frame, (640, 360))

    # Convert BGR (OpenCV) to RGB (GIF)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frames.append(frame)

cap.release()

# Save GIF at 10 FPS
imageio.mimsave(gif_path, frames, fps=10, loop=0)

print(f"GIF saved successfully: {gif_path}")
