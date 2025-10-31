import cv2, os
"""Разбивка видео на кадры для датасета"""
video = "crowd.mp4"
out_dir = "frames/"
os.makedirs(out_dir, exist_ok=True)

cap = cv2.VideoCapture(video)
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite(os.path.join(out_dir, f"frame_{frame_id:06d}.png"), frame)
    frame_id += 1

cap.release()
print(f"{frame_id} кадры {out_dir}")
