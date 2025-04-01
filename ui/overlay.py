import cv2

def draw_overlay(frame, text, color=(0, 255, 0)):
    overlay = frame.copy()
    h, w, _ = frame.shape
    cv2.rectangle(overlay, (0, h - 80), (w, h), color, -1)
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, text, (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
