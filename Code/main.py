"""
Owner: 
Uday Garg U11302797
"""

import cv2
import numpy as np


def detect_skin(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 48, 80], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    return mask

def find_hand_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=lambda x: cv2.contourArea(x)) if contours else None

def extract_features(contour):
    features = {'fingers': 0, 'thumb': False}
    if contour is not None and len(contour) > 3:
        hull = cv2.convexHull(contour, returnPoints=False)
        if hull is not None and len(hull) > 3:
            try:
                defects = cv2.convexityDefects(contour, hull)
            except cv2.error as e:
                print(f"Error calculating convexity defects: {e}")
                return features  
            
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(contour[s][0])
                    end = tuple(contour[e][0])
                    far = tuple(contour[f][0])
                    
                    a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                    b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                    c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                    
                    angle = np.arccos((b**2 + c**2 - a**2) / (2*b*c))  
                    
                    # Angle less than 90 degree, considered as potential finger
                    if angle <= np.pi / 2 and d > 10000:
                        features['fingers'] += 1
                        if angle > np.pi / 3 and (start[1] > far[1] or end[1] > far[1]):  
                            features['thumb'] = True

    return features


def recognize_gesture(features):
    fingers = features.get('fingers', 0)
    has_thumb = features.get('thumb', False)
    if fingers == 4 and has_thumb:
        return "Open Hand"
    elif fingers == 1:
        return "Pointing"
    elif fingers == 2 and not has_thumb:
        return "Peace"
    elif fingers < 2:
        return "Fist"
    return "Unknown"


def main():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # preprocessed = preprocess_frame(frame)
        skin_mask = detect_skin(frame)
        hand_contour = find_hand_contours(skin_mask)
        features = extract_features(hand_contour)
        gesture = recognize_gesture(features)

        if hand_contour is not None:
            x, y, w, h = cv2.boundingRect(hand_contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, gesture, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Gesture', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
