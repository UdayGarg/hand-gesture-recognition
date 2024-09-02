import cv2
import numpy as np

# Load template images for gesture recognition
fist_template = cv2.imread('/Users/uday/Desktop/Desktop-Udays_MacBook_Pro/MSAI/CS585/IMG_6176.jpg', 0)
open_hand_template = cv2.imread('/Users/uday/Desktop/Desktop-Udays_MacBook_Pro/MSAI/CS585/IMG_6177.jpg', 0)
thumbs_up_template = cv2.imread('/Users/uday/Desktop/Desktop-Udays_MacBook_Pro/MSAI/CS585/IMG_6178.jpg', 0)

# Check if the templates are loaded correctly
print(f"Fist template dimensions: {fist_template.shape}")
print(f"Open hand template dimensions: {open_hand_template.shape}")
print(f"Thumbs up template dimensions: {thumbs_up_template.shape}")

fist_template = cv2.resize(fist_template, (500, 500))
open_hand_template = cv2.resize(open_hand_template, (500, 500))
thumbs_up_template = cv2.resize(thumbs_up_template, (500, 500))

# Apply the same preprocessing to the templates as the frames
_, fist_template = cv2.threshold(fist_template, 60, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
_, open_hand_template = cv2.threshold(open_hand_template, 60, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
_, thumbs_up_template = cv2.threshold(thumbs_up_template, 60, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
fist_count = 0
open_hand_count = 0
thumbs_up_count = 0
# Function to preprocess frame
def preprocess_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Thresholding the image
    _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return thresh


# def match_template(frame, template, gesture_name):
#     # Ensure the template is smaller than the frame
#     if frame.shape[0] < template.shape[0] or frame.shape[1] < template.shape[1]:
#         # print("Template is larger than the frame. Resizing the template...")
#         scale_factor = min(frame.shape[0] / template.shape[0], frame.shape[1] / template.shape[1])
#         template = cv2.resize(template, (0, 0), fx=scale_factor, fy=scale_factor)
    
#     method = cv2.TM_CCOEFF_NORMED
#     res = cv2.matchTemplate(frame, template, method)
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
#     if max_val > 0.5:
#         top_left = max_loc
#         h, w = template.shape
#         bottom_right = (top_left[0] + w, top_left[1] + h)
#         cv2.rectangle(frame, top_left, bottom_right, 255, 2)
#         cv2.putText(frame, gesture_name, top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
#         return True
#     return False


# # Main function to capture video and process frames
# def main():
#     global fist_count, open_hand_count, thumbs_up_count
#     cap = cv2.VideoCapture(0)  # Capture video from webcam

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Preprocess the frame
#         thresh_frame = preprocess_frame(frame)

#         # Perform template matching for each gesture
#         if match_template(thresh_frame, fist_template, 'Fist'):
#             fist_count += 1
#         elif match_template(thresh_frame, open_hand_template, 'Open Hand'):
#             open_hand_count += 1
#         elif match_template(thresh_frame, thumbs_up_template, 'Thumbs Up'):
#             thumbs_up_count += 1

#         # Display the resulting frame
#         cv2.imshow('Frame', frame)

#         # Exit condition
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release the capture and close any open windows
#     cap.release()
#     cv2.destroyAllWindows()

#     # Print total counts for each gesture
#     print(f"Total fist count: {fist_count}")
#     print(f"Total open hand count: {open_hand_count}")
#     print(f"Total thumbs up count: {thumbs_up_count}")

# if __name__ == "__main__":
#     main()


def match_template(frame, template, gesture_name):
    method = cv2.TM_CCOEFF_NORMED
    res = cv2.matchTemplate(frame, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    match = max_val > 0.5
    top_left = max_loc if match else None
    return match, top_left, template.shape[::-1]  # Return the match status, top-left corner, and template size (w, h)

# Main function to capture video and process frames
def main():
    global fist_count, open_hand_count, thumbs_up_count
    cap = cv2.VideoCapture(0)  # Capture video from webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        thresh_frame = preprocess_frame(frame)
        gesture_detected = False
        gesture_name = 'Unknown'

        # Perform template matching for each gesture
        for template, name in [(fist_template, 'Fist'), (open_hand_template, 'Open Hand'), (thumbs_up_template, 'Thumbs Up')]:
            match, top_left, (w, h) = match_template(thresh_frame, template, name)
            if match:
                gesture_detected = True
                gesture_name = name
                bottom_right = (top_left[0] + w, top_left[1] + h)
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
                cv2.putText(frame, gesture_name, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                break  # Stop checking after the first detected gesture

        # If no gesture detected, display 'Unknown'
        if not gesture_detected:
            cv2.putText(frame, gesture_name, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close any open windows
    cap.release()
    cv2.destroyAllWindows()

    # Print total counts for each gesture
    print(f"Total fist count: {fist_count}")
    print(f"Total open hand count: {open_hand_count}")
    print(f"Total thumbs up count: {thumbs_up_count}")

if __name__ == "__main__":
    main()