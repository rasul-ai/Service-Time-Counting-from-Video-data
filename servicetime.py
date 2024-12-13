import cv2
import time
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("./yolov8n.pt")

# Define the source video
source = "./fringestorez.mp4"

# Open the video file
cap = cv2.VideoCapture(source)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# selected area to search
bbox = (870, 90, 1250, 540)

# Initialize variables
resize_factor = 0.5  # Speed up processing
frame_interval = total_frames // 168  # 168 sec video
# print(frame_interval)
max_gap_frames = 3  # Tolerate up to 3 second of missing frames
customer_times = []  # Store time for each detected customer
current_customer_time = 0  # Track time for the current customer
person_in_bbox = False  # Flag to track if a person is currently in the bbox
gap_counter = 0  # Count consecutive frames without detection

frame_number = 0  # Frame counter
customer_count = 0  # Total number of unique customers
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1

    # Process every frame matching the interval
    if frame_number % frame_interval != 0:
        continue

    # Resize frame
    frame_resized = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)

    # Run inference
    results = model(frame_resized, verbose=False)
    detected_person = False

    # Check detections
    for result in results:
        for detection in result.boxes:
            class_id = int(detection.cls)
            confidence = detection.conf
            x1, y1, x2, y2 = detection.xyxy[0].tolist()

            # Scale bounding boxes back to original size
            x1 /= resize_factor
            y1 /= resize_factor
            x2 /= resize_factor
            y2 /= resize_factor

            # Check if detection is a person inside the defined bbox
            if class_id == 0 and confidence > 0.5:
                if x1 >= bbox[0] and y1 >= bbox[1] and x2 <= bbox[2] and y2 <= bbox[3]:
                    detected_person = True
                    break

    if detected_person:
        # Reset gap counter
        gap_counter = 0
        current_customer_time += 1
        time_gap = frame_number//frame_interval
        # print(time_gap)


        # If transitioning from not detected to detected
        if not person_in_bbox:
            person_in_bbox = True  # Update the state
    else:
        if person_in_bbox:
            # Increment gap counter if person is not detected
            gap_counter += 1
            # print(gap_counter)
            if gap_counter > max_gap_frames:
                # If the gap exceeds the tolerance, finalize current customer
                print("#######Customer#########")
                person_in_bbox = False
                customer_times.append(current_customer_time)
                current_customer_time = 0
                customer_count += 1  # Count this as a unique customer

# Capture the last customer's time if still detected
if person_in_bbox:
    customer_times.append(current_customer_time)
    customer_count += 1

# Calculate total and average time
total_time = sum(customer_times)
average_time = total_time / customer_count if customer_count > 0 else 0

# Display results
print(f"Total number of customers: {customer_count}")
print(f"Total time a person appears in the bounding box: {total_time:.2f} seconds")
print(f"Average time per customer: {average_time:.2f} seconds")

# Release video capture
cap.release()
