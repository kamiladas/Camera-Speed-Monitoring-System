# Camera-Speed-Monitoring-System


## Project Description

This project was created as part of coursework at the Kielce University of Technology and aims to monitor road traffic, measure vehicle speeds, and recognize license plates. The system utilizes artificial intelligence and advanced image processing technologies to provide valuable real-time information.

## Project Assumptions

- Monitoring takes place on a one-way road.
- The system uses two measurement points (cameras) placed at a specific distance from each other.
- Vehicle speed is measured by calculating the time elapsed between the moments the vehicle crosses the first and second measurement points. Based on the known distance between the cameras and the measured time, the vehicle's speed is calculated. If a vehicle exceeds the speed limit, the event is recorded by the system.

## Implementation

### Components

The project consists of several key components, including vehicle detection, speed measurement, license plate recognition, and a user interface.

### Machine Learning Model

The project uses an advanced machine learning model for real-time vehicle detection. This model is based on deep neural networks, which are specifically trained to identify vehicles in various lighting and environmental conditions. Common architectures such as YOLO (You Only Look Once), SSD (Single Shot MultiBox Detector), or Faster R-CNN are typically used for object detection tasks.

### Key Functions

#### `perform_object_detection`

This function uses the preloaded model to process video frames for vehicle identification. It returns the locations of detected vehicles along with their probabilities.

```python
def perform_object_detection(sess, detection_graph, frame):
    image_np_expanded = np.expand_dims(frame, axis=0)

    # Get tensors
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    CAR_CLASS_ID = 3

    # Perform object detection
    boxes, scores, classes = sess.run(
        [detection_boxes, detection_scores, detection_classes],
        feed_dict={image_tensor: image_np_expanded})

    boxes = np.squeeze(boxes)
    scores = np.squeeze(scores)
    classes = np.squeeze(classes)

    return frame, boxes, scores, classes, CAR_CLASS_ID
```

#### Drawing Detection Areas

The project defines the area for vehicle detection as the section between two measurement points, i.e., the cameras. Processing the image within this designated area allows not only vehicle detection but also speed measurement and monitoring road behavior.

```python
def draw_transparent_line(img, pt1, pt2, color, thickness, alpha):
    mask = np.zeros_like(img)
    cv2.line(mask, pt1, pt2, color, thickness)
    return cv2.addWeighted(img, 1 - alpha, mask, alpha, 0)

def process_frame_cam1(frame1, point1, point2):
    global last_cross_time1, time_line1_crossed
    
    if point1 != (-1, -1) and point2 != (-1, -1):
        frame1 = draw_transparent_line(frame1, pt1, pt2, (0, 0, 255), 2, 0.3)
        
    return resize_frame(frame1, scale_percent)
```

#### Speed Measurement

The function `calculate_speed` calculates the vehicle's speed as the ratio of the distance traveled to the time taken. This method's accuracy depends on the precision of time measurement and the consistency of the distance between the cameras.

#### License Plate Recognition

License plate recognition is a critical component of advanced traffic monitoring systems. In this project, the functionality is implemented by the functions `plate_detect` and `read_plate_process`, which work together to detect and read license plates.

##### Haarcascade Algorithm

The project uses the Haarcascade algorithm for detecting license plates, particularly effective for Russian license plates.

```python
def detect_collision(detected_pixels, line_point1, line_point2, margin=-10):
    if line_point1 == (-1, -1) or line_point2 == (-1, -1):
        return False
    
    y_line = (line_point1[1]*scale_percent/100) + margin

    for pixel_y in detected_pixels:
        if pixel_y <= y_line:
            return True

    return False
```

#### Reading License Plates with EasyOCR

EasyOCR is used to read and process text from detected license plates.

## User Interface (GUI)

The graphical user interface (GUI) is designed to be intuitive and user-friendly, providing access to all important system functions, including video display from cameras, vehicle detection, speed reading, and license plate identification.

### Technologies and Tools

- **TensorFlow**: Used for training and running the deep learning models.
- **OpenCV**: Used for image processing and computer vision tasks.
- **Tkinter**: Used for creating the GUI.
- **EasyOCR**: Used for optical character recognition of license plates.

### Functionality

- **Vehicle Detection**: Using machine learning models to identify vehicles in real-time.
- **Speed Measurement**: Calculating vehicle speed based on the time taken to travel between two points.
- **License Plate Recognition**: Using OCR techniques to read and process license plate information.
- **Violation Log**: Documenting and tracking traffic violations.

## Conclusion

This project represents a significant step towards using AI in monitoring and managing road traffic. It offers possibilities for improving road safety and opens new avenues for analyzing traffic data and developing intelligent transportation systems.
