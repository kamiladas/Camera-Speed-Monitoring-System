import tkinter as tk
from threading import Thread
import cv2
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import tensorflow as tf
import os
from plate_detect import detect_and_save_license_plates, recognize_license_plate
from read_plate import process_image
from time import sleep
save_directory = "C:\\Users\\Adas\\OneDrive\\Desktop\\Projekt_Speed_detector\\detected_plates"
distance=None
speed_limit=None
timer_c1=0
timer_c2 =0
proces_video_started=False
input_value = None  # Globalna zmienna do przechowywania danych z Tkinter
frame_resized1=None
frame_resized2=None
input_value = None  # Globalna zmienna do przechowywania danych z Tkinter
MIN_CONFIDENCE = 0.97

drawing = False
drawing_line1 = True
current_cam = 1
point1, point2 = (-1, -1), (-1, -1)
point1_line2, point2_line2 = (-1, -1), (-1, -1)
scale_percent = 70

import time

last_cross_time1 = 0  # Zmienna do przechowywania czasu ostatniego przekroczenia linii 1
last_cross_time2 = 0  # Zmienna do przechowywania czasu ostatniego przekroczenia linii 2
cross_disable_time = 3

time_line1_crossed = None
time_line2_crossed = None

vehicles = {}


cap1 = cv2.VideoCapture('C:\\Users\\Adas\\OneDrive\\Desktop\\Projekt_Speed_detector\\video\\trafic_example2.mp4')
cap2 = cv2.VideoCapture('C:\\Users\\Adas\\OneDrive\\Desktop\\Projekt_Speed_detector\\video\\trafic_example2.mp4')

PATH_TO_FROZEN_GRAPH = "C:/Users/Adas/OneDrive/Desktop/Projekt_Speed_detector/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb"


   

def load_detection_model(path_to_frozen_graph):
    detection_graph = tf.compat.v1.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(path_to_frozen_graph, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

detection_graph = load_detection_model(PATH_TO_FROZEN_GRAPH)




def detect_collision(detected_pixels, line_point1, line_point2, margin=-10):
    # Sprawdzanie, czy linia jest poprawnie zdefiniowana
    if line_point1 == (-1, -1) or line_point2 == (-1, -1):
       
        return False  # Brak kolizji, ponieważ linia nie jest poprawnie zdefiniowana
    
    # Zakładając, że linia jest pozioma, bierzemy tylko współrzędne y.
    y_line = (line_point1[1]*scale_percent/100) + margin  # Dodanie marginesu

    for _, pixel_y in detected_pixels:
        if pixel_y <= y_line:
           
            return True  # Kolidujący piksel

    return False  # Brak kolizji

def save_frame(frame, frame_number):
    def save_frame_thread():
        directory_path = "C:\\Users\\Adas\\OneDrive\\Desktop\\Projekt_Speed_detector\\screenshoots"
        filename = f"{frame_number}.jpg"
        filepath = os.path.join(directory_path, filename)
        
        cv2.imwrite(filepath, frame)
        print(f"Frame saved as {filepath}")

    # Utworzenie i uruchomienie nowego wątku
    thread = Thread(target=save_frame_thread)
    thread.start()
    thread.join()



def draw_transparent_line(img, pt1, pt2, color, thickness, alpha):
    mask = np.zeros_like(img)
    cv2.line(mask, pt1, pt2, color, thickness)
    return cv2.addWeighted(img, 1 - alpha, mask, alpha, 0)



#############################################################################################################################################
def process_frame_cam1(frame1, point1, point2):
    global last_cross_time1, time_line1_crossed
    
    
    if point1 != (-1, -1) and point2 != (-1, -1):
        frame1 = draw_transparent_line(frame1, point1, point2, (0, 0, 255), 2, 0.3)
        
        # Jeśli samochód przekroczył linię, dodajemy anotację do obrazu

    return resize_frame(frame1, scale_percent)
##################################################################################################################################################
def process_frame_cam2(frame2, point1_line2, point2_line2):
    global last_cross_time2, time_line2_crossed

    if point1_line2 != (-1, -1) and point2_line2 != (-1, -1):
        frame2 = draw_transparent_line(frame2, point1_line2, point2_line2, (0, 255, 0), 2, 0.3)
        
        
    
    return resize_frame(frame2, scale_percent)

def resize_frame(frame, scale_percent):
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    new_dim = (width, height)
    return cv2.resize(frame, new_dim, interpolation=cv2.INTER_AREA)

def draw_line(event, x, y, flags, param):
    global current_cam, scale_percent

    # Określenie, czy kursor jest w obszarze kamery 1 czy 2 na podstawie wartości x
    if x < frame_resized1.shape[1]:  # Jeśli kursor jest w obszarze kamery 1
        draw_line_cam1(event, x, y, flags, param)
    else:  # Jeśli kursor jest w obszarze kamery 2
        # Przesunięcie wartości x, aby odnieść się do lokalnej pozycji w obszarze kamery 2
        x -= frame_resized1.shape[1]
        draw_line_cam2(event, x, y, flags, param)

def draw_line_cam1(event, x, y, flags, param):
    global point1, point2, drawing, drawing_line1, scale_percent
    x = int(x * (100 / scale_percent))
    y = int(y * (100 / scale_percent))

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        if drawing_line1:
            point1 = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            if drawing_line1:
                point2 = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if drawing_line1:
            point2 = (x, y)


def draw_line_cam2(event, x, y, flags, param):
    global point1_line2, point2_line2, drawing, drawing_line1, scale_percent
    x = int(x * (100 / scale_percent))
    y = int(y * (100 / scale_percent))

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        if not drawing_line1:
            point1_line2 = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            if not drawing_line1:
                point2_line2 = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if not drawing_line1:
            point2_line2 = (x, y)




executor = ThreadPoolExecutor()
frame_number = 0


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

# Inicjalizacja śledzenia
tracker = cv2.TrackerKCF_create()

# Zmienna do przechowywania bieżących pozycji środka masy obiektów
current_object_positions = []
def camera1_crossing(vehicle_id, crossing_time):
    vehicles[vehicle_id] = crossing_time

# Funkcja do wywołania, gdy auto przekroczy linię w zasięgu kamery 2
def camera2_crossing(vehicle_id):
    if vehicle_id in vehicles:
        start_time = vehicles[vehicle_id]
        end_time = time.time()
        travel_time = end_time - start_time
        print(f"Auto {vehicle_id} pokonało odcinek w czasie: {travel_time} sekund.")
        # Tutaj możesz usunąć zapis o pojeździe, jeśli nie będzie już potrzebny
       ## del vehicles[vehicle_id]
        return travel_time  # Zwróć travel_time
    else:
        return None 


def draw_detected_objects(frame, boxes, scores, classes, CAR_CLASS_ID, return_detected_pixels=False):
    detected_pixels = []  # Lista pikseli wykrytych obiektów

    for box, score, class_id in zip(boxes, scores, classes):
        if class_id == CAR_CLASS_ID and score > MIN_CONFIDENCE:
            ymin, xmin, ymax, xmax = box
            (left, right, top, bottom) = (xmin * frame.shape[1], xmax * frame.shape[1],
                                         ymin * frame.shape[0], ymax * frame.shape[0])
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)

            # Jeśli potrzebujemy zwrócić piksele wykrytych obiektów
            if return_detected_pixels:
                detected_pixels.extend([(x, y) for x in range(int(left), int(right)) for y in range(int(top), int(bottom))])

    if return_detected_pixels:
        return frame, detected_pixels
    else:
        return frame
    
def show_speed(frame_resized2, speed):
    if speed is not None:
        if isinstance(speed, float):
            text = f"Speed: {speed:.2f} km/h"
        else:
        # Jeśli speed nie jest liczbą zmiennoprzecinkową, obsłuż to inaczej
            text = "Speed: N/A"
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (50, 50)  # Możesz zmienić te wartości, aby dostosować położenie tekstu
        font_scale = 1
        font_color = (0, 0, 255)  # Czerwony kolor tekstu  # Biały kolor tekstu
        line_type = 2
        cv2.putText(frame_resized2, text, position, font, font_scale, font_color, line_type)
    return frame_resized2

def show_plate(frame_resized2, plate):
    if plate is not None:
        text = f"Plate: {plate}"
    else:
        text = "Plate: N/A"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 0, 255)  # Czerwony kolor tekstu
    line_type = 2
    margin = 8  # Margines wokół tekstu

    # Obliczenie rozmiaru tekstu
    text_size = cv2.getTextSize(text, font, font_scale, line_type)[0]

    # Pozycja tekstu w prawym górnym rogu
    text_x = frame_resized2.shape[1] - text_size[0] - margin  # Szerokość obrazu minus szerokość tekstu
    text_y = text_size[1] + margin  # Wysokość tekstu plus margines

    # Pozycja prostokąta
    rect_start = (text_x - margin, text_y - text_size[1] - margin)
    rect_end = (text_x + text_size[0] + margin, text_y + margin)

    # Rysowanie prostokąta wokół tekstu
    cv2.rectangle(frame_resized2, rect_start, rect_end, font_color, cv2.FILLED)

    # Dodawanie tekstu
    cv2.putText(frame_resized2, text, (text_x, text_y), font, font_scale, (255, 255, 255), line_type)

    return frame_resized2
    
        

   

def process_video(cap1, cap2, detection_graph):
    cv2.namedWindow('Frame')
    cv2.setMouseCallback('Frame', draw_line)
    global frame_resized1, frame_resized2,drawing_line1,current_cam,proces_video_started,timer_c1, timer_c2 
    proces_video_started=True
    start_time = time.time() 
    frame1_count = 0
    vehicle_id = 0
    temp_id = 0
    detection_delay = 1
    calculated_speed=None
    display_start_time = None
    executor = ThreadPoolExecutor(max_workers=2)
    delay_frames = 40  # Liczba klatek opóźnienia dla wideo 2
    plate=None
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        while True:
            ret1, frame1 = cap1.read()
           
            if ret1 and frame1_count==0:
               ret2, frame2 =  ret1, frame1  # Zapamiętaj ostatnią klatkę z wideo 1      
            frame1_count+=1
        
            if frame1_count > delay_frames:
               ret2, frame2 = cap2.read()

            if not ret1:
                cap1.set(cv2.CAP_PROP_POS_FRAMES, 80)
                ret1, frame1 = cap1.read()

            if not ret2:
                frame1_count = 0
                frame2=temp_frame_c2
                cap2.set(cv2.CAP_PROP_POS_FRAMES, 110)

            temp_frame_c1 = frame1
            temp_frame_c2= frame2
            future_cam1 = executor.submit(process_frame_cam1, frame1, point1, point2)
            future_cam2 = executor.submit(process_frame_cam2, frame2, point1_line2, point2_line2)
             
            frame_resized1 = future_cam1.result()
            frame_resized2 = future_cam2.result()

            frame_resized1, boxes, scores, classes, CAR_CLASS_ID = perform_object_detection(sess, detection_graph, frame_resized1)
            frame_resized2, boxes2, scores2, classes2, CAR_CLASS_ID2 = perform_object_detection(sess, detection_graph, frame_resized2)
            frame_resized1, detected_pixels = draw_detected_objects(frame_resized1, boxes, scores, classes, CAR_CLASS_ID, return_detected_pixels=True)
            frame_resized2, detected_pixels2 = draw_detected_objects(frame_resized2, boxes2, scores2, classes2, CAR_CLASS_ID2, return_detected_pixels=True)
       
    
           
            collisionc1=detect_collision(detected_pixels, point1_line2, point1_line2)##c1
            collisionc2=detect_collision(detected_pixels2, point1, point2)  ##c2

            if collisionc1:
                if time.time() - timer_c1 >= detection_delay:
                    camera1_crossing(vehicle_id, time.time())
                    timer_c1 = time.time() 
                    print("kolizja cam1")
                    print(vehicle_id)
                    vehicle_id=vehicle_id+1
                    
            if(collisionc2):
                    if time.time() - timer_c2 >= detection_delay:
                       camera2_crossing(temp_id)
                       calculated_speed=calculate_speed(temp_id)                    
                       temp_id=temp_id+1
                       timer_c2 = time.time() 
                       print("kolizja cam2")
                       if calculated_speed is not None and speed_limit_entry is not None and calculated_speed>float(speed_limit_value): ### logika ograniczenia predkosci
                            save_frame(temp_frame_c2, temp_id-1)  # Zapisz klatkę     
                            plate_detect(temp_id-1)
                            plate=read_plate_process(temp_id-1)
                            


            
               
            if calculated_speed is not None and display_start_time is None:
                 display_start_time = time.time()

            if display_start_time is not None:
                frame_resized2 = show_speed(frame_resized2, calculated_speed)
                if plate is not None:
                   frame_resized2= show_plate(frame_resized2,plate)

            if display_start_time is not None and time.time() - display_start_time > 1:
               display_start_time = None  # Zresetuj timer

            combined_frame = np.hstack((frame_resized1, frame_resized2))
            cv2.imshow('Frame', combined_frame)

            key = cv2.waitKey(1)
            if key == ord('c'):
                drawing_line1 = not drawing_line1
                current_cam = 2 if current_cam == 1 else 1
                pass
            elif key == 27:
                break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
pass

speed_limit_entry = None
distance_entry = None
result_label = None
speed_limit_value=None

def calculate_speed(vehicle_id):
    global speed_limit, distance, speed_limit_value

    # Próba konwersji speed_limit i distance na liczby
    try:
        distance_value = float(distance)
        speed_limit_value = float(speed_limit)
    except ValueError:
        print("Błąd: Nieprawidłowe wartości liczbowe dla speed_limit lub distance.")
        return None

    # Sprawdź, czy wartości są sensowne
    if speed_limit_value <= 0 or distance_value <= 0:
        print("Błąd: Nieprawidłowe wartości speed_limit lub distance.")
        return None

    travel_time = camera2_crossing(vehicle_id)  # Zwraca czas podróży w sekundach

    # Sprawdź, czy czas podróży jest prawidłowy
    #if travel_time is None or travel_time <= 0:
      #  print("Błąd: Nieprawidłowy czas podróży. Rejestracja następnego przejazdu.")
      #  return "retry"  # Sygnalizuje konieczność ponownego pomiaru

    # Oblicz prędkość
    speed = (float(distance_value) / float(travel_time)) * 3.6  # Konwersja m/s na km/h
    print(f"Prędkość: {speed} km/h")
    return speed


def start_processing():
    global speed_limit_entry, distance_entry, result_label

    # Utwórz i uruchom wątek do przetwarzania wideo
    process_thread = Thread(target=process_video, args=(cap1, cap2, detection_graph))
    process_thread.start()
    process_button.pack_forget()

    # Dodaj pola do wprowadzania danych i nowy przycisk
    tk.Label(root, text="Ograniczenie prędkości (km/h):").pack()
    speed_limit_entry = tk.Entry(root)
    speed_limit_entry.pack()

    tk.Label(root, text="Dystans (m):").pack()
    distance_entry = tk.Entry(root)
    distance_entry.pack()


    tk.Label(root, text="Wynik:").pack()
    result_label = tk.Label(root, text="wynik Oo?")
    result_label.pack()

    submit_button = tk.Button(root, text="Przetwórz dane", command=process_data)
    submit_button.pack()

def plate_detect(vehicle_id):
    base_path = "C:\\Users\\Adas\\OneDrive\\Desktop\\Projekt_Speed_detector\\screenshoots\\"
    image_filename = f"{vehicle_id}.JPG"
    image_path = base_path + image_filename

    save_directory = "C:\\Users\\Adas\\OneDrive\\Desktop\\Projekt_Speed_detector\\detected_plates"

    # Definicja funkcji, która będzie uruchomiona w nowym wątku
    def process_and_save():
        detect_and_save_license_plates(image_path, save_directory,2.7, image_filename)

    # Utworzenie i uruchomienie nowego wątku
    process_thread = Thread(target=process_and_save)
    process_thread.start()
    process_thread.join()    

def process_data():
    global speed_limit,distance
    # Pobierz dane z pól Entry
    speed_limit = speed_limit_entry.get()
    distance = distance_entry.get()
    # Wyświetl wynik
    result_label.config(text="Twój przesłany tekst")
    pass


def read_plate_process(vehicle_id):
     result=process_image(vehicle_id, iterations=1)
     return result



   
    
root = tk.Tk()
root.title("Video Processing")

# Dodaj przycisk do GUI
process_button = tk.Button(root, text="Process Video", command=start_processing)
process_button.pack()

# Ustaw rozmiar okna
window_width = 600
window_height = 200
root.geometry(f"{window_width}x{window_height}")

# Pobierz rozmiary ekranu
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Oblicz współrzędne x i y
x_coordinate = (screen_width - window_width) // 2
y_coordinate = screen_height - window_height - 200  # 50 pikseli od dolnej krawędzi

# Ustaw pozycję okna
root.geometry(f"+{x_coordinate}+{y_coordinate}")

# Uruchom pętlę główną
root.mainloop()
#process_video(cap1, cap2, detection_graph)

