import cv2
import os
import pytesseract
from threading import Thread
# Ścieżka do Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Funkcja OCR do rozpoznawania znaków na tablicy rejestracyjnej
def recognize_license_plate(plate_image):
    gray_plate = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    _, binary_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(binary_plate)
    return text.strip()

# Funkcja do detekcji i zapisywania tablic rejestracyjnych
def detect_and_save_license_plates(image_path, save_path="", scale_factor=2.7, image_id=""):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Couldn't load the image.")
        return
    else:
        print("Image loaded successfully.")
    
    # Powiększanie całego obrazu
    height, width = image.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    print("Image resized successfully.")
    
    cascade_path = "C:\\Users\\Adas\\OneDrive\\Desktop\\Projekt_Speed_detector\\plates_module\\haarcascade_russian_plate_number.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    
    if cascade.empty():
        print("Error: Couldn't load the cascade classifier.")
        return
    else:
        print("Cascade classifier loaded successfully.")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plates = cascade.detectMultiScale(gray, scaleFactor=1.25, minNeighbors=6, minSize=(10, 12))

    if len(plates) == 0:
        print("No license plates detected.")
        return
    else:
        print(f"Detected {len(plates)} license plates.")

    for index, (x, y, w, h) in enumerate(plates):
        license_plate = image[y:y+h, x:x+w]
        def save_file():
            save_filename = os.path.join(save_path, image_id)
            cv2.imwrite(save_filename, license_plate)   
            print(f"Saved detected license plate to {save_filename}")
        process_thread = Thread(target=save_file)
        process_thread.start()
        process_thread.join()    
        


image_path = "C:\\Users\\Adas\\OneDrive\\Desktop\\Projekt_Speed_detector\\screenshoots\\0.jpg"
save_path = "C:\\Users\\Adas\\OneDrive\\Desktop\\Projekt_Speed_detector\\detected_plates"
scale_factor = 2.7  # Możesz dostosować ten współczynnik w zależności od potrzeb
image_id = "1.jpg"  # Opcjonalnie, możesz ustawić unikalny identyfikator dla zapisanego obrazu

detect_and_save_license_plates(image_path, save_path, scale_factor, image_id)