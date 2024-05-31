import re
import cv2
import time
from easy_paddle_ocr import TextRecognizer
import os

text_recognizer = TextRecognizer()

# Wyrażenie regularne dopasowujące tylko wielkie litery i cyfry
pattern = re.compile(r'^[A-Z0-9]+$')

best_result = {
    "image": "",
    "filtered_text": "",
    "confidence": 0,
    "inference_time": 1
}


def process_image(image_id, iterations=1):
    # Zaktualizowana ścieżka, która teraz używa image_id
    file_path = f"C:\\Users\\Adas\\OneDrive\\Desktop\\Projekt_Speed_detector\\detected_plates\\{image_id}.jpg"
    
    if os.path.exists(file_path):
        for _ in range(iterations):
            image = cv2.imread(file_path)
            start = time.time()
            prediction = text_recognizer.read(image)
           # print(prediction["text"])
            return prediction["text"]

 
# Identyfikator obrazu
result=process_image(1)
print (result)
# Wyświetlenie najlepszego wyniku
#print(f'\n[+] Best Result:')
#print(f'[+] image: {best_result["image"]}')
#print(f'[+] filtered text: {best_result["filtered_text"]}')
#print(f'[+] confidence: {int(best_result["confidence"] * 100)}%')
#print(f'[+] inference time: {int(best_result["inference_time"])} milliseconds')
