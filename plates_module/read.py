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
    "inference_time": 0
}

def process_image(filename, iterations=5):
    if os.path.exists(filename):
        for _ in range(iterations):
            image = cv2.imread(filename)
            start = time.time()
            prediction = text_recognizer.read(image)
            recognized_text = prediction["text"]

            # Filtrowanie wyników przy użyciu wyrażenia regularnego
            filtered_text = re.match(pattern, recognized_text)

            if filtered_text and prediction["confidence"] > best_result["confidence"]:
                best_result["image"] = filename
                best_result["filtered_text"] = filtered_text.group(0)
                best_result["confidence"] = prediction["confidence"]
                best_result["inference_time"] = (time.time() - start) * 1000

    else:
        print(f'File {filename} does not exist.')

# Ścieżka do pliku
file_path = "C:\\Users\\Adas\\OneDrive\\Desktop\\Projekt_Speed_detector\\tablice_rej\\detected_plates\\detected_plate_0.jpg"

# Przetwarzanie obrazu z 5 iteracjami
process_image(file_path, iterations=1)

# Wyświetlenie najlepszego wyniku
print(f'\n[+] Best Result:')
print(f'[+] image: {best_result["image"]}')
print(f'[+] filtered text: {best_result["filtered_text"]}')
print(f'[+] confidence: {int(best_result["confidence"] * 100)}%')
print(f'[+] inference time: {int(best_result["inference_time"])} milliseconds')
