# Импортируем необходимые функции и библиотеки
from tools import extract_text_from_image
import cv2

# Задаем путь к изображению
image_path = r"C:\Users\user\Desktop\test_images\test2 (1) — копия.jpg"

# Задаем параметры модели YOLO и OCR
yolo_weights_path = r"C:\Users\user\Desktop\ddata\exp3\weights\best.pt"
ocr_models_directory = "C:/Users/user/Desktop/modelsocr"

# Загружаем изображение как объект numpy.ndarray
image = cv2.imread(image_path)

# Извлекаем текст из изображения, используя функции YOLO и EasyOCR
text_results = extract_text_from_image(
    image_or_path=image_path,          # Путь к изображению
    yolo_weights_path=yolo_weights_path, # Путь к весам модели YOLO
    ocr_models_directory=ocr_models_directory, # Директория с моделями OCR
    recog_network="best_accuracy",     # Сеть распознавания OCR
    yolo_device="cpu",                 # Устройство для выполнения YOLO (CPU)
    ocr_gpu=False                      # Использовать ли GPU для OCR (False)
)

# Печатаем результаты распознавания текста
print(text_results)