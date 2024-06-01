# Импортируем необходимые функции и библиотеки
from tools import extract_text_from_image, extract_text_from_imageTrOCR
import cv2

# Задаем путь к изображению
image_path = r"C:\Users\user\Desktop\photo_2024-05-27_15-25-05 (2).jpg"

# Задаем параметры модели YOLO и OCR
yolo_weights_path = r"C:\Users\user\Desktop\yolo9models\best - Copy.pt"
ocr_models_directory = "C:/Users/user/Desktop/modelsocr"

# Загружаем изображение как объект numpy.ndarray
image = cv2.imread(image_path)

# Извлекаем текст из изображения, используя функции YOLO и EasyOCR
text_results = extract_text_from_image(
    image_or_path=image_path,            # Путь к изображению
    yolo_weights_path=yolo_weights_path, # Путь к весам модели YOLO
    ocr_models_directory=ocr_models_directory, # Директория с моделями OCR
    recog_network="None-BiLSTM_Annt",    # Сеть распознавания OCR
    yolo_device="cpu",                   # Устройство для выполнения YOLO (CPU)
    ocr_gpu=False                        # Использовать ли GPU для OCR (False)
)

# Печатаем результаты распознавания текста
print("EasyOCR", text_results)

TrOCR_directory = r"C:\Users\user\TrOCR\trocr-base-ru"

# Загружаем изображение как объект numpy.ndarray
image = cv2.imread(image_path)

# Извлекаем текст из изображения, используя функции YOLO и TrOCR
text_results = extract_text_from_imageTrOCR(
    image_or_path=image_path,            # Путь к изображению
    yolo_weights_path=yolo_weights_path, # Путь к весам модели YOLO
    TrOCR_directory=TrOCR_directory,     # Директория с моделями TrOCR
    yolo_device="cpu",                   # Устройство для выполнения YOLO (CPU)
    ocr_gpu=False                        # Использовать ли GPU для OCR (False)
)

# Печатаем результаты распознавания текста
print("TrOCR", text_results)
