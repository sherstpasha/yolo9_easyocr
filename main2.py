# Импортируем необходимые функции и библиотеки
from tools import extract_text_from_imageEasyOcrTrOCR, extract_text_from_imageTrOCR
import cv2

# Задаем путь к изображению
image_path = r"C:\Users\user\Desktop\done\100картинок добавить\100image\images\-5_jpg.rf.37baaf6b19eae3f8eb596f613d1ae5b9.jpg"

# Задаем параметры модели YOLO и OCR
yolo_weights_path = r"C:\Users\user\Desktop\yolo9models\best - Copy.pt"
craft_models_directory = "C:/Users/user/Desktop/modelsocr"
TrOCR_directory = r"C:\Users\user\TrOCR\trocr-base-ru"

# Загружаем изображение как объект numpy.ndarray
image = cv2.imread(image_path)

# Извлекаем текст из изображения, используя функции YOLO и EasyOCR
text_results = extract_text_from_imageEasyOcrTrOCR(
    image_or_path=image_path,            # Путь к изображению
    TrOCR_directory=TrOCR_directory,  
    craft_gpu="cpu",                   # Устройство для выполнения YOLO (CPU)
    ocr_gpu=False                        # Использовать ли GPU для OCR (False)
)

# Печатаем результаты распознавания текста
print("EasyOCR", text_results)

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
