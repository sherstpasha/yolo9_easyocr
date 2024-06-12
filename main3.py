# Импортируем необходимые функции и библиотеки
from tools import extract_text_from_imageEasyOcrOnly
import cv2

# Задаем путь к изображению
image_path = r"C:\Users\user\Desktop\done\100картинок добавить\100image\images\-5_jpg.rf.37baaf6b19eae3f8eb596f613d1ae5b9.jpg"

ocr_models_directory = "C:/Users/user/Desktop/modelsocr"

# Загружаем изображение как объект numpy.ndarray
image = cv2.imread(image_path)

# Извлекаем текст из изображения, используя функции YOLO и EasyOCR
text_results = extract_text_from_imageEasyOcrOnly(
    image_or_path=image_path,            # Путь к изображению
    ocr_models_directory=ocr_models_directory, # Директория с моделями OCR
    recog_network="None-BiLSTM_Annt",    # Сеть распознавания OCR
    ocr_gpu=False                        # Использовать ли GPU для OCR (False)
)

# Печатаем результаты распознавания текста
print("EasyOCR", text_results)