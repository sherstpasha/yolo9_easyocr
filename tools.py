import cv2
import numpy as np
from yolo9.detect_function import detect_image
from easyocr.recognize_function import recognize_text_from_images

def crop_box_from_image(image, box):
    if isinstance(image, np.ndarray):
        xmin, ymin, xmax, ymax = box['bbox']
        cropped_img = image[ymin:ymax, xmin:xmax]
        return cropped_img
    else:
        raise ValueError("image должен быть объектом numpy.ndarray")
    

def extract_text_from_image(image_or_path, yolo_weights_path, ocr_models_directory, recog_network="best_accuracy", yolo_device="cpu", ocr_gpu=False):
    """
    Extracts text from an image using YOLO for object detection and EasyOCR for text recognition.

    Parameters:
    image_or_path (str or np.ndarray): Path to the image or the image itself as a numpy.ndarray.
    yolo_weights_path (str): Path to the YOLO weights file.
    ocr_models_directory (str): Directory containing OCR models.
                                Example structure:
                                ocr_models_directory/model/best_accuracy.pth
                                ocr_models_directory/user_network/best_accuracy.py
                                ocr_models_directory/user_network/best_accuracy.yaml
    recog_network (str): Name of the OCR network to use (default is "best_accuracy").
    yolo_device (str): Device to run YOLO on, either "cpu" or the GPU index (default is "cpu").
    ocr_gpu (bool): Whether to use GPU for OCR (default is False).

    Returns:
    list: A list of tuples where each tuple contains bounding box coordinates and the recognized text.
          Format: [([x1, y1, x2, y2], "string"), ...]
    """
    
    # Проверяем, является ли входное изображение путем или numpy.ndarray
    if isinstance(image_or_path, str):
        image = cv2.imread(image_or_path)
        if image is None:
            raise ValueError("Невозможно загрузить изображение по указанному пути.")
    elif isinstance(image_or_path, np.ndarray):
        image = image_or_path
    else:
        raise ValueError("image_or_path должен быть либо строкой пути, либо объектом numpy.ndarray")

    # Обнаруживаем bounding boxes
    boxes = detect_image(weights=yolo_weights_path, source=image, sort_boxes=True, device=yolo_device)

    # Вырезаем bounding boxes из изображения
    cropped_images = [crop_box_from_image(image, box) for box in boxes]

    # Распознаем текст из вырезанных изображений
    recognized_texts = recognize_text_from_images(image_pieces=cropped_images, models_directory=ocr_models_directory, recog_network=recog_network, gpu=ocr_gpu)

    # Формируем результат в формате [([x1, y1, x2, y2], "string"), ...]
    results = [([box['bbox'][0], box['bbox'][1], box['bbox'][2], box['bbox'][3]], text) for box, text in zip(boxes, recognized_texts)]

    return results