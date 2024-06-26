import cv2
import numpy as np
from yolo9.detect_function import detect_image
from easyocr.recognize_function import recognize_text_from_images
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, GenerationConfig
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from easyocr.easyocr import Reader
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull


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


def recognize_text_from_imagesTrOCR(image_pieces, models_directory, gpu=False):
    """
    Recognizes text from a list of image pieces using TrOCR.

    Parameters:
    - image_pieces (list): List of image pieces as PIL Image objects.
    - models_directory (str): Path to the directory containing the pre-trained TrOCR models.
    - gpu (bool): Whether to use GPU for OCR (default is False).

    Returns:
    - List of recognized texts.
    """
    device = torch.device('cuda' if gpu else 'cpu')

    processor = TrOCRProcessor.from_pretrained(models_directory)
    model = VisionEncoderDecoderModel.from_pretrained(models_directory)
    model.to(device)
    # Initialize EasyOCR readers

    recognized_texts = []
    for image_piece in image_pieces:
        # Convert PIL Image to OpenCV format
        image_cv = cv2.cvtColor(np.array(image_piece), cv2.COLOR_RGB2BGR)
        # Perform text recognition
        pixel_values = processor(images=image_cv, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        recognized_texts.append(generated_text)
    
    return recognized_texts

def extract_text_from_imageTrOCR(image_or_path, yolo_weights_path, TrOCR_directory, yolo_device="cpu", ocr_gpu=False):
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
    recognized_texts = recognize_text_from_imagesTrOCR(image_pieces=cropped_images, models_directory=TrOCR_directory, gpu=ocr_gpu)

    # Формируем результат в формате [([x1, y1, x2, y2], "string"), ...]
    results = [([box['bbox'][0], box['bbox'][1], box['bbox'][2], box['bbox'][3]], text) for box, text in zip(boxes, recognized_texts)]

    return results


import os
import pandas as pd
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, GenerationConfig, DataCollatorForSeq2Seq
from transformers import default_data_collator
import torch
from torch.utils.data import Dataset
from accelerate import Accelerator, DataLoaderConfiguration

class OCRDataset(Dataset):
    def __init__(self, csv_file, root_dir, processor):
        self.annotations = pd.read_csv(csv_file, sep='^([^,]+),', engine='python', usecols=['filename', 'words'], keep_default_na=False)
        self.root_dir = root_dir
        self.processor = processor

        # Проверка наличия изображений в папке и удаление отсутствующих
        self.annotations['exists'] = self.annotations['filename'].apply(lambda x: os.path.exists(os.path.join(root_dir, x)))
        self.annotations = self.annotations[self.annotations['exists']].drop(columns=['exists'])
        
        print(self.annotations)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        text = self.annotations.iloc[idx, 1]
        
        # Preprocess the image and text
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze()
        labels = self.processor.tokenizer(text, return_tensors="pt").input_ids.squeeze()

        return {"pixel_values": pixel_values, "labels": labels}

def custom_collator(features):
    pixel_values = torch.stack([f["pixel_values"] for f in features])
    labels = pad_sequence([f["labels"] for f in features], batch_first=True, padding_value=-100)
    return {"pixel_values": pixel_values, "labels": labels}

def train_trocr(train_csv, train_root, val_csv, val_root, model_name, output_dir, epochs=3, batch_size=8, learning_rate=5e-5, gpu=False):
    # Initialize the processor and model
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Set decoder_start_token_id and bos_token_id
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.bos_token_id = processor.tokenizer.cls_token_id

    # Prepare the datasets
    train_dataset = OCRDataset(csv_file=train_csv, root_dir=train_root, processor=processor)
    val_dataset = OCRDataset(csv_file=val_csv, root_dir=val_root, processor=processor)

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",  # Evaluate every `eval_steps`
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        fp16=gpu,
        output_dir=output_dir,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=100,  # Log every 100 steps
        save_steps=100,  # Save model every 500 steps
        eval_steps=100,  # Evaluate every 500 steps
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        save_total_limit=1,  # Only keep the best model
        load_best_model_at_end=True,  # Load the best model at the end of training
        metric_for_best_model="eval_loss",  # Use evaluation loss to determine the best model
        greater_is_better=False  # Lower eval_loss is better
    )

    # Define the trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=custom_collator,
        tokenizer=processor.tokenizer,
    )

    # Train the model
    trainer.train()

    # Save the trained model and processor
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

    # Create and save the GenerationConfig object
    generation_config = GenerationConfig(max_length=64)
    generation_config.save_pretrained(output_dir)






import os


def convert_boxes_to_points(boxes):
    """
    Преобразует боксы из формата [x_min, x_max, y_min, y_max] в формат [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].

    Parameters:
    boxes (list): Список боксов, где каждый бокс представлен координатами [x_min, x_max, y_min, y_max].

    Returns:
    list: Преобразованный список боксов в формате [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
    """
    formatted_boxes = []
    for box in boxes:
        x_min, x_max, y_min, y_max = [max(0, coord) for coord in box]
        formatted_box = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
        formatted_boxes.append(formatted_box)
    return formatted_boxes

def group_boxes_into_lines(boxes):

    def get_x_coordinates(box):
        x_coords = [point[0] for point in box]
        min_x = min(x_coords)
        max_x = max(x_coords)
        mean_x = sum(x_coords) / len(x_coords)
        return min_x, max_x, mean_x

    # Группировка боксов в строки
    points = np.array([get_x_coordinates(box) for box in boxes])
    min_x_points, max_x_points, mean_x_points = points[:, 0], points[:, 1], points[:, 2]

    cond = mean_x_points[:-1] > min_x_points[1:]
    cond = np.hstack([[True], cond])

    grouped_lines = []
    current_line = []

    for box, is_new_line in zip(boxes, cond):
        if is_new_line:
            if current_line:
                grouped_lines.append(current_line)
            current_line = [box]
        else:
            current_line.append(box)

    if current_line:
        grouped_lines.append(current_line)

    combined_polygons = []

    for line in grouped_lines:
        if not line:
            continue

        points = []
        for box in line:
            points.extend(box)

        points = np.array(points)
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]

        combined_polygons.append(hull_points.tolist())

    return grouped_lines, combined_polygons


def crop_polygon(image_np, polygon):
    """
    Вырезает часть изображения по полигону с белым фоном.

    Parameters:
    image_np (numpy.ndarray): Входное изображение в формате numpy array.
    polygon (list): Полигон, представленный списком точек [[x1, y1], [x2, y2], ...].

    Returns:
    numpy.ndarray: Вырезанная часть изображения.
    tuple: Bounding box координаты (x_min, y_min, x_max, y_max).
    """
    # Преобразуем координаты полигона в целые числа
    polygon = [(int(x), int(y)) for x, y in polygon]

    # Создаем маску
    mask = Image.new('L', (image_np.shape[1], image_np.shape[0]), 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(polygon, outline=1, fill=1)
    mask = np.array(mask)

    # Создаем белый фон
    white_background = np.ones_like(image_np) * 255

    # Применяем маску к изображению
    masked_image_np = np.where(mask[..., None], image_np, white_background)

    # Обрезаем по минимальной и максимальной границам полигона
    x_min, y_min = np.min(polygon, axis=0)
    x_max, y_max = np.max(polygon, axis=0)
    bbox = (x_min, y_min, x_max, y_max)
    cropped_image_np = masked_image_np[y_min:y_max, x_min:x_max]

    return cropped_image_np, bbox

def detect_image_craft(image, ocr_gpu=False, return_only_lines=True):

    # Инициализация EasyOCR Reader
    reader = Reader(['ru'],
                    gpu=ocr_gpu, recognizer=None)

    # Определение текста с ограничивающими рамками
    boxes, _ = reader.detect(image, slope_ths=1., reformat = False)

    # Преобразование боксов в нужный формат
    formatted_boxes = convert_boxes_to_points(boxes[0])

    # Группируем боксы в строки и создаем полигоны
    grouped_lines, combined_polygons = group_boxes_into_lines(formatted_boxes)

    if return_only_lines:
        return combined_polygons
    else:
        return formatted_boxes, grouped_lines, combined_polygons


def extract_text_from_imageEasyOcrTrOCR(image_or_path, TrOCR_directory, craft_gpu=False, ocr_gpu=False):
    
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
    polygons = detect_image_craft(image=image,
                                  ocr_gpu=craft_gpu,
                                  return_only_lines=True)

    # Вырезаем bounding boxes из изображения
    cropped_images, boxes = zip(*tuple([crop_polygon(image, polygon) for polygon in polygons]))

    # Создаем папку output, если ее нет
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Сохраняем каждое обрезанное изображение
    for idx, cropped_image in enumerate(cropped_images):
        if cropped_image.shape[1] > 0:
            cropped_image_pil = Image.fromarray(cropped_image)
            cropped_image_pil.save(os.path.join(output_dir, f"cropped_image_{idx+1}.jpg"))

    # Распознаем текст из вырезанных изображений
    recognized_texts = recognize_text_from_imagesTrOCR(image_pieces=cropped_images, models_directory=TrOCR_directory, gpu=ocr_gpu)

    # # Формируем результат в формате [([x1, y1, x2, y2], "string"), ...]
    results = [([box[0], box[1], box[2], box[3]], text) for box, text in zip(boxes, recognized_texts)]

    return results


def extract_text_from_imageEasyOcrOnly(image_or_path, ocr_models_directory, recog_network="best_accuracy", ocr_gpu=False):
    """
    Extracts text from an image using EasyOCR for text recognition.

    Parameters:
    image_or_path (str or np.ndarray): Path to the image or the image itself as a numpy.ndarray.
    ocr_models_directory (str): Directory containing OCR models.
                                Example structure:
                                ocr_models_directory/model/best_accuracy.pth
                                ocr_models_directory/user_network/best_accuracy.py
                                ocr_models_directory/user_network/best_accuracy.yaml
    recog_network (str): Name of the OCR network to use (default is "best_accuracy").
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
    polygons = detect_image_craft(image=image, ocr_gpu=ocr_gpu, return_only_lines=True)

    # Вырезаем bounding boxes из изображения
    cropped_images, boxes = zip(*tuple([crop_polygon(image, polygon) for polygon in polygons]))

    # Распознаем текст из вырезанных изображений
    recognized_texts = recognize_text_from_images(image_pieces=cropped_images, models_directory=ocr_models_directory, recog_network=recog_network, gpu=ocr_gpu)

    # Формируем результат в формате [([x1, y1, x2, y2], "string"), ...]
    results = [([box['bbox'][0], box['bbox'][1], box['bbox'][2], box['bbox'][3]], text) for box, text in zip(boxes, recognized_texts)]

    return results