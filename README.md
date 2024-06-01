```python
# Импортируем необходимые функции и библиотеки
from tools import extract_text_from_image
import cv2

# Задаем путь к изображению
image_path = "path/to/your/image.jpg"

# Задаем параметры модели YOLO и OCR
yolo_weights_path = "path/to/yolo/weights/best.pt"
ocr_models_directory = "path/to/ocr/models"

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
```

### Ожидаемый вывод

Пример вывода результатов распознавания текста:
```
[([944, 519, 2647, 633], 'присковъ Сьверной системы'), 
 ([941, 729, 2649, 831], 'какъ болье отдаленной оъ'), 
 ([1005, 928, 2637, 1029], 'ллыххъ мiъ стъ строены дв'), 
 ([936, 1120, 2688, 1216], 'больницы для призре иiя боль'), 
 ([937, 1318, 2698, 1419], 'ныхъ пабочихъ присльдованiи'), 
 ([932, 1513, 2652, 1608], 'ихъ наприски иобратно'), 
 ([950, 1694, 2583, 1791], 'Больницы эти также нах'), 
 ([921, 1895, 2687, 1987], 'дятся еъ удовлетворитеаьномъъ'), 
 ([915, 1939, 1005, 2007], ''), 
 ([972, 2251, 2672, 2391], 'Кро- м того на золотыхъ про'), 
 ([926, 2491, 2672, 2573], 'мыслахъ О ихъ системъ Ени'), 
 ([1449, 2680, 2621, 2778], 'округа содержатся'), 
 ([916, 2880, 2620, 2962], 'на счетъ золотопромыилн'), 
 ([917, 3073, 2667, 3165], 'никовъ отредь ленные отъ ПГра'), 
 ([908, 3264, 2642, 3349], 'вительства повивальные бабки.'), 
 ([951, 3440, 2583, 3541], 'По значительному простран'), 
 ([909, 3637, 2611, 3728], 'ству золоть иъ поомысловъ')]
```

### Примечание

- `ocr_models_directory` должен содержать модели OCR в следующей структуре:
  ```
  ocr_models_directory/model/best_accuracy.pth
  ocr_models_directory/user_network/best_accuracy.py
  ocr_models_directory/user_network/best_accuracy.yaml
  ```

  Пример структуры папки:
  ```
  path/to/ocr/models/
  ├── model/
  │   └── best_accuracy.pth
  ├── user_network/
  │   ├── best_accuracy.py
  │   └── best_accuracy.yaml
  ```

  - `best_accuracy.pth`: модель весов для OCR.
  - `best_accuracy.py`: скрипт настройки модели.
  - `best_accuracy.yaml`: файл конфигурации модели.

  Пример структуры можно найти в папке `models_directory_example`.

- Модель для детекции строк YOLO9 можно скачать по следующей ссылке:
  [YOLO9 модель для детекции строк](https://drive.google.com/file/d/1I56uvn7kAMZIh7AZfDIIwjAysJj_sdu7/view?usp=sharing)

- Модели для распознавания текста OCR можно скачать по следующей ссылке:
  [Модели для распознавания текста OCR](https://drive.google.com/drive/folders/1c8U6gk4qBdjvvi1zU8UEQc_gmS0B3NUK?usp=sharing)

- Модель для распознавания текста TrOCR можно скачать по следующей ссылке:
  [Модели для распознавания текста OCR](https://drive.google.com/drive/folders/1OL-AxBqs0_hiC61hYc8dCwEEW3E8HuuJ?usp=sharing)

