import os
import cv2
import numpy as np
import random
from sklearn.model_selection import train_test_split


class ChessboardDatasetGenerator:

    classes = [
        'empty_white', 'empty_black',
        'white_pawn', 'white_knight', 'white_bishop', 'white_rook', 'white_queen', 'white_king',
        'black_pawn', 'black_knight', 'black_bishop', 'black_rook', 'black_queen', 'black_king'
    ]
    
    def __init__(self, output_dir="dataset", train_ratio=0.8):
        self.output_dir = output_dir
        self.train_ratio = train_ratio

        # Создаем структуру папок
        self.create_directory_structure()

    def create_directory_structure(self):
        """Создает структуру папок для датасета"""
        directories = [
            f"{self.output_dir}/images/train",
            f"{self.output_dir}/images/val",
            f"{self.output_dir}/labels/train",
            f"{self.output_dir}/labels/val"
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def detect_chessboard_cells(self, image_path):
        """
        Обнаруживает шахматную доску и разбивает на клетки
        Возвращает список клеток и их координаты
        """
        # Загружаем изображение
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        cells = []
        width = image.shape[0] // 8
        height = image.shape[1] // 8
        for i in range(8):
            for j in range(8):
                border = 10

                cells.append({'image': image[j * height + border:  (j + 1) * height - border,
                                             i * width + border:(i + 1) * width - border],
                              'position': (j, i),
                              'bbox': (i * width + border,
                                       j * height + border,
                                       width - border * 2,
                                       height - border * 2,)})

        return cells

    def classify_cell(self, position):
        """
        Классифицирует клетку для начальной расстановки фигур
        position: (row, col) где row=0..7, col=0..7
        """
        row, col = position
        
        # Начальная расстановка фигур
        if row == 0:  # Черные фигуры (первый ряд)
            if col == 0 or col == 7:
                return self.classes.index('black_rook')    # Ладьи
            elif col == 1 or col == 6:
                return self.classes.index('black_knight')  # Кони
            elif col == 2 or col == 5:
                return self.classes.index('black_bishop')  # Слоны
            elif col == 3:
                return self.classes.index('black_queen')   # Ферзь
            elif col == 4:
                return self.classes.index('black_king')    # Король
        
        elif row == 1:  # Черные пешки
            return self.classes.index('black_pawn')
        
        elif row == 6:  # Белые пешки
            return self.classes.index('white_pawn')
        
        elif row == 7:  # Белые фигуры (последний ряд)
            if col == 0 or col == 7:
                return self.classes.index('white_rook')    # Ладьи
            elif col == 1 or col == 6:
                return self.classes.index('white_knight')  # Кони
            elif col == 2 or col == 5:
                return self.classes.index('white_bishop')  # Слоны
            elif col == 3:
                return self.classes.index('white_queen')   # Ферзь
            elif col == 4:
                return self.classes.index('white_king')    # Король
        
        else:  # Пустые клетки (ряды 2,3,4,5)
            return self.classes.index('empty_black' if (col + row) % 2 else 'empty_white')

    def generate_yolo_annotation(self, annotations,  image_size):
        """Генерирует аннотацию в формате YOLO"""
        result = ""
        for annotation in annotations:
            class_id, bbox = annotation

            x_center = (bbox[0] + bbox[2] / 2) / image_size[0]
            y_center = (bbox[1] + bbox[3] / 2) / image_size[1]
            width = bbox[2] / image_size[0]
            height = bbox[3] / image_size[1]
            result += f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
        return result
    
    def process_chessboard_image(self, image_path, output_prefix, classes1):
        """
        Обрабатывает одно изображение шахматной доски
        и генерирует датасет из отдельных клеток
        """
        try:
            # Обнаруживаем и извлекаем клетки
            cells = self.detect_chessboard_cells(image_path)

            # Оригинальное изображение для справки
            original_image = cv2.imread(image_path)
            image_height, image_width = original_image.shape[:2]

            results = []
            annotation_list = []
            bbox = []
            classes = []
            images = []
            positions = []


            for idx, cell in enumerate(cells):
                # Классифицируем клетку
                # class_id = self.classify_cell(cell['position'])
                class_id = classes1[idx]

                # Сохраняем изображение клетки
                cell_filename = f"{output_prefix}_cell_{idx:02d}.png"
                cv2.imwrite(cell_filename, cell['image'])

                # Создаем YOLO аннотацию
                annotation_list.append((class_id, cell['bbox']))
                bbox.append(cell['bbox'])
                classes.append(class_id)
                images.append(cell['image'])
                positions.append(cell['position'])

            annotation = self.generate_yolo_annotation(
                annotation_list,
                (image_width, image_height)
            )

            results.append({
                'image_path': image_path,
                'annotation': annotation,
                'bbox': bbox,
                'classes': classes,
                'images': images,
                'positions': positions
            })

            return results

        except Exception as e:
            print(f"Ошибка при обработке {image_path}: {e}")
            return []

    def generate_dataset_from_images(self, image_path):
        """Генерирует полный датасет из списка изображений"""
        all_samples = []

        print(
            f"Обработка изображения {1}/{1}: {image_path}")

        output_prefix = f"temp_cell_{1:03d}"
        samples = self.process_chessboard_image(image_path, output_prefix,
                                                [self.classify_cell((idx % 8, idx // 8)) for idx in range(64)])
        all_samples.extend(samples)
        dataset_size = 128

        for i in range(dataset_size):
        
            classes1 = create_synthetic_chessboard(all_samples, image_path, f"chessboard{i+1}.png")
            samples = self.process_chessboard_image(f"chessboard{i+1}.png", output_prefix,
                                                classes1)
            all_samples.extend(samples)

        # Разделяем на train/val
        train_samples, val_samples = train_test_split(
            all_samples, train_size=self.train_ratio, random_state=0
        )
        # Сохраняем датасет
        self.save_dataset(train_samples, val_samples)

        # Очищаем временные файлы
        self.cleanup_temp_files()

        print(f"Датасет создан!")
        print(f"Train samples: {len(train_samples)}")
        print(f"Val samples: {len(val_samples)}")
        print(f"Total samples: {len(all_samples)}")

    def save_dataset(self, train_samples, val_samples):
        """Сохраняет датасет в нужной структуре"""

        # Сохраняем train samples
        for i, sample in enumerate(train_samples):
            # Копируем изображение
            new_image_path = f"{self.output_dir}/images/train/image_{i:06d}.png"
            os.rename(sample['image_path'], new_image_path)

            # Сохраняем аннотацию
            label_path = f"{self.output_dir}/labels/train/image_{i:06d}.txt"
            with open(label_path, 'w') as f:
                f.write(sample['annotation'])

        # Сохраняем val samples
        for i, sample in enumerate(val_samples):
            new_image_path = f"{self.output_dir}/images/val/image_{i+len(train_samples):06d}.png"
            os.rename(sample['image_path'], new_image_path)

            label_path = f"{self.output_dir}/labels/val/image_{i+len(train_samples):06d}.txt"
            with open(label_path, 'w') as f:
                f.write(sample['annotation'])

    def cleanup_temp_files(self):
        """Очищает временные файлы"""
        for file in os.listdir('.'):
            if file.startswith('temp_cell_') and file.endswith('.png'):
                os.remove(file)

# Пример использования

def create_synthetic_chessboard(cell_images, image_path, output_path, add_pieces=True):
    """Создает синтетическое изображение шахматной доски для тестирования"""
    image = cv2.imread(image_path)

    width = image.shape[0] // 8
    height = image.shape[1] // 8

    def paste(i, j, pst):
        border = 10
        
        image[i * height + border:  (i + 1) * height - border,
                                j * width + border:(j + 1) * width - border] = pst
    classes1 = [0 for _ in range(64)]

    black_fileds = [None for i in range(len(ChessboardDatasetGenerator.classes))]
    white_fileds = [None for i in range(len(ChessboardDatasetGenerator.classes))]
    for i, cell in enumerate(cell_images[0]['images']):
        idx = cell_images[0]['classes'][i]
        r, c = cell_images[0]['positions'][i]
        if (r + c) % 2:
            black_fileds[idx] = cell
        else:
            white_fileds[idx] = cell
    
    for i in range(8):
        for j in range(8):
            if (i + j) % 2:
                cell = black_fileds[1]
                classes1[i + j * 8] = 1
            else:
                cell = white_fileds[0]
                classes1[i + j * 8] = 0
            paste(i, j, cell)

    for i in range(random.randint(1, 32)):
        r, c = random.randint(0, 7),  random.randint(0, 7)
        cell = None
        while cell is None:
            if (r + c) % 2:
                idx = random.randint(0, len(black_fileds) - 1)
                cell = black_fileds[idx]
                classes1[r + c * 8] = idx
            else:
                idx = random.randint(0, len(black_fileds) - 1)
                cell = white_fileds[idx]
                classes1[r + c * 8] = idx
        paste(r, c, cell)


    cv2.imwrite(output_path, image)
    print(f"Создано синтетическое изображение: {output_path}")
    return classes1

def main():
    # Создаем генератор датасета
    os.system('cp 11.png 1.png')

    generator = ChessboardDatasetGenerator("chess_dataset")

    # Список путей к вашим изображениям шахматных досок
    image_path = "1.png"

    # Генерируем датасет
    generator.generate_dataset_from_images(image_path)

    # create_synthetic_chessboard('board.png', add_pieces=True)



if __name__ == "__main__":
    main()
