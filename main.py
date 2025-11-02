import math
import random
import cv2
import numpy as np
import pyautogui  # trigger
import time
import chess
import chess.engine
from PIL import ImageGrab
import mss      # trigger
import threading
from collections import deque

import chess

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# if fuck i cannot do anything else so i wrote this ma fuck functinon instead of reading docs

def rotate_field(f):
    return [[f[7 - i][7 - j] for j in range(8)] for i in range(8)]

def rotate_sq(x):
    r, c = x % 8, x // 8
    r = 7 - r
    c = 7 - c
    return r + c * 8 

def change_fen_turn(fen_string, value):
    """
    Changes the active player in a FEN string.
    
    Args:
        fen_string (str): A valid FEN string
        
    Returns:
        str: FEN string with the active player changed
    """
    parts = fen_string.split(' ')
    
    if len(parts) < 2:
        raise ValueError("Invalid FEN string")
    
    # Change active color (second field, index 1)
    parts[1] = value
    
    return ' '.join(parts)

def create_board_from_index_array(index_array):
    """
    Create chess.Board from 2D array of indices based on the provided naming scheme
    """
    # Mapping from indices to piece symbols and empty squares
    index_to_piece = {
        0: '.',  # empty_white
        1: '.',  # empty_black
        2: 'P',  # white_pawn
        3: 'N',  # white_knight
        4: 'B',  # white_bishop
        5: 'R',  # white_rook
        6: 'Q',  # white_queen
        7: 'K',  # white_king
        8: 'p',  # black_pawn
        9: 'n',  # black_knight
        10: 'b', # black_bishop
        11: 'r', # black_rook
        12: 'q', # black_queen
        13: 'k'  # black_king
    }
    
    # Convert index array to FEN
    fen_rows = []
    
    for row in index_array:
        fen_row = ""
        empty_count = 0
        
        for index in row:
            piece_char = index_to_piece.get(index, '.')  # Default to empty if unknown index
            
            if piece_char == '.':
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += piece_char
        
        if empty_count > 0:
            fen_row += str(empty_count)
        
        fen_rows.append(fen_row)
    
    # Create FEN string (assuming white to move, no castling, no en passant)
    fen = "/".join(fen_rows) + " w - - 0 1"
    
    return chess.Board(fen)


class ChessBot:
    def __init__(self):
        self.board = chess.Board()
        self.engine = None
        self.is_running = False
        self.last_board_state = None
        self.move_history = deque(maxlen=10)
        self.screen_region = None
        self.piece_templates = self.load_piece_templates()
        
    def load_piece_templates(self):
        """Загрузка шаблонов шахматных фигур"""
        # В реальном проекте здесь должны быть изображения фигур
        pieces = {}
        piece_names = ['wp', 'wn', 'wb', 'wr', 'wq', 'wk',
                      'bp', 'bn', 'bb', 'br', 'bq', 'bk']
        
        # Заглушка - в реальном проекте загрузите PNG изображения фигур
        for piece in piece_names:
            pieces[piece] = None  # Здесь должны быть загруженные изображения
            
        return pieces
    
    def setup_engine(self, engine_path="fairy-stockfish"):
        """Настройка шахматного движка"""
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        except Exception as e:
            print(e)
            print("Stockfish не найден. Установите Stockfish и укажите правильный путь.")
            self.engine = None
    
    def calibrate_screen(self):
        """Калибровка области экрана с шахматной доской"""
        print("Калибровка шахматной доски...")
        print("Наведите курсор на левый верхний угол доски и нажмите Enter")
        input()
        top_left = pyautogui.position()
        
        print("Наведите курсор на правый нижний угол доски и нажмите Enter")
        input()
        bottom_right = pyautogui.position()
        
        self.screen_region = {
            'left': top_left.x,
            'top': top_left.y,
            'width': bottom_right.x - top_left.x,
            'height': bottom_right.y - top_left.y
        }
        
        print(f"Область доски: {self.screen_region}")
        with open('screen.txt', 'w') as f:
            f.write(str(self.screen_region))

        return self.screen_region
    
    def capture_board(self):
        """Захват изображения шахматной доски"""
        if not self.screen_region:
            return None
            
        with mss.mss() as sct:
            screenshot = sct.grab(self.screen_region)
            img = np.array(screenshot)
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    def preprocess_image(self, image):
        """Предобработка изображения для анализа"""
        # Конвертация в grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Применение Gaussian blur для уменьшения шума
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Бинаризация
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def detect_board_state(self, image, who):
        """Определение текущего состояния доски"""
        # Этот метод требует сложной реализации с компьютерным зрением
        # Упрощенная версия для демонстрации


        # Загрузка предобученной модели YOLOv8
        model = YOLO('best.pt')  # yolov8n.pt - самая легкая версия

        results = model.predict(image, verbose=False)

        # Визуализация результатов
        field = [[None for _ in range(8)] for _ in range(8)]
        for r in results:
            im_array = r.plot()  # изображение с bounding boxes
            
            centers = []
            mn = mx = [r.boxes.data[0][0], r.boxes.data[0][1]]
            mn = mx.copy()
            for i in r.boxes.data:
                mn[0] = min(mn[0], i[0])
                mn[1] = min(mn[1], i[1])
                mx[0] = max(mx[0], i[2])
                mx[1] = max(mx[1], i[3])
            for i in r.boxes.data:
                centers.append([float((i[0] + i[2]) / 2 - mn[0]) / (mx[0] - mn[0]), float((i[1] + i[3]) / 2 - mn[1]) / (mx[1] - mn[1])])
                for j in range(2):
                    centers[-1][j] = math.floor(centers[-1][j] * 8)
                if (field[centers[-1][1]][centers[-1][0]] is None):
                    field[centers[-1][1]][centers[-1][0]] = int(i[5])
            if len(r.boxes.data) < 64:
                raise ValueError('No field')
            cv2.imwrite('result_board.png', im_array)
        if who == 'b':
            field = rotate_field(field)
        return create_board_from_index_array(field)
    
    def find_changes(self, current_state, previous_state):
        """Поиск изменений на доске"""
        if not previous_state:
            return None
        
        # Сравнение FEN нотаций для определения хода
        if current_state.fen() != previous_state.fen():
            # Анализ различий между состояниями доски
            
            current_board = chess.Board(current_state.fen())
            
            previous_board = chess.Board(previous_state.fen())
            
            # Находим различающиеся ходы
            print(previous_board.legal_moves, current_board.fen())
            for move in previous_board.legal_moves:
                test_board = previous_board.copy()
                test_board.push(move)
                if str(test_board) == str(current_board):
                    return move
                
                
        return None
    
    def get_best_move(self, current_state, time_limit=1.0):
        """Получение лучшего хода от движка"""
        if not self.engine:
            return None
            
        try:
            result = self.engine.play(current_state, 
                                    chess.engine.Limit(time=time_limit))
            return result.move
        except:
            return None
    
    def execute_move(self, move, who):
        """Выполнение хода на экране"""
        if not move:
            return
            
        # Преобразование хода в координаты экрана
        from_square = move.from_square
        to_square = move.to_square
        

        if who == 'b':
            from_square = rotate_sq(from_square)
            to_square = rotate_sq(to_square)
        
        # Конвертация шахматных координат в пиксели
        from_x, from_y = self.square_to_pixels(from_square)
        to_x, to_y = self.square_to_pixels(to_square)
        
        # Выполнение drag-and-drop
        pyautogui.moveTo(from_x, from_y)
        pyautogui.mouseDown()
        time.sleep(0.1)
        pyautogui.moveTo(to_x, to_y)
        pyautogui.mouseUp()
        
        print(f"Выполнен ход: {move}")
    
    def square_to_pixels(self, square):
        """Конвертация шахматного поля в координаты экрана"""
        if not self.screen_region:
            return (0, 0)
            
        # Получаем координаты поля (0-7, 0-7)
        file_idx = square % 8
        rank_idx = 7 - (square // 8)  # Инвертируем rank для правильного отображения
        
        # Вычисляем пиксельные координаты
        square_width = self.screen_region['width'] / 8
        square_height = self.screen_region['height'] / 8
        
        x = self.screen_region['left'] + (file_idx * square_width) + (square_width / 2)
        y = self.screen_region['top'] + (rank_idx * square_height) + (square_height / 2)
        
        return (int(x), int(y))
    
    def monitor_opponent_move(self, who):
        """Мониторинг ходов противника"""
        check_interval = 0.5  # Проверка каждые 0.5 секунд
        delay_before_move = [0, 0]
        
        while self.is_running:
            try:
                # Захват текущего состояния доски
                current_image = self.capture_board()
                
                if current_image is None:
                    time.sleep(check_interval)
                    continue
                
                current_state = self.detect_board_state(current_image, who)

                if self.last_board_state is None:
                    self.last_board_state = current_state
                if who == 'b':
                    current_state = chess.Board(change_fen_turn(current_state.fen(), 'b'))
                    self.last_board_state = chess.Board(change_fen_turn(self.last_board_state.fen(), 'w'))
                else:
                    current_state = chess.Board(change_fen_turn(current_state.fen(), 'w'))
                    self.last_board_state = chess.Board(change_fen_turn(self.last_board_state.fen(), 'b'))
               
                # print(current_state)
                # print(f"\nFEN: {current_state.fen()}")


                # Поиск изменений
                opponent_move = self.find_changes(current_state, self.last_board_state)
                if self.last_board_state and current_state.fen() != self.last_board_state.fen():
                    print(current_state)
                    print()
                    print(self.last_board_state)
                if opponent_move and opponent_move != self.move_history[-1] if self.move_history else True:
                    print(f"Обнаружен ход противника: {opponent_move}")
                    
                    # Обновление доски
                    self.move_history.append(opponent_move)
                    
                    # Ответный ход
                    if not current_state.is_game_over():
                        time.sleep(random.uniform(*delay_before_move))
                        self.make_our_move(current_state, who)
                
                self.last_board_state = current_state
                time.sleep(check_interval)
                
            except Exception as e:
                print(f"Ошибка в мониторинге: {e}")
                time.sleep(check_interval)
    
    def make_our_move(self, current_state, who):
        """Выполнение нашего хода"""
        print("Поиск лучшего хода...")
        best_move = self.get_best_move(current_state)
        
        if best_move and best_move in current_state.legal_moves:
            self.execute_move(best_move, who)
            current_state.push(best_move)
            self.move_history.append(best_move)
            print(f"Сделан ход: {best_move}")
        else:
            print("Не удалось найти допустимый ход")
    
    def calibrate(self):
        self.calibrate_screen()

    def start(self, who):
        """Запуск бота"""
        print("Запуск шахматного бота...")
        
        # Калибровка
        with open('screen.txt') as f:
            self.screen_region = eval(f.read())
        
        # Настройка движка
        self.setup_engine()
        
        if not self.engine:
            print("Ошибка: шахматный движок не доступен")
            return
        
        self.is_running = True
        
        # Запуск мониторинга в отдельном потоке
        monitor_thread = threading.Thread(target=self.monitor_opponent_move, args=(who,))
        monitor_thread.daemon = True
        monitor_thread.start()
        
        print("Бот запущен. Нажмите Ctrl+C для остановки.")
        
        try:
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Остановка бота"""
        print("Остановка бота...")
        self.is_running = False
        if self.engine:
            self.engine.quit()

def main():
    bot = ChessBot()
    
    print("Шахматный бот")
    print("1. Запустить бота")
    print("2. Калибровка")
    print("3. Выход")
    
    # choice = input("Выберите опцию: ")
    choice = "1"
    who_moves = 'w' if input('who moves') == 'w' else 'b'
    
    if choice == "1":
        bot.start(who_moves)
    elif choice == "2":
        bot.calibrate()
    else:
        print("Выход")

if __name__ == "__main__":
    main()