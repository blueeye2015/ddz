#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
欢乐斗地主 - 固定坐标版
基于实际窗口测量，放弃自动检测，直接配置
"""

import cv2
import numpy as np
import time
import threading
import ctypes
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import json
import os

try:
    import win32gui
    import win32con
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False

try:
    from PIL import ImageGrab
except ImportError:
    pass

try:
    import mss
except ImportError:
    pass

# 两阶段 YOLO 识别器（替换旧模板匹配）
try:
    from ddz_yolo_recognizer import TwoStageRecognizerV3
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


# ==================== 配置 ====================

# 你的实际窗口尺寸（从截图测量）
WINDOW_WIDTH = 2071
WINDOW_HEIGHT = 1231

# 游戏区域：如果游戏画面只是客户区的一部分（如网页/小程序内嵌），需设置偏移
# width/height 为 0 时自动使用客户区剩余尺寸
GAME_REGION = {
    'x_offset': 0,
    'y_offset': 0,
    'width': 0,
    'height': 0
}

# 各元素相对于游戏区域的像素坐标（基于截图测量）
# 这些值需要根据你的实际分辨率微调一次
ELEMENTS = {
    # 我的手牌区域（底部棕色条）
    'my_hand': {
        'x': 100, 'y': 980,
        'w': 1870, 'h': 180
    },
    # 底牌区域（左上角）
    'bottom_cards': {
        'x': 50, 'y': 50,
        'w': 500, 'h': 130
    },
    # 中央出牌区
    'play_area': {
        'x': 400, 'y': 280,
        'w': 1250, 'h': 600
    },
    # 上家（顶部中央头像区）
    'top_player': {
        'x': 900, 'y': 20,
        'w': 250, 'h': 100,
        'count_x': 960, 'count_y': 40
    },
    # 左家
    'left_player': {
        'x': 20, 'y': 350,
        'w': 280, 'h': 180,
        'count_x': 80, 'count_y': 380
    },
    # 右家
    'right_player': {
        'x': 1760, 'y': 350,
        'w': 280, 'h': 180,
        'count_x': 1820, 'count_y': 380
    },
    # 地主标识位置（三个候选）
    'landmark_left': {'x': 80, 'y': 320, 'w': 80, 'h': 40},
    'landmark_top': {'x': 950, 'y': 80, 'w': 80, 'h': 40},
    'landmark_right': {'x': 1800, 'y': 320, 'w': 80, 'h': 40},
    # 底部信息栏（倍率、金币等）
    'bottom_bar': {
        'x': 0, 'y': 1160,
        'w': WINDOW_WIDTH, 'h': 70
    }
}

# 缩放基准（用于其他分辨率适配）
BASE_WIDTH = 2071
BASE_HEIGHT = 1231


# ==================== 窗口操作 ====================

class WindowFinder:
    def __init__(self):
        self.hwnd = None
        # 设置 DPI 感知，确保 GetClientRect/ClientToScreen 返回物理像素（与 mss 一致）
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
        
    def find(self, title="腾讯欢乐斗地主") -> Optional[int]:
        def callback(hwnd, extra):
            if win32gui.IsWindowVisible(hwnd):
                t = win32gui.GetWindowText(hwnd)
                if title in t:
                    extra.append(hwnd)
            return True
        
        handles = []
        win32gui.EnumWindows(callback, handles)
        
        if handles:
            self.hwnd = handles[0]
            # 置顶
            placement = win32gui.GetWindowPlacement(self.hwnd)
            if placement[1] == win32con.SW_SHOWMINIMIZED:
                win32gui.ShowWindow(self.hwnd, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(self.hwnd)
            time.sleep(0.2)
            return self.hwnd
        
        return None
    
    def get_client_rect(self) -> Dict[str, int]:
        """获取客户区绝对屏幕坐标（使用 ClientToScreen 最准确）"""
        cr = win32gui.GetClientRect(self.hwnd)
        left, top = win32gui.ClientToScreen(self.hwnd, (0, 0))
        right, bottom = win32gui.ClientToScreen(self.hwnd, (cr[2], cr[3]))
        
        return {
            'x': left,
            'y': top,
            'width': right - left,
            'height': bottom - top
        }


# ==================== 截屏 ====================

class Capture:
    def __init__(self):
        self.mss = mss.mss()
        
    def capture(self, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """截取指定区域 (left, top, right, bottom)"""
        monitor = {
            "left": bbox[0], "top": bbox[1],
            "width": bbox[2] - bbox[0], "height": bbox[3] - bbox[1]
        }
        img = np.array(self.mss.grab(monitor))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


# ==================== 坐标系统 ====================

class FixedCoords:
    """
    固定坐标系统
    基于基准分辨率，自动适配实际窗口缩放
    支持 GAME_REGION 偏移（游戏画面只是客户区一部分时使用）
    """
    
    def __init__(self, client_rect: Dict[str, int], actual_size: Tuple[int, int] = None):
        self.client = client_rect
        
        # 游戏区域在客户区内的偏移和实际尺寸
        self.game_x = GAME_REGION['x_offset']
        self.game_y = GAME_REGION['y_offset']
        
        # 优先使用截图实际尺寸（解决 DPI 缩放导致的坐标不匹配）
        if actual_size:
            actual_w, actual_h = actual_size
        else:
            actual_w = client_rect['width']
            actual_h = client_rect['height']
        
        self.game_w = GAME_REGION['width'] if GAME_REGION['width'] > 0 else actual_w - self.game_x
        self.game_h = GAME_REGION['height'] if GAME_REGION['height'] > 0 else actual_h - self.game_y
        
        # 缩放比例：游戏区域尺寸 vs 基准尺寸
        self.scale_x = self.game_w / BASE_WIDTH
        self.scale_y = self.game_h / BASE_HEIGHT
        self.scale = (self.scale_x + self.scale_y) / 2
        
        print(f"[坐标系统] 客户区: {client_rect['width']}x{client_rect['height']} @ ({client_rect['x']},{client_rect['y']})")
        print(f"[坐标系统] 截图实际尺寸: {actual_w}x{actual_h}")
        print(f"[坐标系统] 游戏区域: {self.game_w}x{self.game_h} @ 偏移({self.game_x},{self.game_y})")
        print(f"[坐标系统] 缩放比例: X={self.scale_x:.3f}, Y={self.scale_y:.3f}")
        
    def get_abs(self, name: str) -> Dict[str, int]:
        """获取元素的屏幕绝对坐标（用于鼠标操作等）"""
        if name not in ELEMENTS:
            raise ValueError(f"未知元素: {name}")
        
        elem = ELEMENTS[name]
        # 游戏区域左上角在屏幕上的绝对位置
        base_x = self.client['x'] + self.game_x
        base_y = self.client['y'] + self.game_y
        
        result = {}
        for key in ['x', 'count_x']:
            if key in elem:
                result[key] = int(base_x + elem[key] * self.scale_x)
        for key in ['y', 'count_y']:
            if key in elem:
                result[key] = int(base_y + elem[key] * self.scale_y)
        if 'w' in elem:
            result['w'] = int(elem['w'] * self.scale_x)
        if 'h' in elem:
            result['h'] = int(elem['h'] * self.scale_y)
        
        return result
    
    def get_rel(self, name: str) -> Dict[str, int]:
        """获取元素相对于客户区左上角的坐标（用于 ROI 提取、画框）"""
        if name not in ELEMENTS:
            raise ValueError(f"未知元素: {name}")
        
        elem = ELEMENTS[name]
        result = {}
        for key in ['x', 'count_x']:
            if key in elem:
                result[key] = int(self.game_x + elem[key] * self.scale_x)
        for key in ['y', 'count_y']:
            if key in elem:
                result[key] = int(self.game_y + elem[key] * self.scale_y)
        if 'w' in elem:
            result['w'] = int(elem['w'] * self.scale_x)
        if 'h' in elem:
            result['h'] = int(elem['h'] * self.scale_y)
        
        return result
    
    def extract(self, screenshot: np.ndarray, name: str) -> np.ndarray:
        """从客户区截图中提取元素 ROI"""
        rect = self.get_rel(name)
        x = rect['x']
        y = rect['y']
        w = rect['w']
        h = rect['h']
        
        # 边界检查
        h_s, w_s = screenshot.shape[:2]
        x = max(0, min(x, w_s - 1))
        y = max(0, min(y, h_s - 1))
        w = min(w, w_s - x)
        h = min(h, h_s - y)
        
        if w <= 0 or h <= 0:
            return np.array([])
        
        return screenshot[y:y+h, x:x+w]


# ==================== 高速捕获 ====================

class HighSpeedCapture:
    def __init__(self, fps: int = 30, buffer_sec: float = 2.0):
        self.fps = fps
        self.buffer = deque(maxlen=int(fps * buffer_sec))
        self.running = False
        self.thread = None
        self.capture = Capture()
        self.stats = {'frames': 0, 'start': 0}
        
    def start(self, client_rect: Dict[str, int]):
        self.running = True
        self.client = client_rect
        self.stats['start'] = time.time()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        print(f"[捕获] 启动 {self.fps} FPS")
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        print("[捕获] 已停止")
        
    def _loop(self):
        interval = 1.0 / self.fps
        bbox = (
            self.client['x'], self.client['y'],
            self.client['x'] + self.client['width'],
            self.client['y'] + self.client['height']
        )
        
        while self.running:
            t0 = time.time()
            try:
                img = self.capture.capture(bbox)
                self.buffer.append({
                    'time': time.time(),
                    'image': img
                })
                self.stats['frames'] += 1
            except Exception as e:
                print(f"[捕获错误] {e}")
            
            elapsed = time.time() - t0
            if elapsed < interval:
                time.sleep(interval - elapsed)
    
    def latest(self):
        return self.buffer[-1] if self.buffer else None
    
    def recent(self, seconds: float = 1.0):
        cutoff = time.time() - seconds
        return [f for f in self.buffer if f['time'] >= cutoff]


# ==================== 主控 ====================

class DDZBot:
    def __init__(self):
        self.finder = WindowFinder()
        self.capture = None
        self.coords = None
        self.highspeed = None
        # 优先使用 YOLO 两阶段识别器，否则回退到模板匹配
        if YOLO_AVAILABLE and os.path.exists('ddz_detect_best.pt') and os.path.exists('card_classifier_best.pth'):
            self.recognizer = TwoStageRecognizerV3()
            print("[识别器] 使用 YOLO 两阶段识别")
        else:
            self.recognizer = CardRecognizer()
            print("[识别器] 使用模板匹配（YOLO 模型未找到）")
        
    def init(self) -> bool:
        """初始化：找窗口 → 加载配置 → 建坐标 → 启动捕获"""
        print("=" * 50)
        print("欢乐斗地主 - 固定坐标版")
        print("=" * 50)
        
        # 1. 找窗口
        print("\n[1/4] 查找窗口...")
        hwnd = self.finder.find()
        if not hwnd:
            print("❌ 未找到窗口，请确认游戏已打开")
            return False
        print(f"✅ 窗口句柄: {hwnd}")
        
        # 2. 加载坐标配置（如果存在）
        print("\n[2/4] 加载配置...")
        config_file = "ddz_config.json"
        global ELEMENTS, BASE_WIDTH, BASE_HEIGHT, GAME_REGION
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                saved = json.load(f)
                ELEMENTS = saved['elements']
                BASE_WIDTH = saved['base_width']
                BASE_HEIGHT = saved['base_height']
                if 'game_region' in saved:
                    GAME_REGION = saved['game_region']
            print(f"✅ 已加载: {config_file}")
        else:
            print("⚠️ 未找到 ddz_config.json，使用默认坐标")
        
        # 3. 获取客户区坐标
        client = self.finder.get_client_rect()
        print(f"\n[3/4] 客户区: ({client['x']}, {client['y']}) "
              f"{client['width']}x{client['height']}")
        
        # 4. 建立坐标系统
        print("\n[4/4] 建立坐标...")
        self.coords = FixedCoords(client)
        
        # 打印各元素位置（相对客户区）
        for name in ['my_hand', 'bottom_cards', 'play_area', 
                    'top_player', 'left_player', 'right_player']:
            rect = self.coords.get_rel(name)
            print(f"   {name:12s}: ({rect.get('x',0)}, {rect.get('y',0)}) "
                  f"{rect.get('w',0)}x{rect.get('h',0)}")
        
        # 4. 启动捕获
        self.highspeed = HighSpeedCapture(fps=30)
        self.highspeed.start(client)
        
        # 用截图实际尺寸校准坐标系统（解决 DPI 缩放偏差）
        f = self.highspeed.latest()
        if f:
            actual_w, actual_h = f['image'].shape[1], f['image'].shape[0]
            if abs(actual_w - client['width']) > 10 or abs(actual_h - client['height']) > 10:
                print(f"[DPI校准] 截图尺寸({actual_w}x{actual_h})与客户区({client['width']}x{client['height']})不一致，重新校准坐标")
                self.coords = FixedCoords(client, actual_size=(actual_w, actual_h))
        
        print("\n✅ 初始化完成！")
        return True
    
    def save_debug(self, img: np.ndarray, filename: str = "debug_fixed.png"):
        """保存调试图"""
        vis = img.copy()
        
        # 画框
        colors = {
            'my_hand': (0, 255, 0),
            'bottom_cards': (255, 0, 0),
            'play_area': (0, 0, 255),
            'top_player': (255, 255, 0),
            'left_player': (255, 0, 255),
            'right_player': (0, 255, 255),
        }
        
        for name, color in colors.items():
            rect = self.coords.get_rel(name)
            x, y = rect['x'], rect['y']
            w, h = rect.get('w', 0), rect.get('h', 0)
            if w > 0 and h > 0:
                cv2.rectangle(vis, (x, y), (x+w, y+h), color, 2)
                cv2.putText(vis, name, (x, max(y-5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 标注游戏区域边界（白色虚线）
        gx, gy = self.coords.game_x, self.coords.game_y
        gw, gh = self.coords.game_w, self.coords.game_h
        cv2.rectangle(vis, (gx, gy), (gx+gw, gy+gh), (255, 255, 255), 2)
        cv2.putText(vis, "GAME_REGION", (gx, max(gy-5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imwrite(filename, vis)
        print(f"[调试] 已保存: {filename}")
        return filename
    
    def extract_regions(self, img: np.ndarray):
        """提取各区域保存"""
        for name in ['my_hand', 'bottom_cards', 'play_area']:
            roi = self.coords.extract(img, name)
            if roi.size > 0:
                fname = f"region_{name}.png"
                cv2.imwrite(fname, roi)
                print(f"[提取] {fname}: {roi.shape[1]}x{roi.shape[0]}")
            else:
                print(f"[提取] region_{name}.png: 提取失败（区域超出范围或尺寸为0）")
    
    def run(self):
        """运行测试"""
        if not self.init():
            return
        
        print("\n[测试] 运行5秒...")
        for i in range(5):
            time.sleep(1)
            s = self.highspeed.stats
            fps = s['frames'] / (time.time() - s['start'] + 0.001)
            print(f"  {i+1}s: {s['frames']}帧, FPS:{fps:.1f}")
        
        # 用主线程单次截图（与 capture_templates 一致，避免 mss 后台线程偏差）
        print("\n[测试] 主线程截图...")
        cap = Capture()
        bbox = (self.coords.client['x'], self.coords.client['y'],
                self.coords.client['x'] + self.coords.client['width'],
                self.coords.client['y'] + self.coords.client['height'])
        img = cap.capture(bbox)
        
        print("\n[测试] 保存调试...")
        self.save_debug(img)
        self.extract_regions(img)
        
        # 手牌识别
        print("\n[测试] 识别手牌...")
        hand_img = self.coords.extract(img, 'my_hand')
        if hand_img.size > 0:
            cards = self.recognizer.recognize(hand_img.copy())
            print(f"手牌: {cards} (共{len(cards)}张)")
            cv2.imwrite("debug_recognize.png", hand_img)

        # 底牌识别（小牌专用方法）
        print("\n[测试] 识别底牌...")
        bottom_img = self.coords.extract(img, 'bottom_cards')
        if bottom_img.size > 0:
            bottom_cards = self.recognizer.recognize_bottom(bottom_img.copy())
            names = [n for n, c in bottom_cards]
            confs = [f"{c:.0%}" for n, c in bottom_cards]
            print(f"底牌: {names} (共{len(names)}张)")
            print(f"      置信度: {confs}")
            cv2.imwrite("debug_bottom.png", bottom_img)

        # 出牌识别（过滤桌面干扰）
        print("\n[测试] 识别出牌...")
        play_img = self.coords.extract(img, 'play_area')
        if play_img.size > 0:
            play_cards = self.recognizer.recognize_play(play_img.copy())
            print(f"出牌: {play_cards} (共{len(play_cards)}张)")
            cv2.imwrite("debug_play.png", play_img)

        self.highspeed.stop()


# ==================== 校准工具 ====================

def detect_game_region(img: np.ndarray) -> Tuple[int, int, int, int]:
    """
    自动检测客户区中的游戏区域
    基于斗地主蓝色桌面背景进行颜色分割
    返回: (x_offset, y_offset, width, height) 相对于客户区
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 斗地主桌面蓝色背景范围
    lower = np.array([95, 50, 80])
    upper = np.array([125, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    
    # 形态学闭运算填充小孔
    kernel = np.ones((30, 30), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 找最大连通区域
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, img.shape[1], img.shape[0]
    
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    img_area = img.shape[1] * img.shape[0]
    
    # 如果蓝色区域太小（不到画面 10%），放弃自动检测
    if area < img_area * 0.1:
        return 0, 0, img.shape[1], img.shape[0]
    
    x, y, w, h = cv2.boundingRect(largest)
    return x, y, w, h


def calibrate():
    """
    校准工具：手动调整坐标
    运行后会显示窗口截图，按方向键微调区域位置
    按 G 切换调整游戏区域（GAME_REGION），解决游戏画面不在客户区左上角的问题
    按 T 自动检测游戏区域
    """
    finder = WindowFinder()
    hwnd = finder.find()
    if not hwnd:
        print("未找到窗口")
        return
    
    client = finder.get_client_rect()
    cap = Capture()
    
    # 加载现有配置
    config_file = "ddz_config.json"
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            saved = json.load(f)
            global ELEMENTS, BASE_WIDTH, BASE_HEIGHT, GAME_REGION
            ELEMENTS = saved['elements']
            BASE_WIDTH = saved['base_width']
            BASE_HEIGHT = saved['base_height']
            if 'game_region' in saved:
                GAME_REGION = saved['game_region']
            print(f"已加载配置: {config_file}")
    
    # 当前调整状态
    current_elem = 'my_hand'
    adjust_region = False  # False=调整元素, True=调整游戏区域
    step = 5
    
    print("\n=== 校准模式 ===")
    print("按键说明:")
    print("  W/A/S/D = 移动当前区域")
    print("  Q/E     = 减/加宽度")
    print("  R/F     = 减/加高度")
    print("  +/-     = 调整步长")
    print("  N       = 下一个元素")
    print("  G       = 切换：调整元素 / 调整游戏区域(GAME_REGION)")
    print("  T       = 自动检测游戏区域（基于蓝色背景）")
    print("  ESC     = 保存并退出")
    
    # 建立坐标系统（循环外创建一次，避免重复打印）
    coords = FixedCoords(client)
    
    while True:
        # 截图（整个客户区）
        bbox = (client['x'], client['y'],
                client['x'] + client['width'],
                client['y'] + client['height'])
        img = cap.capture(bbox)
        
        debug = img.copy()
        region_changed = False
        
        # 给非游戏区域加半透明黑色遮罩，让 GAME_REGION 范围一目了然
        gx, gy = coords.game_x, coords.game_y
        gw, gh = coords.game_w, coords.game_h
        overlay = debug.copy()
        cv2.rectangle(overlay, (0, 0), (client['width'], gy), (0, 0, 0), -1)          # 上方
        cv2.rectangle(overlay, (0, gy+gh), (client['width'], client['height']), (0, 0, 0), -1)  # 下方
        cv2.rectangle(overlay, (0, gy), (gx, gy+gh), (0, 0, 0), -1)                   # 左侧
        cv2.rectangle(overlay, (gx+gw, gy), (client['width'], gy+gh), (0, 0, 0), -1)  # 右侧
        cv2.addWeighted(overlay, 0.5, debug, 0.5, 0, debug)
        
        # 绘制游戏区域边界（红色框，醒目）
        cv2.rectangle(debug, (gx, gy), (gx+gw, gy+gh), (0, 0, 255), 2)
        cv2.putText(debug, "GAME_REGION", (gx, max(gy-5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 绘制所有元素区域
        for name, elem in ELEMENTS.items():
            if adjust_region:
                color = (128, 128, 128)  # 灰色=不可调
                thickness = 1
            elif name == current_elem:
                color = (0, 255, 255)     # 黄色=当前调整
                thickness = 3
            else:
                color = (0, 255, 0)       # 绿色=其他
                thickness = 1
            
            rect = coords.get_rel(name)
            x, y = rect['x'], rect['y']
            w, h = rect.get('w', 100), rect.get('h', 100)
            cv2.rectangle(debug, (x, y), (x+w, y+h), color, thickness)
            if not adjust_region:
                cv2.putText(debug, name, (x, max(y-5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 显示信息
        if adjust_region:
            info = (f"Adjust: GAME_REGION | Step: {step} | "
                    f"Offset: ({GAME_REGION['x_offset']}, {GAME_REGION['y_offset']}) | "
                    f"Size: ({GAME_REGION['width'] or 'auto'}, {GAME_REGION['height'] or 'auto'})")
        else:
            e = ELEMENTS[current_elem]
            info = (f"Adjust: {current_elem} | Step: {step} | "
                    f"Pos: ({e['x']}, {e['y']}) | Size: ({e.get('w',0)}x{e.get('h',0)})")
        cv2.putText(debug, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("Calibrate", debug)
        cv2.waitKey(1)  # 确保窗口已创建
        
        # 把校准窗口移到游戏窗口右侧，避免重叠导致截屏递归
        try:
            cal_hwnds = []
            def _cb(hwnd, _):
                if win32gui.IsWindowVisible(hwnd) and "Calibrate" in win32gui.GetWindowText(hwnd):
                    cal_hwnds.append(hwnd)
                return True
            win32gui.EnumWindows(_cb, None)
            if cal_hwnds:
                # 移到游戏窗口右侧，不遮挡游戏画面
                cv2.moveWindow("Calibrate", client['x'] + client['width'] + 20, client['y'])
                win32gui.SetForegroundWindow(cal_hwnds[0])
        except Exception:
            pass
        
        key = cv2.waitKey(100) & 0xFF
        
        if key == 27:  # ESC
            break
        # 统一转小写处理按键（兼容 Caps Lock / Shift）
        key_char = chr(key).lower() if 32 <= key <= 126 else ''
        
        if key == 27:
            break
        elif key_char == 'g':
            adjust_region = not adjust_region
            print(f"切换模式: {'调整游戏区域(GAME_REGION)' if adjust_region else '调整元素'}")
        elif key_char == 't':
            print("正在自动检测游戏区域...")
            xo, yo, w, h = detect_game_region(img)
            GAME_REGION['x_offset'] = xo
            GAME_REGION['y_offset'] = yo
            GAME_REGION['width'] = w
            GAME_REGION['height'] = h
            print(f"检测到游戏区域: ({xo}, {yo}) {w}x{h}")
            region_changed = True
        elif key_char == '0':
            # 快速重置 GAME_REGION 为全客户区
            GAME_REGION['x_offset'] = 0
            GAME_REGION['y_offset'] = 0
            GAME_REGION['width'] = 0
            GAME_REGION['height'] = 0
            region_changed = True
            print("GAME_REGION 已重置为全客户区")
        elif key == ord('+'):
            step += 1
        elif key == ord('-'):
            step = max(1, step - 1)
        elif adjust_region:
            if key_char == 'w':
                GAME_REGION['y_offset'] -= step
                region_changed = True
            elif key_char == 's':
                GAME_REGION['y_offset'] += step
                region_changed = True
            elif key_char == 'a':
                GAME_REGION['x_offset'] -= step
                region_changed = True
            elif key_char == 'd':
                GAME_REGION['x_offset'] += step
                region_changed = True
            elif key_char == 'q':
                GAME_REGION['width'] = max(10, GAME_REGION['width'] - step)
                region_changed = True
            elif key_char == 'e':
                GAME_REGION['width'] = max(10, GAME_REGION['width'] + step)
                region_changed = True
            elif key_char == 'r':
                GAME_REGION['height'] = max(10, GAME_REGION['height'] - step)
                region_changed = True
            elif key_char == 'f':
                GAME_REGION['height'] = max(10, GAME_REGION['height'] + step)
                region_changed = True
        else:
            if key_char == 'w':
                ELEMENTS[current_elem]['y'] -= step
            elif key_char == 's':
                ELEMENTS[current_elem]['y'] += step
            elif key_char == 'a':
                ELEMENTS[current_elem]['x'] -= step
            elif key_char == 'd':
                ELEMENTS[current_elem]['x'] += step
            elif key_char == 'q':
                ELEMENTS[current_elem]['w'] = max(10, ELEMENTS[current_elem].get('w', 100) - step)
            elif key_char == 'e':
                ELEMENTS[current_elem]['w'] = ELEMENTS[current_elem].get('w', 100) + step
            elif key_char == 'r':
                ELEMENTS[current_elem]['h'] = max(10, ELEMENTS[current_elem].get('h', 100) - step)
            elif key_char == 'f':
                ELEMENTS[current_elem]['h'] = ELEMENTS[current_elem].get('h', 100) + step
            elif key_char == 'n':
                keys = list(ELEMENTS.keys())
                idx = keys.index(current_elem)
                current_elem = keys[(idx + 1) % len(keys)]
                print(f"切换到: {current_elem}")
    
        if region_changed:
            coords = FixedCoords(client)
    
    cv2.destroyAllWindows()
    
    # 保存配置
    config = {
        'base_width': BASE_WIDTH,
        'base_height': BASE_HEIGHT,
        'game_region': GAME_REGION,
        'elements': ELEMENTS
    }
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"\n配置已保存: {config_file}")


# ==================== 卡牌识别 ====================

class CardRecognizer:
    """
    基于模板匹配的卡牌识别器
    """

    KEY_MAP = {
        '3': '3', '4': '4', '5': '5', '6': '6', '7': '7',
        '8': '8', '9': '9', '0': '10',
        'j': 'J', 'q': 'Q', 'k': 'K', 'a': 'A', '2': '2',
        's': 'SJ', 'b': 'BJ',
    }

    def __init__(self, templates_dir: str = "templates"):
        self.templates_dir = templates_dir
        self.templates = {}
        self._load_templates()

    def _load_templates(self):
        if not os.path.exists(self.templates_dir):
            print(f"[识别器] 模板目录不存在: {self.templates_dir}")
            return
        count = 0
        for fname in sorted(os.listdir(self.templates_dir)):
            if not fname.lower().endswith('.png'):
                continue
            name = fname[:-4]
            path = os.path.join(self.templates_dir, fname)
            img = cv2.imread(path)
            if img is None:
                continue
            self.templates[name] = img
            count += 1
        print(f"[识别器] 已加载 {count} 个模板: {list(self.templates.keys())}")

    def _split_cards(self, hand_img: np.ndarray, debug_path: str = None, y_limit: float = 0.50) -> list:
        """
        分割区域内的牌
        y_limit: 只保留 y < h_total * y_limit 的轮廓（手牌用 0.5 排除"炸"字，底牌/出牌用 1.0 不限制）
        """
        if hand_img.size == 0:
            return []
        
        h_total, w_total = hand_img.shape[:2]
        
        # 双通道文字检测：HSV 抓红色数字 + 灰度自适应阈值抓黑字，取并集
        hsv = cv2.cvtColor(hand_img, cv2.COLOR_BGR2HSV)
        # 红色数字：H 在 0-10 或 170-180，S>80
        red1 = cv2.inRange(hsv, np.array([0, 80, 50]), np.array([10, 255, 255]))
        red2 = cv2.inRange(hsv, np.array([170, 80, 50]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red1, red2)
        
        # 黑色/所有文字：灰度自适应阈值
        gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
        black_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 11, 2)
        
        # 合并：红色数字 + 黑色文字
        combined_mask = cv2.bitwise_or(red_mask, black_mask)
        # 轻微开运算去噪
        kernel_open = np.ones((2, 2), np.uint8)
        inv_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open)
        
        if debug_path:
            cv2.imwrite(debug_path, inv_mask)
            cv2.imwrite("debug_hand_img.png", hand_img)
        
        contours, _ = cv2.findContours(inv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            # 关键过滤：
            # 1. y_limit 控制是否排除底部区域（手牌用 0.5 排除"炸"字）
            # 2. 宽高在合理范围，排除细碎噪声
            # 3. 面积不能太小
            if 4 <= w <= 50 and 8 <= h <= 60 and area >= 20 and y < h_total * y_limit:
                cx = x + w // 2
                text_boxes.append((cx, x, y, w, h))
        
        print(f"[调试] 检测到 {len(text_boxes)} 个有效轮廓（已排除炸字/噪声）")
        if len(text_boxes) < 2:
            return []
        
        text_boxes.sort(key=lambda b: b[0])
        
        # 聚类分牌：固定阈值 22px
        # 同一张牌内的数字/花色间距通常 5-20px < 22，不会分牌
        # 相邻牌间距通常 25-80px > 22，会正确分牌
        # 35张牌重叠时最小间距约25px，22能正确分开
        GAP_THRESHOLD = 22
        
        groups = [[text_boxes[0]]]
        for box in text_boxes[1:]:
            if box[0] - groups[-1][-1][0] > GAP_THRESHOLD:
                groups.append([box])
            else:
                groups[-1].append(box)
        
        # 框的左边界对齐该组最左侧的文字轮廓，确保每张牌的 card_img
        # 都从牌面数字的最左边缘开始，这样 _extract_digit 提取的内容一致
        cards = []
        for i, group in enumerate(groups):
            # 该组最左侧文字的 x 坐标
            min_text_x = min(b[1] for b in group)
            # 向左扩展 5px，防止数字在最左侧文字左边被截掉
            x1 = max(0, min_text_x - 5)
            # 固定牌宽 78px（25张/1950px≈78px间距，刚好不重叠）
            x2 = min(w_total, x1 + 78)
            cw = x2 - x1
            if cw > 25:
                cards.append((x1, 0, cw, h_total))
        
        # 保存分割调试图
        debug_vis = hand_img.copy()
        for i, (cx, x, y, w, h) in enumerate(text_boxes):
            cv2.rectangle(debug_vis, (x, y), (x+w, y+h), (0, 255, 255), 1)
        for i, (x, y, w, h) in enumerate(cards):
            color = (0, 255, 0) if i % 2 == 0 else (255, 0, 0)
            cv2.rectangle(debug_vis, (x, y), (x+w, y+h), color, 2)
            cv2.putText(debug_vis, str(i+1), (x, max(y-3, 12)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        out_name = debug_path.replace("_mask", "_boxes") if debug_path else "debug_split_boxes.png"
        cv2.imwrite(out_name, debug_vis)
        
        return cards

    def _extract_digit(self, card_img: np.ndarray) -> np.ndarray:
        """提取牌面数字区域：只在左侧搜索，避免右侧重叠区混入相邻牌"""
        h, w = card_img.shape[:2]
        
        # 只在左侧 40px 内做自适应阈值，避免截到右侧重叠的相邻牌
        search_w = min(w, 40)
        left_region = card_img[0:min(h, 110), 0:search_w]
        if left_region.size == 0:
            return card_img[2:min(h, 102), 2:min(w, 62)]
        
        gray = cv2.cvtColor(left_region, cv2.COLOR_BGR2GRAY)
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            # 放宽过滤，确保细碎字母也能被抓到
            if 2 <= bw <= 35 and 5 <= bh <= 70:
                boxes.append((bx, by, bx+bw, by+bh))
        
        # 兜底：如果左侧啥也没检测到，固定提取左上角
        if len(boxes) == 0:
            return card_img[2:min(h, 102), 2:min(w, 62)]
        
        # 按 x 排序，保留最左侧 1-2 个轮廓（数字+花色）
        boxes.sort(key=lambda b: b[0])
        keep = [boxes[0]]
        if len(boxes) >= 2 and boxes[1][0] - boxes[0][0] < 30:
            keep.append(boxes[1])
        
        x1 = max(0, min(b[0] for b in keep) - 2)
        y1 = max(0, min(b[1] for b in keep) - 2)
        x2 = min(search_w, max(b[2] for b in keep) + 2)
        y2 = min(min(h, 110), max(b[3] for b in keep) + 2)
        
        # 限制宽度不超过 45px，再次防止越界
        if x2 - x1 > 45:
            x2 = x1 + 45
        
        return left_region[y1:y2, x1:x2]

    def match_card(self, card_img: np.ndarray, save_path: str = None) -> tuple:
        if not self.templates or card_img.size == 0:
            return None, 0.0
        digit = self._extract_digit(card_img)
        if digit.size == 0:
            return None, 0.0
        
        # 保存 digit 调试图
        if save_path:
            cv2.imwrite(save_path, digit)
        
        # 灰度 + 自适应二值化，消除颜色（红/黑）和光照差异
        gray_digit = cv2.cvtColor(digit, cv2.COLOR_BGR2GRAY)
        _, bin_digit = cv2.threshold(gray_digit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 去除细线干扰（如相邻框重叠导致的绿线/边界线）
        kernel = np.ones((2, 2), np.uint8)
        bin_digit = cv2.morphologyEx(bin_digit, cv2.MORPH_OPEN, kernel)
        
        NORM_SIZE = (64, 64)
        digit_norm = cv2.resize(bin_digit, NORM_SIZE, interpolation=cv2.INTER_AREA)
        
        best_name, best_score = None, -1.0
        for name, tmpl in self.templates.items():
            gray_tmpl = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
            _, bin_tmpl = cv2.threshold(gray_tmpl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            tmpl_norm = cv2.resize(bin_tmpl, NORM_SIZE, interpolation=cv2.INTER_AREA)
            
            res = cv2.matchTemplate(digit_norm.astype(np.float32), tmpl_norm.astype(np.float32), cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if max_val > best_score:
                best_score = max_val
                best_name = name
        return best_name, best_score

    def recognize(self, hand_img: np.ndarray, debug: bool = False, prefix: str = "", y_limit: float = 0.50) -> list:
        """识别手牌/底牌/出牌，返回 [(name, score), ...] 按从左到右排序"""
        debug_path = f"debug_{prefix}split_mask.png" if debug else None
        cards = self._split_cards(hand_img, debug_path=debug_path, y_limit=y_limit)
        results = []
        # 先全部识别（不画框，避免框线污染后续 card_img）
        for idx, (x, y, w, h) in enumerate(cards):
            card_img = hand_img[y:y+h, x:x+w].copy()
            digit_path = f"debug_digit_{prefix}{idx+1:02d}.png"
            name, score = self.match_card(card_img, save_path=digit_path)
            tag = name if name else "None"
            print(f"  {prefix}牌{idx+1:2d}: {tag:4s} score={score:.3f}  size={card_img.shape[1]}x{card_img.shape[0]}")
            results.append((name, score, x, y, w, h))
        
        # 统一画框
        if debug:
            for name, score, x, y, w, h in results:
                color = (0, 255, 0) if score > 0.45 else (0, 0, 255)
                cv2.rectangle(hand_img, (x, y), (x+w, y+h), color, 2)
                if name:
                    cv2.putText(hand_img, f"{name}:{score:.2f}", (x, max(y-5, 15)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        final = [(n, s, x) for n, s, x, y, w, h in results if n and s > 0.45]
        final.sort(key=lambda r: r[2])
        return [(n, s) for n, s, _ in final]


def capture_templates():
    """交互式模板采集：截取 my_hand，自动分割，用户按键标记"""
    finder = WindowFinder()
    hwnd = finder.find()
    if not hwnd:
        print("未找到窗口")
        return

    client = finder.get_client_rect()
    cap = Capture()

    config_file = "ddz_config.json"
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            saved = json.load(f)
            global ELEMENTS, BASE_WIDTH, BASE_HEIGHT, GAME_REGION
            ELEMENTS = saved['elements']
            BASE_WIDTH = saved['base_width']
            BASE_HEIGHT = saved['base_height']
            if 'game_region' in saved:
                GAME_REGION = saved['game_region']
            print(f"已加载配置: {config_file}")

    recognizer = CardRecognizer()
    os.makedirs("templates", exist_ok=True)

    print("\n=== 模板采集模式 ===")
    print("按键: 3-9=数字, 0=10, J/Q/K/A/2=牌面, S=小王, B=大王")
    print("      空格=跳过, R=重新截取, ESC=退出")

    bbox = (client['x'], client['y'],
            client['x'] + client['width'],
            client['y'] + client['height'])

    while True:
        img = cap.capture(bbox)
        # 用截图实际尺寸创建坐标系统，避免 DPI 缩放偏差
        coords = FixedCoords(client, actual_size=(img.shape[1], img.shape[0]))
        print(f"[调试] 截图尺寸: {img.shape[1]}x{img.shape[0]}")
        hand_img = coords.extract(img, 'my_hand')
        print(f"[调试] my_hand 区域尺寸: {hand_img.shape[1]}x{hand_img.shape[0]}")
        cv2.imwrite("debug_hand_img.png", hand_img)
        if hand_img.size == 0:
            print("手牌区域提取失败，按 R 重试，ESC 退出")
            key = cv2.waitKey(0) & 0xFF
            if key == 27:
                break
            continue

        cards = recognizer._split_cards(hand_img, debug_path="debug_split_mask.png")
        print(f"\n检测到 {len(cards)} 张牌")
        if not cards:
            print("未分割出手牌，已保存调试图 debug_split_mask.png")
            print("按 R 重试，ESC 退出")
            key = cv2.waitKey(0) & 0xFF
            if key == 27:
                break
            continue

        # 显示分割结果
        debug_img = hand_img.copy()
        for i, (x, y, w, h) in enumerate(cards):
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(debug_img, str(i+1), (x, max(y-5, 15)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("Split Result", debug_img)
        print("[提示] 显示分割结果 1.5 秒，请查看绿色框是否正确...")
        cv2.waitKey(1500)

        # 逐张标记
        for i, (x, y, w, h) in enumerate(cards):
            digit = recognizer._extract_digit(hand_img[y:y+h, x:x+w])
            if digit.size == 0:
                continue

            display = np.zeros((220, 420, 3), dtype=np.uint8)
            dh, dw = digit.shape[:2]
            y_off, x_off = max(0, (200-dh)//2), max(0, (400-dw)//2)
            if dh <= 200 and dw <= 400:
                display[y_off:y_off+dh, x_off:x_off+dw] = digit

            # 醒目提示
            info = f"=== 第 {i+1}/{len(cards)} 张 ==="
            cv2.putText(display, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display, f"已有模板: {list(recognizer.templates.keys())}", (10, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(display, "按对应键保存: 3-9,0,J,Q,K,A,2,S,B", (10, 180),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(display, "SPACE=跳过  R=重截  ESC=结束", (10, 205),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)
            
            cv2.imshow("Template Capture", display)
            cv2.waitKey(1)
            # 置顶窗口
            try:
                t_hwnds = []
                def _tcb(hwnd, _):
                    if win32gui.IsWindowVisible(hwnd) and "Template Capture" in win32gui.GetWindowText(hwnd):
                        t_hwnds.append(hwnd)
                    return True
                win32gui.EnumWindows(_tcb, None)
                if t_hwnds:
                    win32gui.SetForegroundWindow(t_hwnds[0])
            except Exception:
                pass

            key = cv2.waitKey(0) & 0xFF
            if key == 27:
                break
            elif key == ord('r') or key == ord('R'):
                print("重新截取...")
                break
            elif key == ord(' '):
                print(f"  跳过第 {i+1} 张")
                continue

            key_char = chr(key).lower() if 32 <= key <= 126 else ''
            if key_char in CardRecognizer.KEY_MAP:
                name = CardRecognizer.KEY_MAP[key_char]
                if name in recognizer.templates:
                    print(f"  '{name}' 已存在，按 O 覆盖，其他键跳过")
                    k2 = cv2.waitKey(0) & 0xFF
                    if k2 != ord('o') and k2 != ord('O'):
                        print(f"  跳过覆盖 '{name}'")
                        continue
                path = os.path.join("templates", f"{name}.png")
                cv2.imwrite(path, digit)
                recognizer.templates[name] = digit
                print(f"  ✅ 已保存模板: {name}")
            else:
                print(f"  未知键 '{key_char}'，跳过")

        print("\n按 R 截取新手牌，ESC 退出")
        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            break

    cv2.destroyAllWindows()
    saved = list(recognizer.templates.keys())
    print(f"\n=============================")
    print(f"模板采集完成！共保存 {len(saved)} 个模板")
    print(f"模板列表: {saved}")
    print(f"模板目录: {os.path.abspath('templates')}")
    print(f"=============================")


# ==================== 入口 ====================

def save_screenshots():
    """快速保存手牌截图到 screenshots/ 目录，用于 YOLO 数据集准备"""
    import time
    os.makedirs("screenshots", exist_ok=True)

    finder = WindowFinder()
    hwnd = finder.find()
    if not hwnd:
        print("❌ 未找到游戏窗口")
        return

    # 加载校准配置（如果存在）
    config_file = "ddz_config.json"
    global ELEMENTS, BASE_WIDTH, BASE_HEIGHT, GAME_REGION
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            saved = json.load(f)
            ELEMENTS = saved['elements']
            BASE_WIDTH = saved['base_width']
            BASE_HEIGHT = saved['base_height']
            if 'game_region' in saved:
                GAME_REGION = saved['game_region']
        print(f"[配置] 已加载: {config_file}")

    client = finder.get_client_rect()
    coords = FixedCoords(client)
    cap = Capture()
    bbox = (client['x'], client['y'],
            client['x'] + client['width'],
            client['y'] + client['height'])

    # 用截图实际尺寸校准坐标（和 DDZBot.init 一致）
    img = cap.capture(bbox)
    if img is not None and img.size > 0:
        actual_size = (img.shape[1], img.shape[0])
        coords = FixedCoords(client, actual_size=actual_size)
        print(f"[校准] 截图尺寸: {actual_size[0]}x{actual_size[1]}, "
              f"缩放: {coords.scale_x:.4f}x{coords.scale_y:.4f}")

    idx = len([f for f in os.listdir("screenshots") if f.startswith("hand_")])

    print("=" * 50)
    print("截图保存模式")
    print("=" * 50)
    print("操作说明:")
    print("  [S] 保存当前手牌截图")
    print("  [Q] 退出")
    print("=" * 50)

    import msvcrt
    import time

    while True:
        time.sleep(0.05)
        if msvcrt.kbhit():
            key = msvcrt.getch().decode('utf-8', errors='ignore').lower()
            if key == 's':
                img = cap.capture(bbox)
                hand_img = coords.extract(img, 'my_hand')
                if hand_img.size > 0:
                    path = f"screenshots/hand_{idx:03d}.png"
                    cv2.imwrite(path, hand_img)
                    print(f"✅ 已保存: {path} ({hand_img.shape[1]}x{hand_img.shape[0]})")
                    idx += 1
                else:
                    print("❌ 手牌区域为空")
            elif key == 'q':
                break

    print(f"\n共保存 {idx} 张截图到 screenshots/")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--calibrate":
        calibrate()
    elif len(sys.argv) > 1 and sys.argv[1] == "--capture-templates":
        capture_templates()
    elif len(sys.argv) > 1 and sys.argv[1] == "--save-screenshot":
        save_screenshots()
    else:
        bot = DDZBot()
        bot.run()