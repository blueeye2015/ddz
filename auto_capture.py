#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动截图工具 - 支持 mss 和 win32gui 两种方式
用于批量收集斗地主完整截图，供整图识别训练使用

用法:
    python auto_capture.py              # 默认 mss 模式，手动输窗口名
    python auto_capture.py --backend mss --interval 1.0 --output raw_auto
    python auto_capture.py --backend win32 --interval 0.5
"""

import cv2
import numpy as np
import os
import sys
import time
import argparse
import msvcrt
import ctypes

# 设置 DPI 感知，确保坐标和 mss 截图像素一致
try:
    ctypes.windll.user32.SetProcessDPIAware()
except Exception:
    pass

# 尝试导入 win32gui 截图模块
try:
    import win32gui, win32ui, win32con
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False
    print("[警告] win32gui 不可用，只能用 mss 模式")

try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False


class WindowCaptureMSS:
    """mss 截图方式 - 窗口必须在屏幕可见区域"""
    def __init__(self, window_title="腾讯欢乐斗地主"):
        self.title = window_title
        self.mss = mss.mss()
        self.bbox = None
        
    def find_window(self):
        """查找窗口并返回屏幕绝对坐标 bbox (left, top, right, bottom)"""
        def callback(hwnd, extra):
            if win32gui.IsWindowVisible(hwnd):
                t = win32gui.GetWindowText(hwnd)
                if self.title in t:
                    extra.append(hwnd)
            return True
        
        handles = []
        win32gui.EnumWindows(callback, handles)
        if not handles:
            return None
        
        hwnd = handles[0]
        # 获取客户区坐标（相对于屏幕）
        cr = win32gui.GetClientRect(hwnd)
        left, top = win32gui.ClientToScreen(hwnd, (0, 0))
        right, bottom = win32gui.ClientToScreen(hwnd, (cr[2], cr[3]))
        self.bbox = (left, top, right, bottom)
        return self.bbox
    
    def capture(self):
        """截图并返回 BGR numpy 数组"""
        if self.bbox is None:
            if self.find_window() is None:
                return None
        left, top, right, bottom = self.bbox
        monitor = {"left": left, "top": top, "width": right-left, "height": bottom-top}
        img = np.array(self.mss.grab(monitor))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


class WindowCaptureWin32:
    """win32gui BitBlt 截图方式 - 某些窗口可后台截图"""
    def __init__(self, window_title="腾讯欢乐斗地主"):
        self.title = window_title
        self.hwnd = None
        self.w = 0
        self.h = 0
        self.cropped_x = 0
        self.cropped_y = 0
        
    def find_window(self):
        self.hwnd = win32gui.FindWindow(None, self.title)
        if not self.hwnd:
            # 模糊匹配
            def callback(hwnd, extra):
                if win32gui.IsWindowVisible(hwnd):
                    t = win32gui.GetWindowText(hwnd)
                    if self.title in t:
                        extra.append(hwnd)
                return True
            handles = []
            win32gui.EnumWindows(callback, handles)
            if handles:
                self.hwnd = handles[0]
            else:
                return False
        
        # 获取窗口尺寸（含边框和标题栏）
        window_rect = win32gui.GetWindowRect(self.hwnd)
        self.w = window_rect[2] - window_rect[0]
        self.h = window_rect[3] - window_rect[1]
        
        # 裁剪边框和标题栏（常见 Windows 窗口参数）
        border_pixels = 8
        titlebar_pixels = 30
        self.w = self.w - (border_pixels * 2)
        self.h = self.h - titlebar_pixels - border_pixels
        self.cropped_x = border_pixels
        self.cropped_y = titlebar_pixels
        return True
    
    def capture(self):
        """截图并返回 BGR numpy 数组"""
        if self.hwnd is None:
            if not self.find_window():
                return None
        
        try:
            wDC = win32gui.GetWindowDC(self.hwnd)
            dcObj = win32ui.CreateDCFromHandle(wDC)
            cDC = dcObj.CreateCompatibleDC()
            dataBitMap = win32ui.CreateBitmap()
            dataBitMap.CreateCompatibleBitmap(dcObj, self.w, self.h)
            cDC.SelectObject(dataBitMap)
            cDC.BitBlt((0, 0), (self.w, self.h), dcObj, (self.cropped_x, self.cropped_y), win32con.SRCCOPY)
            
            signedIntsArray = dataBitMap.GetBitmapBits(True)
            img = np.fromstring(signedIntsArray, dtype='uint8')
            img.shape = (self.h, self.w, 4)
            
            dcObj.DeleteDC()
            cDC.DeleteDC()
            win32gui.ReleaseDC(self.hwnd, wDC)
            win32gui.DeleteObject(dataBitMap.GetHandle())
            
            img = img[..., :3]  # 去掉 alpha
            img = np.ascontiguousarray(img)
            # win32ui 截图是 BGR 格式，但需要确认
            return img
        except Exception as e:
            print(f"[截图错误] {e}")
            return None


def auto_capture(backend="mss", interval=1.0, output_dir="raw_auto", window_title="腾讯欢乐斗地主", offset_x=0, offset_y=0):
    """自动截图主循环"""
    
    if backend == "win32" and not WIN32_AVAILABLE:
        print("win32gui 不可用，回退到 mss")
        backend = "mss"
    
    if backend == "mss":
        cap = WindowCaptureMSS(window_title)
    else:
        cap = WindowCaptureWin32(window_title)
    
    # 查找窗口
    print(f"[{backend}] 查找窗口: {window_title}")
    if backend == "mss":
        bbox = cap.find_window()
        if bbox is None:
            print("未找到窗口")
            return
        # 应用偏移
        if offset_x != 0 or offset_y != 0:
            cap.bbox = (bbox[0] + offset_x, bbox[1] + offset_y, bbox[2] + offset_x, bbox[3] + offset_y)
            print(f"窗口位置(含偏移): ({cap.bbox[0]},{cap.bbox[1]}) {cap.bbox[2]-cap.bbox[0]}x{cap.bbox[3]-cap.bbox[1]}")
        else:
            print(f"窗口位置: ({bbox[0]},{bbox[1]}) {bbox[2]-bbox[0]}x{bbox[3]-bbox[1]}")
    else:
        if not cap.find_window():
            print("未找到窗口")
            return
        print(f"窗口尺寸: {cap.w}x{cap.h}")
    
    # 预览截图
    print("\n预览截图...")
    preview = cap.capture()
    if preview is not None:
        preview_path = "debug_capture_preview.png"
        cv2.imwrite(preview_path, preview)
        print(f"预览已保存: {preview_path}")
        print("请检查预览图是否正确，按 Enter 开始截图，按 Q 退出...")
        while True:
            if msvcrt.kbhit():
                key = msvcrt.getch().decode('utf-8', errors='ignore').lower()
                if key == 'q':
                    print("已取消")
                    return
                elif key == '\r' or key == '\n':
                    break
            time.sleep(0.1)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n自动截图启动:")
    print(f"  方式: {backend}")
    print(f"  间隔: {interval} 秒")
    print(f"  输出: {output_dir}/")
    print(f"  按 Q 停止\n")
    
    count = 0
    last_time = 0
    
    while True:
        # 检查按键（非阻塞）
        if msvcrt.kbhit():
            key = msvcrt.getch().decode('utf-8', errors='ignore').lower()
            if key == 'q':
                print(f"\n已停止，共截图 {count} 张")
                break
        
        now = time.time()
        if now - last_time < interval:
            time.sleep(0.05)
            continue
        
        last_time = now
        img = cap.capture()
        if img is None:
            print("截图失败，重试...")
            time.sleep(1)
            continue
        
        count += 1
        fname = os.path.join(output_dir, f"auto_{count:04d}_{int(now)}.png")
        cv2.imwrite(fname, img)
        
        # 每 10 张报告一次
        if count % 10 == 0:
            print(f"  已截图 {count} 张... 按 Q 停止")
        else:
            print(f"  #{count:04d} {img.shape[1]}x{img.shape[0]} saved")


def main():
    parser = argparse.ArgumentParser(description="自动截图工具")
    parser.add_argument("--backend", choices=["mss", "win32"], default="mss",
                        help="截图方式: mss(默认,兼容性好) / win32(某些窗口可后台截图)")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="截图间隔(秒), 默认 1.0")
    parser.add_argument("--output", type=str, default="raw_auto",
                        help="输出目录, 默认 raw_auto")
    parser.add_argument("--window", type=str, default="腾讯欢乐斗地主",
                        help="窗口标题, 默认'腾讯欢乐斗地主'")
    parser.add_argument("--offset-x", type=int, default=0,
                        help="X方向偏移(像素), 默认0")
    parser.add_argument("--offset-y", type=int, default=0,
                        help="Y方向偏移(像素), 默认0")
    
    args = parser.parse_args()
    
    auto_capture(
        backend=args.backend,
        interval=args.interval,
        output_dir=args.output,
        window_title=args.window,
        offset_x=args.offset_x,
        offset_y=args.offset_y
    )


if __name__ == "__main__":
    main()
