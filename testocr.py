import easyocr, cv2, numpy as np
reader = easyocr.Reader(['en'], gpu=False)

img = cv2.imread("region_my_hand.png")
# 只取上半部分，避免底部"炸"字干扰
roi = img[:80, :]
results = reader.readtext(roi, detail=1)

for r in results:
    bbox, text, conf = r
    x = min(p[0] for p in bbox)
    print(f"x={x:.0f}  text='{text}'  conf={conf:.2f}")