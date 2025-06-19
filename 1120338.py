import cv2

# 載入模型
detector = cv2.CascadeClassifier("project/classifier/cascade.xml")

# 載入圖片
image = cv2.imread("project/test.jpg")
image = cv2.resize(image, (800, 600))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 偵測（更嚴格）
detections = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

print(f"偵測到 {len(detections)} 個站牌")

# 畫框（只畫合理大小的）
for (x, y, w, h) in detections:
    if w * h < 500:  # 避免畫出雜訊小框
        continue
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)

cv2.imshow("result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
