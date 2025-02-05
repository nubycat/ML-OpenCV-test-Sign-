import cv2
import numpy as np
import imutils
import easyocr
from matplotlib import pyplot as plt

img = cv2.imread("imges/cars_5.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img_filter = cv2.bilateralFilter(gray, 11, 15, 15)
edges = cv2.Canny(img_filter, 30, 200)

cont = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cont = imutils.grab_contours(cont)
cont = sorted(cont, key=cv2.contourArea, reverse=True)

pos = None
for c in cont:
    approx = cv2.approxPolyDP(c, 10, True)

    if len(approx) == 4:
        pos = approx
        break


mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [pos], 0, 255, -1)
bitwise_img = cv2.bitwise_and(img, img, mask=mask)

(x, y) = np.where(mask == 255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropp = gray[x1:x2, y1:y2]

text = easyocr.Reader(["en"])
text = text.readtext(cropp)

if not text:
    print("Ошибка: EasyOCR не смог распознать номер!")
    exit()

res = text[0][-2]

# рамка вокруг номера (с отступом 20px)
final_image = cv2.rectangle(
    img,
    (y1 - 20, x1 - 20),  # верхний левый угол
    (y2 + 20, x2 + 20),  # нижний правый угол
    (0, 255, 0),
    4,
)

# отображение текста (смещение вниз)
text_x = y1 - 20  # Привязан к левому краю рамки
text_y = x2 + 100  # Смещен ниже нижнего края рамки (x2 + 40)

final_image = cv2.putText(
    img,
    res,
    (text_x, text_y),
    cv2.FONT_HERSHEY_PLAIN,
    5,
    (0, 0, 255),
    6,
)


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Car")
# plt.axis("off")
plt.show()
