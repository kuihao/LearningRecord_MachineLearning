"""
Reference: 
    * https://shengyu7697.github.io/blog/2019/11/29/Python-OpenCV-camera/
    * https://shengyu7697.github.io/blog/2019/11/28/Python-install-opencv/
Tips:
    * 擷取攝影機影像，需要先建立一個 VideoCapture，語法〔cv2.VideoCapture(0)〕，
      若只有一台攝影機則代號給 0
    * 使用語法〔cap.isOpened()〕確認攝影機裝置是否開啟
    * 用迴圈使用語法〔cap.read()〕每次從攝影機讀取一張影像
    * 影像處理：
        * ［cv2.cvtColor()］將影像從彩色轉成灰階
        * ［cv2.imshow()］ 將影像顯示出來
        * ［cv2.waitKey(1)］等待按鍵事件發生，按下 q 鍵則 break
    * 最後要使用［release()］釋放該攝影機裝置
"""
# import cv2
from cv2 import cv2 # In VScode use this code

<<<<<<< HEAD
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
=======
cap = cv2.VideoCapture(1)
>>>>>>> 2b009985213951b3356718271058904f8168eef3
if not cap.isOpened():
    print("Cannot open camera")
    exit()

<<<<<<< HEAD
print("fin")

=======
>>>>>>> 2b009985213951b3356718271058904f8168eef3
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # color to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 顯示圖片
    cv2.imshow('frame', frame)
    # 按下 q 鍵離開迴圈
    if cv2.waitKey(1) == ord('q'):
        break

# 釋放該攝影機裝置
cap.release()
cv2.destroyAllWindows()