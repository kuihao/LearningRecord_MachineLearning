Pythorch learning note
===
Date: 2020/11/28<br>
Source：
> video: https://youtu.be/kQeezFrNoOg
> slide: http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML2020/PyTorch_Introduction.slides.html#/
> Article: https://medium.com/@auwit0205/pytorch-%E7%B0%A1%E6%98%93%E4%BB%8B%E7%B4%B9-45e25d3269b2

---
0. 學習目標
    * 這個微分套件的使用流程及原理
    * 如何疊 Active Function、Module
    * Pythrch 如何操作 Data Processing
    * Nvidia 套件 
1. numpy 與 tourch 資料結構的異同
2. Tensor.view 取代 np.reshape
    > Tensor (張量) 是一個可用來表示在一些向量、純量和其他張量之間的線性關係的多線性函數，這些線性關係的基本例子有內積、外積、線性映射以及笛卡兒積。---- from wikipedia
3. 廣播功能： 兩結構的維度不同時 Tourch 會自偵測擴增維度並計算
4. Gradient 產生示意圖
5. tourch.device：指定使用 CPU 或 GPU 做計算及移動
6. 自動計算微分的功能
7. Sequential 功能：可以連接一連串的 Neuron (不同的 function) 接再一起變成一個模組，可以循序運算
8. 如何用迴圈取得 Model 的所有參數
9. Loss Function 模組
10. Optimizor: 各種 Gradient Method

