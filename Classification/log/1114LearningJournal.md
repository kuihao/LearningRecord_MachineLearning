Logistic Regression
===
2020/11/14

---
* **Make a Funcion Set**<br>
令 P_w,b 為 σ(z)
令 z = w * x + b = Sum( w_i * x_i ) + b
> **f_w,b = sigoma(z) = P_w,b(C1|x)**<br>
> f 函式 output 的值就是 posterior probility 的分類預測機率<br>
> f 函式 output 的值就是 sigomoid(z)
* **Goodness of a function**
> 令 f_w,b = P_w,b(C1|x)<br>
> 嘗試不同的 w 與 b 以計算 N 筆資料輸入 f 之後，預測其所屬分類的機率，並試圖找可得到最大機率的那個 w, b<br>
> Loss(f_w_b(x)) = Loss(w, b) = 連乘f_w_b(x_i)...【如果答案是 C1 就乘f_w_b(x_i)；反之則乘 (1-f_w_b(x_i))，即 C2】<br>

> 為了方便計算，把求最大值改成求最小值：<br>
> 令 w* ,b* 為最佳解則 Loss 的表達式轉換為  
> w* ,b* = arg max L(w,b) = **arg min (-1) * ln(L(w,b))**

> 但因為這樣還是太難看了，希望能用 Summation 表達式，因此將符號做編碼轉換：<br>
> 令 y_n^head 為 訓練資料的真值<br>
> 令 Class 1 的 y_n^head = 1 <br>
> 令 Class 2 的 y_n^head = 0 <br>
> 令 C() 為 Cross entropy ... 就是 Loss 的精華，用來判別不同類別應留下的資料，就是兩個白努力分配之間的 Cross entropy<br>
> 若資料為 C1 則設其 **y_i^head = 1**，同理 C2 設為 **y_i^head = 2**

> 因此我們目前訓練資料的數對長這樣<br>
> (input, output) = (x_n,  y_n^head)

> 最終 Loss fun. 簡潔為<br>
> Loss(f(x)) = Sum( C(f(x_n), y_n^head) )<br>
> **最精華的部分就是簡化為 Cross Entropy**<br>
> 當 y_n^head = 0 或 1，都能剛好刪去表達式中的部分，保留所需要的最終結果<br>
> (如同 Linear Regression 要結合多種屬性代入不同函數斜率的 Muti-Linear regression 時的方法類似)<br>
> ![Loss function derive](https://github.com/kuihao/Learning-record__Machine-learning/blob/main/Classification/log/LogisticRegresssion.jpg "Loss function derive")<br>
    
    * Cross Entropy 就是在計算兩種分配有多接近，當兩種分配完全一致，其 Cross Entropy = 0

* **Find the best function**
    * 對 Loss function 作微分
    * Cross entropy 會比 Gradient descent 更容易、快速找到最佳解