Logistic Regression Note
===
Date: 2020/11/18<br>

---

1. Linear Regression V.S. Logistic Regression
    * step 1. 設定函數模型
        * Linear: f_w_b(x) output 可以是任意值 
        * Logistic: f_w_b(x) output 因為通過 sigomoid fun. 故範圍介於 0~1
    * step 2. 決定如何判斷好的模型
        * Linear: RMSE 計算每個預測值與正解的差距，取正數而開方，加總後算期望值(求平均)，又簡稱為 Square Error 方法
        * Logistic:<br>
        令 f(x_n) 為分類預測<br>
        令 y_n 為分類正解<br>
        則 Loss 是計算 f(x_n) 與 y_n 之間的 Cross Entropy 的最小值，值越小表示兩者越接近
    * step 3. 搭配 step 2 找出最好的模型(的參數)
        * Linear:<br>
        使用 Gradient Descent，其實和 Logistic 的式子是一樣的
        * Logistic:<br>
        也是使用 Gradient Descent<br>
        (更新的 w) = (舊 w) - η( Σ_n( -(y_n^head - f(x_n)) ) * x_i_n  )  

2. Logistic Regression 不使用 RMSE 作為 Loss 的原因是計算 Gradient Descent 時，微分後的剃度變化非常小，使參數更新的效率很差；採用 Cross Entropy 則參數更新變化大，效率佳

學習紀錄：https://youtu.be/hSXFuypLukA?t=1828