Logistic Regression Note
===
Date: 2020/11/18-19<br>

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

3. Discriminative Model (Logistic Regression) 是直接找參數 (w, b)<br>
   Generative Model 是 mu_1, Mu_2, sigoma_inverse 再計算出 w^T 及 b

4. 其實 Discriminative Model (Logistic Regression) 和 Generative Model 是用<br>
    **同一個 Model(Function Set)** 即使用同一個 Finction Space，都是找參數 w 與 b<br>
    然而兩者找出的最佳解的參數 (w, b) 是**不一樣的**<br>
    因為兩者對找尋 Model 的假設不同，只有 Generative Method 有對 Probility Distribution 有假設 (設為高斯分布、簡單分布、白努力分布...)，而 Logistic Regression 則沒有作假設

5. 經常情況為 Discriminative Model 優於 Generative Model

6. Generative Model 是假設訓練資料、測試資料是抽樣自某個「機率模型(分布)」，也就是腦補完整的分布、完美的抽樣應為如何。也就是事實上，Discriminative Model 完全反映自訓練資料的情況作預測，預測受 input data 的影響大，而 Generative Model 會考慮一個假設的分布模型再作抽樣，其預測時受 Input data 的影響較小，Generative Model 的好處就是能把有些問題的資料忽視掉。

7. Discriminative Model 是直接計算 posterior probility，也就是直接假設 posterior 去找參數，而 Generative Model 則是還先找了 Formulation 前面的 prior 及 class-independent 的 probility。額外先找尋分拆的 prior 及 class-independent 的 probility 的**好處是 prior 及 class-independent 甚至能來自不同的來源！** <br>
    以語音辨識來說，語音辨識就是預測(分析)「某句話被說出的機率 ---> prior」，但其實訓練時並不需要語音，只要一堆字句、文章就能訓練出預測句子的 Model，這就是 Language model；而語音結合文字的部分則是 class-independent 的機率，這樣能使預測更準確。<br>
    語音辨識是 nuarl network 是 Discriminative (DNN?)，但其實整個語音辨識系統是 Generative 的 System，語音辨識是組起來的。

8. Muti-class Classification -- *Softmax*<br>
    函數通常會放在類神經網路的最後一層就是 Softmax，針對 test data x 計算了它在三個類別的預測機率 z1, z2, z3，並丟進 Softmax。<br>
    Softmax 函數裡面有兩大功能，一是對輸入值取 exp() 對 Max Value (最大值) 作強化，二是作 Normalization (Min-Max正規化)，最後出來的三個值就會皆於 0~1 並且差距區分很大。<br>
    Softmax 輸出的三個值可令為 y1, y2, y3 就視為 posterier probility。<br>
    *Why exp()? Bishop, p209~210*

9. 延伸 Maximum entropy 它也是 classifier，其實就是 Logistic Regression，但是從另一個觀點說明為什麼 classifier 長這樣。 目前本章是學習從高斯分布的機率模型來推導出 Softmax 這個 classifier 的 function。但 Maximum entropy 是從 information theory 觀點說明

10. 計算 Cross Entropy 就根據 Maxima likelyhood 去推導轉成找最小值的方法來找最好的 function，此方法亦適用 Muti-class Classification

11. Logistic Regression 的限制，當 Features 取得不好，分布狀態無法使用 Regression 區分，則需要 **Feature Transformation**，轉換方式很多種只要能順利把 Features 映射到能區分的空間即可，但**這個轉換希望能讓機器自己來做** --> 我們就把很多個 Logistic Regression 連接起來就能辦到了<br>
    可以第一層 Logistic Regressions 作 Feature Transformation，再把第一層輸出作為第二層 Logistic Regressions 的輸入作 Classification，達到機器自動 Feature Transformation 並完成 Classification。<br>
    同理，再不同的情境可能會接上第三層、第四層，然後把每一層的一個函數取名叫做 Nural，整個架構美稱其為 Nural Network 類神經網路，就是所謂 Deep Learning 深度學習。<br>
    其實只是多個 functions 彼此串接(?)

 