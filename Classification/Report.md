# Report of Classification (客戶年收入預測)
題目：依據金融客戶的個資 (背景資料)，預測分類該用戶的年收入是否大於 50,000 美元

----
實驗報告：
1. **Q：實作的 Generative Model 及 Logistic Regression 的準確率，何者較佳？Why？**
    <br>
    **A：**
    <br>［方法 / 預測正確率］

        Logistic Regression / Training accuracy = 0.88470
        Generative Model / Training accuracy =  0.86932
    **小結：如同上課的解說「一般而言 Logistic Regression 的結果會比 Generative Model 來得好」，實驗結果亦如此，由於 Generative 的方法是對資料假設一個機率分布所做的抽樣，並不會完全反映 Input Data 的情形，相較之下 Logistic Regression 則是直接針對資料計算 Posterior，直接受 Input Data 影響。** (不過本次尚未將資料傳至課程 Kaggle 驗證，往後會學習如何使用 Kaggle)

2. **Q：實作 Logistic Regression 的正規化 (Regularization)，並討論其對於模型準確率的影響。接著嘗試對正規項使用不同的權重 (lambda)，並討論其影響。(有關 regularization 請參考 https://goo.gl/SSWGhf p.35)**
    <br>
    **A：** *(目前 Regularization 遇到些困難，原欲嘗試直接對 Regularization Loss Function 計算微分，但似乎基礎理論本質上是不同的。網路上林軒田老師有針對 Regularization 進行講授，之後會補足這方面的知識)*

3. **Q：Best Model 訓練方式和準確率如何？**
    <br>
    **A：**
    <br>**最佳 Logistic Regression 訓練方式** 
    * 迭代次數： 200
    * batch size： 128
    * Gradient Descent 方法：AdaGrad
    * learning_rate： 0.1
    * 正確率 (Training accuracy)： 0.88556

4. **Q：請實作輸入特徵標準化 (Feature Normalization) 說明對於模型有何影響。**
    <br>
    **A：**
    **實驗結果為未經標準化的資料，其正確率下降至 0.88255；並且觀察 Loss 值的變化發現，未經標準化的資料其 Loss 值震盪非常劇烈，參數收斂較為困難**
    <br> 
    **有標準化的資料**
    <br>![Loss Graphic of Normalization](https://github.com/kuihao/Learning-record__Machine-learning/blob/main/Classification/Nomalized.jpg "Loss Graphic of Normalization")
    <br> 
    **未標準化的資料**
    <br>![Loss Graphic of Non-Normalization](https://github.com/kuihao/Learning-record__Machine-learning/blob/main/Classification/NonNomarlization.jpg "Loss Graphic of Non-Normalization")