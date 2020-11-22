Deep Learning Note
===
Date: 2020/11/22<br>
來源：https://www.youtube.com/watch?v=Dr-WRlEFefw&feature=youtu.be

---
1. History: 最初提出這種預測模型 Protectron 類似 Logistic Regression，後來經過演變，前後提出多接幾層 Hidden Layer 或改用「受限玻爾茲曼機」（英語：Restricted Boltzmann machine, RBM）於找尋 Greadient Descent 的 Initialization，同個方法改了名字稱為 Deep-learning，又發展了 GPU 加速運算。2011 年 Deep-learning 在語音辨識成效良好，2012 年 Deep-learning 技術在 ILSVRC Image Competition 獲勝引發關注。

2. Machine Learning 的第一步： Find a Function 其實就可以說是找一個類神經網路 (Neural Network)
    * 使用不同的連接方式來建構 Neural Network，得到不同的 Structure (Model)
    * 令 Θ 為任一個 Neural Network 裡所有 Neurons 排序而成的參數集 

3. 最簡單的架構就是 Fully Connected Feedforward Network
    * Fully Connected： 每一層之間都全連通
    * Feedforward： 層層之間的傳遞方向皆是後向前

4. Deep = Many Hidden Layers
    * DNN = Deep Neural Network
    * 目前已經可以這樣認知，Neural Network = Deep-learning Method

5. (參考下圖) 整個 Neural Network 架構的運算其實可以很完美的轉換成「Matrix Operation (矩陣運算)」
    * Input Vector (Input Layer) 轉換成 Input Matrix (藍色矩陣)<br>
      每列表示不同筆輸入資料
    * 參數部分構成 Weight Matrix (黃色矩陣) 及 Bias Matrix (綠色矩陣)
    * 通過此 Linear Function 運算之後再交由 Sigoma 進行轉換運算 (正規化、或是使用 Softmax 數值強調函數)，而此 Sigoma 代表 Activation Function (以 Logistic Regression 為例是使用 Sigomoid Function)
    * 計算完畢之後便可得到 Layer 1 的 Output (藍色矩陣)
    * 使用矩陣表示法的好處是矩陣運算可以交付 GPU 進行運算，減輕 CPU 的負擔達到加速的效果
    <br>
    ![DNN turn in to Matrix](https://github.com/kuihao/Learning-record__Machine-learning/blob/main/CNN/Log/MatrixOperation.jpg "DNN turn in to Matrix")

6. 最終整個 Classifiction Neural Network 架構我們可以簡化為以下幾個部分：
    1. Input Vector (Input Layer)：輸入的資料集
    2. Hidden Layer：可以視為 Feature Extractor (特徵萃取器)，也就是這一連串的 Logistic Regression 之複雜計算只是為了特化、萃取、轉化成一組特徵明顯的 Feacture Vector
    3. Output Layer：這並非輸出結果，而是架構的最後一層，此為 Multi-class Classifier，將 Hidden Layer 萃取出的 Feacture 輸入之後，能夠為每個特徵進行正確的分類，也就是分類器做為架構的最後一層。此處將 Output Layer 視為 Multi-class classifier，所以最后一個 Layer 會加上 Softmax Function。

7. 進展到 Deep-learning 的重點就是設計 Hidden Layer 的結構，好的結構 = 好的 Function Set = 容易得出好的結果
    * 要接多少層？
    * 要放幾個 Nuaron？
    * 順序怎麼接？
    ==> 透過經驗、試錯

8. 傳統的人工智慧重點在如何做特徵的擷取、特徵轉換，於 NLP、文字或文章語意辨識皆是如此，並有相當不錯的辨識效果；然而 Deep-learning Method 則是重點於 Model 的設計，用於語音辨識、影像辨識這種例子，人們本身不知道如何抽取「較好的特徵」，因此直接找尋最佳的分類模型反而有傑出的成效。

9. 是否能讓機器自己找出較好的結構？答：基因演算法

10. 目前有無自行設計、非 Fully Connected Feedforward Network 的結構？答：卷積神經網路（Convolutional Neural Network, CNN），是一種前饋神經網路，下回分曉。

11. Loss Function of Neural Network：根據每筆輸入的訓練資料計算出預測分類之後，將其與正解計算 Cross Entropy，接著便能調整 Neural Network 的參數，盡可能使 Cross Entropy Value 越小越好。

12. (接續 11.) Minimize 參數值的方法就是 Gradient Descent，計算參數梯度值並乘上 Learning Rate 再更新參數值，反覆迭代至收斂。
    * Backpropagation：對於 Neural Network 要計算 Gradient (計算微分) 較為複雜  ，Backpropagation 是一個有效率計算微分的方法。並且值得慶幸的是這個時代已有許多 Toolkit 可以為我們省下計算 Backpropagation 這個麻煩。