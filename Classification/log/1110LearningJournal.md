Improvement of Classifiction
===
2020/11/10-13

---
1. Modifying Model
    * 通常 Gaussian 會使用共同的 Covariance matrix 或 mean<br>
    * 可以嘗試讓不同的類別 (Class) 採用 (Share) 相同的 Covariance matrix<br>
    * 其中 mean 對應的是分布的中心位置，Covariance 則是散布程度，<br>
    * 且 Covariance 和輸入資料的平方成正比，若類別有不同的 Covariance 會使 Function 過於複雜而導致 Overfitting<br>
1. Likelihood Change<br>
    * Likeli(μ\*_1, μ\*_2, ∑\*) = GauDis_μ1∑(x1) * GauDis_μ1∑(x2) \*...\* GauDis_μ2∑(x80) * ...<br>
    * 其中 μ\*_1, μ\*_2 算法不變<br>
    * 而 ∑\* 要改成：∑\* = (N(C1)/n)\*∑\*1 + (N(C2)/n)\*∑\*2
    * 視覺化呈現：未共用 Covariance 時，還樹呈現曲線；使用 Covariance 後，函數呈現直線，並可稱為 Linear Model

1. 該選哪種機率分布模型？<br>
    * A: 依情況判斷<br>
    * 例如：*Naive Bayes Classfier* 假設每個分類都是獨立事件，則 P(C1|x) = P(x1|C1) * P(x2|C1) * P(x3|C1) * ...

1. Posterior Probility
    * 令 z 為 ln( P(x|C1)P(x) / P(x|C2)P(C2) )<br>
    * 令 σ(z) 為 Sigmoid Function<br>
    * 令 Posterior Probility 為 P(C1|x)<br>
    * 則簡化公式為：
> P(C1|x)<br> 
> = P(x|C1) / [P(x|C1)P(x) + P(x|C2)P(C2)]<br>
> = 1 / [1 + P(x|C1)P(x) / P(x|C2)P(C2)]<br>
> = 1 / [1 + exp(-z)]<br>
> = σ(z)

> 其中 z 的輸入值剛好就是使用分類模型的輸入值<br>
>  if z = 0, then σ(z) = 0.5<br>
>  if z > 0, then σ(z) >> Infinite large<br>
>  if z < 0, then σ(z) =  Infinite simals<br>

5. 推導 Sigmoid Function 
* 終極目標是計算 P(C1|x)
* 令 σ(z) 為 P(C1|x) 的展開簡化
* 欲計算 z 則必須將 Mean 和 Covarience 代入高斯分布公式
* 若 Covarience 皆相同，則可進一步簡化
![Sigmoid derive 00](https://github.com/kuihao/Learning-record__Machine-learning/blob/main/Classification/log/sigmoid_0.jpg "Sigmoid derive 00")<br>

![Sigmoid derive 01](https://github.com/kuihao/Learning-record__Machine-learning/blob/main/Classification/log/sigmoid_1.jpg "Sigmoid derive 01")<br>

![Sigmoid derive 02](https://github.com/kuihao/Learning-record__Machine-learning/blob/main/Classification/log/sigmoid_2.jpg "Sigmoid derive 02")<br>

![Sigmoid derive 03](https://github.com/kuihao/Learning-record__Machine-learning/blob/main/Classification/log/sigmoid_3.jpg "Sigmoid derive 03")<br>
* 最後將含有 x 的項、常數項各自合併，簡化得知這裡的 z 其實也是 Linear 