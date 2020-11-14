# 2020/11/04-10
來源：https://github.com/Sakura-gh/ML-notes/blob/master/ML-notes-md/6_Classification.md

0.  Regression <b>不適用於分類</b>，因為<br>
    > (1) Regression 的輸出是連續資料，但分類問題的輸出是離散資料<br>
    > (2) 分類問題的輸出類別的值不應有大小之分，預測值與真值之間的距離差的比較是無意義的，因此 Regression 的 Loss Function 也不適用，「距離較遠<b>不代表</b>相差得更多」，分類問題類似<b>是非題</b>，屬於就屬於、不屬於就不屬於。
    
    > Classfiction 必須重新定義 Function Set 及 Loss Function，以下用二元分類問題為例：<br>
    > 類別有 Class 1 及 Class 2
0.  <b>制定 Model (Function Set):</b><br>
    * f(x) 的計算及意義
    > 本次使用機率模型 (**Generative Model**)<br>
    >> **目的**是給某個 **x 向量 (Features)** 作輸入，輸出 x **屬於某類別的機率**
    >
    >> **假設**每筆資料皆是抽樣 (Sample) 自*同一個(?)* **高斯分布模型 (Gaussian Distribution Model)**
    >> **核心想法**是根據 Training Data 分布找到最接近的分布模型，如此一來對於 **任意 x** 都能**從該分布模型生成 (產生) x 屬於該類別的機率**
    >
    > 令 Model 為 <b>f(x)</b>，且 f(x) 內含 g(x)，g(x) 用來將連續資料轉換成離散型態</b><br>
    >> If <b>g(x)>0.5,</b> then <b>f(x) = class 1</b><br>
    >> else if <b>g(x)<0.5,</b> then <b>f(x) = class 2</b><br>
    * g(x) 的計算及意義
    > 令 C1、C2 分別為 Class 1 及 Class 2 的發生事件<BR>
    > 令 <b>g(x) = P(C1|x)</b><br>
    > = P(x 交集 C1) / [P(x 交集 C1) + P(x 交集 C2)]<br>
    > = <b>[P(x|C1) * P(C1)] / [P(x|C1) * P(C1) + P(x|C2) * P(C2)]</b><br>
    > = <b>[P(x|C1) * P(C1)] / P(x)</b>
    >> <b>目標</b>是計算出 <b>P(C1|x)</b>，<b>x 屬於 C1 的機率</b>
    >>> 根據貝氏定理，「取 x 的條件之下，是從 C1 取出的機率，相當於「從所有 Class 中取出 x 的機率」的條件之下，是從「取 C1 的前提下取出 x 的機率」
    >>
    >>> 取 x 發生的前提下，若 x 和 C1 都發生的機率越大，<br>則 x 越有可能屬於 C1 類別
    >
    >> 欲計算 P(C1|x) 則需要計算 Prior (用古典機率), Likelihood (從高斯分布進行抽樣)<br>
    >> * Prior: P(C1), P(C2)<br>
    >> > P(C1) 意義：從特定的高斯分布抽樣出 C1 的機率<br>
    >>> 計算方法： ex. P(C1) = N(C1) / n
    >>
    >> * Likelihood: P(x|C1), P(x|C2)
    >>> P(x|C1) 意義：從 Class 1 抽樣出 x 的機率<br>
    >>> 計算方法：假設不同類別的 Traning Data 其實皆來自某個高斯分布所抽樣而來，因此可以運用利用 **Generative Model** 來產生(抽樣) P(x|C1) 及 P(x|C2)
    >>
    >> Generative Model: P(x)
    >>> 意義：從**特定的高斯分布**抽樣出 P(x) 的機率<br>
    >>> 其中：P(x) =  P(x|C1)P(C1) + P(x|C2)P(C2)
    >>
    >>> 計算方法：**找出高斯分布 (Gaussian Distribution)**
    >>
    >> Gaussian Distribution
    >>> 意義：欲計算 Likelihood 要透過高斯分布，高斯分布的曲線是根據 mean (μ), covariance (∑) 而有不同的變化<br>
    >>> 計算方法：<br>
    >>> 令 GauDis_μ∑(x) 為高斯分布的機率密度函數<br>
    >>> 則 GauDis_μ∑(x) = 
    >>> ![Gaussian Distribution Formula Solution](https://github.com/kuihao/Learning-record__Machine-learning/blob/main/Classification/log/%E9%AB%98%E6%96%AF%E5%88%86%E5%B8%83%E5%85%AC%E5%BC%8F.jpg "Gaussian Distribution Formula Solution")<br>
    >>
    >>> 如何找 μ 及 ∑ ?<br>
    >>> 需要使用 Maximum Likelihood
    >>
    >> 用 **Maximum Likelihood** 找出好的 Gaussian Distribution
    >>> 理論上要窮舉所有 μ, ∑，但實際上只要求平均便可得到最大值
    >>> μ*, ∑* = arg max_μ∑ Like(μ, ∑)<br>
    >>> μ* = mean(x_n)<br>
    >>> ∑* = mean([x_n - μ*][x_n - μ*]^T)<br>
    >>
    >> **最終結果**<br>
    >>> 將 Class 1 的 μ\*_1, ∑\*_1 代入 GauDis(x) 得 P(x|C1)；對Class 2 同理則得 P(x|C2)<br>
    >>> 最後將所有參數代入 g(x) 再用 f(x) 判斷輸出便結束

0.  <b>判別 Function 的好壞 (Loss Function)</b><br>
    > 意義：訓練階段用 Validation Set 判斷 Function 是否準確<br>
    > if P(C1|x) > 0.5 then x 屬於 C1<br>
    > 若預測錯誤，則 Loss++

---    
* [補充 1] 如何解分類的問題？
    > <b>Function Set (Model):</b> 針對抽樣散布資料 (n 個 Featrues 所構成的 n 維散布) 使用貝氏定理計算其屬於 Class 1的機率
    
    > <b>Loss Function:</b> 
    > 訓練時，if 資料輸出的預測類別「錯誤」then Loss++<br>
    > 加總所有資料錯誤預測類別的次數

    1. 如何計算 Function Set?
    > 假設同一類別的資料的數值分布符合高斯分配，則只要用<b>訓練資料集</b>找到一個 Function 使其能足夠準確地「生成每一個散布點對應的機率值」，當輸入未見過的測試資料，就能算出其對應的機率值，並依機率值來判斷它是哪一個類別。
    
    > 既然假設資料存在一個完美的高斯分布，意思就是完美高斯分布「抽樣」出來的資料會等於(極近似)訓練資料的散布狀況   
    
    > (1) 計算 P(C1)、P(C2)，用古典機率算法： P(C1) = n(C1)/n(C1)+n(C2)，其中 n() 為計數個數的函數
    
    > (2) 為了找到某一點的機率，我們要用抽樣；為了要用抽樣，我們必須找到夠準的高斯分布。<br>
    找高斯分布的方法就是計算訓練資料的 mean (/Mu/) 及 covariance (/Sigma/)，以此可計算 P(x|C1)、P(x|C2)
    計算方法就是 Likelihood: L(mean, covariance) = 個別資料機率值的乘積 = 一個收斂型複雜公式<br>
    Max Likelihood: 用公式計算 meanㄝ, covariance
    
    > (3) 將(1)及(2)結合，便可算出 P(C1|x) 即「x 屬於 C1 的機率」，數學式直譯為「指定資料為 x 的前提下，是 C1 類別的機率」
    P(C1|x) = P(x|C1)P(C1)/P(x)，其中 P(x) 稱為 <b>Generative Function</b> = P(x|C1)P(C1)+P(x|C2)P(C2)
    用 Generative Function 便可計算任意輸入 x 的相應輸出的類別機率值

---
* [補充 2] 條件機率
    > 令 n 為事件發生的總次數 <br>
    > 令 N(A) 為 A 事件發生的次數 (**邊緣機率**是某個事件發生的機率。邊緣機率是這樣得到的：在聯合機率中，把最終結果中不需要的那些事件合併成其事件的全機率而消失，對離散隨機變量用求和得全機率，對連續隨機變量用積分得全機率，稱為**邊緣化 Marginalization**) <br>
    > 令 N(B) 為 B 事件發生的次數 <br>
    > 令 N(A∩B) 為 A 交集 B 的事件發生的次數 (**聯合機率**表示兩個事件共同發生的機率) <br>
    > 令 條件機率公式為 P(A|B) = P(A∩B) / P(B) <br>
    > 令 Likelihood 為 p(x|c)<br>
    > 令 Prior 為 p(x)<br>
    > 令 Posterior 為 p(c|x) 
  
    > <b>公式思考 1</b><br>
    > 在 n 次試驗中，根據中央極限定理：<br>
    > P(A|B) ~= (N(A∩B) / n) / (N(B) / n)<br>
    > = N(A∩B) / N(B)<br>

    > <b>公式思考 2</b><br> 
    > 利用排列組合的乘加原理，條件機率就如同連續事件，因此使用乘法原理<br>
    > ex. P(A∩B) = P(A) * P(B|A) = P(B|A) * P(A)

----
