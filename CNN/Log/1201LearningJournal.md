# Tips for Deep Learning - 1
Date: 2020/12/01-02
Src: https://youtu.be/xki61j7z-30

----
1. Deep Learning 的三個步驟
    1. Define the Function Set (Model)
    2. Goodness of Function (Loss Function)
    3. Pick the Best Function (Optimization)

2. 修正 Performance 的步驟
    1. Good Results on Training Data？
        * **若 Training Set 的 Accurancy 不佳** 重新回到 Deep Learning 的三個步驟進行調整，試圖讓 **Training Set** 的結果提升；<br>
        否則**若 Training Set 的 Accurancy 是好的**，則前往下一步
        * 這個步驟是 Deep Learning 獨特的地方，如果是使用 K-nearest Neighbor 或 Decision Tree 這些非 Deep Learning 的方法則不會回頭檢查 Training Set 的正確率，因為其 Accurancy 必定是 100%，所以其實 Deep Learning 並不是那麼容易發生 Overfitting
    2. 檢查 Testing Set 的 Results
        * **若 Testing Set 的 Accurancy 不佳**，表示 **Overfitting**，同樣回到三步驟做調整，並且此時的調整**很可能導致 Training Set 的 Accurancy 不佳**，這時必須再回到三步驟做調整。
        * **若 Training Set 及 Testing Set 都有好的結果**，就可以正式使用這個 Model 了 --> 成功

3. 注意！不是每次 Results 差，就認定為 Overfitting，請檢查 Training Set 的情況
    * 今天 Overfitting 成立的前提是「Training Set 的 Performance 好，且 Testing Set 的 Performance 差」
    * 有可能 Training Set 的 Performance 就差，那麼 Testing Set 的 Performance 通常也差，這就是一開始就沒 Train 好，而非 Overfitting

4. Underfitting：參數 (層數) 不夠多，以至於 Model 的能力不足以解決該問題

5. 使用各種 Deep Learning 的方法時，必須清楚知道現在是在處理 Training Set 的 Performance 還是 Testing Set 的 Performance，不同的理論方法適用的時機點不同
    * 例如：Dropout 是用在 Training Set 表現好，但 Testing Set 表現差的時候 (Overfitting)，但如果一開始 Training Set 的表現就差，那使用 Dropout 只會越 Train 越差。