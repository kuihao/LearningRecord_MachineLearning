# Tips for Deep Learning - 4
Date: 2020/12/06
Src-1: https://youtu.be/xki61j7z-30?t=4225
Src-2: https://youtu.be/Ky1ku1miDow

----
1. 當 Test 的結果不好時 (Overfitting) 的解法__參：**Dropout**
    * **使用時機 (最關鍵):** 原本的 Network 已經確定是 Overfitting，即 Training set 的 Accurancy 很高，但 Testing set 的 Accurancy 很低，此時使用 Dropout 才會有效果
    * 理論上的使用說明：自定義一個 Dropout Rate (通常是 0.5，若 Test Accurancy 差異過大 Rate 可以調升) 針對每一層 Layer Output 根據 Dropout Rate 做 Sampling (抽樣)。被 Sample 到的 Neurons 就被捨棄。此方法的目的和 Early Stopping 及 Regularization 類似，就是去掉一些 Neurons 使 Network 變得細長。
    * 每次更新參數後都要重新 Sample，因此每一輪更新的參數是不一樣的，這也相當於每次都 Train 出不同的 Network，特別的地方是這些不同時間點的 Network 裡面位置對應的 Neuron 仍是同一顆，也就是每個 Networks 之間有共用參數。
    * 最後 Train 出來的 Networks，其 Testing set 的 Accurancy 會下降，因為 Dropout 就是要讓 Network 有些偏離，然而在 Testing set 的表現就會變好。
    * 接著在使用 Testing set 時不使用 Dropout，而是將每個參數乘上 (1 - Dropout Rate)

2. 直觀解釋 Dropout
    * 「被決定要 Dropout 掉的 Neurons」 可想像成無貢獻的冗員，而「留下來的 Neurons 才會更新參數」可想像成它們是積極員工，為了保持該部門的業績而奮力工作 (更新參數) 才能 Carry 冗員
    * 而最終進入 Testing 階段時，所有員工都依最後的能力表現上工，然而因為所有人都有提出貢獻，因此 Testing set 的表現會變好，這就是 Testing 不用做 Dropout 的原因

3. 為何 Testing 時 Weight 要改乘上 Dropout Rate?
    * 可以把 Dropout 想像成終極的 Ensemble，Ensemble 就是把 Training Set 抽樣成許多小 Sets，針對不同的小 Sets 都丟到不同的複雜 Model，最後把結果取平均當作最終輸出。課程提過「複雜的 Model 其 Bias 小 (有包含靶心)、Variance 大 (散布廣)」，若使用很多複雜 Models 做訓練，雖然個別 Variance 很大，但取平均後 Variance 就會小了。
    * 因此 Dropout 每次更新參數都 Sample 出不同的 Network，相當於最高比例的使用不同的複雜 Model 做訓練，因此稱為終極的 Ensemble。

4. Dropout 與 Ensemble 的差異
    * 這兩個方法的差異就是 Testing 時，Ensemble 是取平均值，但 Dropout 是給所有參數乘上 (1 - Dropout Rate)
    * 事實上，若假設一個非常簡單的 Linear Network，其實 Dropout 與 Ensemble 的對參數運算方式是相同的，因此若 Activation Function 不使用 Sigmoid 而是使用 ReLU 或 Maxout 這種使 Network 接近 Linear 的 Function，再搭配 Dropout 就會有更好的表現

5. 回憶 Vanishing Gradient Problem
    * 當 Network 很深的時候，更新參數累計運算至越接近 Output Layer 時，Gradient 值變得非常小，幾乎不改變，呈現收斂的狀況，但實際上最初的 Weight 還只是 Random 的狀態