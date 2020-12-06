# Tips for Deep Learning - 3
Date: 2020/12/03-05
Src: https://youtu.be/xki61j7z-30?t=3310

----
* 此處的 Test 是指 Validation Set 或 Pubic Testing Set
1. 當 Test 的結果不好時 (Overfitting) 的解法__壹：**Early Stopping**
    * 假設你知道 Testing Data 上 Loss 值的變化情況，你必定會在 Testing Set Loss 值最小的時候停下來，而不是在 Training Set Loss 值最小的時候停下來；但 Testing Set 實際上是未知的東西，所以我們需要使用 Validation Set 來替代 Testing Set 去找到比較接近理想停止值的時候
2. 當 Test 的結果不好時 (Overfitting) 的解法__貳：**Regularization**
    * **重新定義了 Loss Function**
    * L1 Regularization
        * 把 L2 Regularization 時的「參數都取平方再求和」改成「參數都取絕對值再求和」
        * 實際上計算微分時就給一個特製 Function，例如老師自定義一個 Sgn (Sign Function)，當參數值大於零則 Sgn() 輸出 1 (微分值為 1)，當參數值小於零則 Sgn() 輸出 -1 (微分值為 -1)，當這麼不巧參數值等於零則 Sgn() 輸出 0 即可
        * 每次更新參數的時候，就減去一個 ηλ sgn(w^t_i)。如果原本 w 是正的，更新後就會減一個  Positive 值讓參數變小；反之則加一個值讓參數變大；簡而言之就是讓參數的絕對值不斷趨近於 0
    * L2 Regularization
        * 作 L2 Norm (L2正规化)
        * 把 Model 參數集 θ 的每個參數都取平方再求和，再將這個算值加在原本的 Loss Function 後面
        * 計算微分時，其實就是得到原本的 w 乘上一個常數 Lambda (可以設為 0.001)，然後整理之後新的對 Loss Function 作 Gradient Descent 的參數更新的式子就產生
        * 直觀意義上，相當於每次更新參數時就不分青紅皂白直接給原始參數乘上一個 0.999 的值，經過反覆迭代 w 就越來越趨近零 (因為常數設為小數)，但不會真的變成零，因為後面還有減一個微分值，最後會取得平衡
    * L1 VS. L2
        * L1 是更新參數時都減去一個固定值
        * L2 是更新參數時都乘上一個**小於 1 的**固定值
        * 參數的絕對值比較大的時候，L2 的參數會下降得快，並且由於 L1 每次減去一個固定值，故 L1 Train 完以後還會留有很多較大的參數
        * 參數的絕對值比較小的時候，L2 的參數會下降得慢，L2 Train 出來的參數平均都是比較小的，而 L1 Train 出來的參數是比較 Sparse，L1 參數會同時存在接近 0 的值及很大的值
        * CNN 的 Task 比較適合用 L1
    * 在 Deep Learning 中，Regularization 和 Early Stoping 的效果差不多，Regularization 的重要性不是那麼強。如果是 SVM (Support Vector Machine, 支持向量機) 則是本身就融入了 Regularization
3. 當 Test 的結果不好時 (Overfitting) 的解法__參：**Dropout**
