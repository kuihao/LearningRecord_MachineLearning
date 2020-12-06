# Backpropagation
Date: 2020/11/23-25

----
1. Backpropagation 就是計算大量參數比較有效率的 Gradient Descent 演算法，先下定義：
    * 令 θ 為 Model 所有的 weights 所組成的向量
        > **θ** is a vector of **weights**
    * 令 l 為 Model Output y 與 y^head 之間的 Cross Entropy，假設 Output Layer 有 n 維，則 
        > **l_n(θ)** 表示第 n 組 Output 之 **Cross Entropy**
    * 令 L(θ) 為 Model 的 **Loss Function**，計算方式是加總所有 Output 之 **Cross Entropy**
        > L(θ) = ∑_n( l_n(θ) )
    * **Gradient：**
        > ∂L(θ)/∂w = ∑_n ( ∂l_n(θ)/∂w )
        * 欲求整體 Loss 值，只要能計算每組 Output 的 Gradient (偏微分)，再將其加總便是所求
2. 用 Chain Rule 解 Neural Network Loss Function 的 Gradient
    * 該如何計算 ∂l(θ)/∂w？ 
        * 令 *範例 Neuron* 是 z = Input_x * w + b，其中 z 就是 Neuron 的 Output
        * 依據 Chain Rule 則
            > ∂L(θ)/∂w = ∂z/∂w * ∂L(θ)/∂z

            > 前者 ∂z/∂w 稱為 **Forward Pass**

            > 後者 ∂L(θ)/∂z 稱為 **Backward Pass**
    * 感覺此方法類似 Dynamic Programming？
        > 更明確地來說這演算法就是 **Bottom-up 的遞迴演算法**，有無重複利用稍後待證 (有 Reuse 才是 DP)
    * **Forward Pass (輕鬆秒算)**
        > ∂z/∂w 的值就是所連接的 **Input_x 的值**
    * **Backward Pass (迭代)**
        * 假設 Activation Function 是 Sigmoid Function
        * 令 Sigmoid Function 為 σ()
        * 假設 *範例 Neuron* 的 Output 會連接 Activation Function
        * 令通過 Activation Function 的值為 a
            > σ(z) = a
        * 用 Chain Rule 拆解 ∂L(θ)/∂z
            > **∂L(θ)/∂z** = ∂a/∂z * ∂L(θ)/∂a

            > **∂a/∂z** = ∂σ(z)/∂z = 對 σ(z) 作微分 = **σ'(z)**
            > * 計算簡單
            > * z 應視為已知**常數**，z 就是 Neuron 的 Output，至少計算 Forward Pass 時便已算出
            > * **σ'(z) 亦可視為常數**
            
            > **∂L(θ)/∂a** = ∑_n( ∂z_n/∂a * ∂L(θ)/∂z_n )
            > * 其實就是一開始欲解的 ∂L(θ)/∂w 的樣態，以此無限遞迴下去
            > * 其中 Summation 裡面的項數就是該 Neuron 的 Output 維度數目
            > * 不過和 ∂L(θ)/∂w 不同的是此處已經拆到底層了 (Basis)，根據已知 a = σ(z) 已經見底不能再拆了，那就來實際計算吧～
            > ---
            > * 假設 *範例 Neuron* 的 Output Layer 只有兩維，則
            >   > ∂L(θ)/∂a = <br>
            >   > [∂z_1/∂a * ∂L(θ)/∂z_1] + <br>
            >   > [∂z_2/∂a * ∂L(θ)/∂z_2] <br>
            >
            > * 令 z_1 = w3 * a + ...
            >   > z_1 是 *範例 Neuron* 後連接第二層 Layer 裡的 Neuron 的 Output，a 作為此 Neuron 的輸入
            > * 令 z_2 = w4 * a + ...
            >   > z_2 也是 *範例 Neuron* 後連接第二層 Layer 裡的 Neuron 的 Output，a 也作為此 Neuron 的輸入
            > * **實際來解** ∂L(θ)/∂a 的 **∂z_1/∂a 和 ∂z_2/∂a**
            >   > 用 a 對 z_1 作微分，即針對 z_1 中含有 a 的項的 a 消去 <br> 顯然 ∂z_1/∂a = w3 <br> 同理 ∂z_2/∂a = w4 <br> 因此部分簡單秒算
            > * **實際來解** ∂L(θ)/∂a 的 **∂L(θ)/∂z_1 和 ∂L(θ)/∂z_2**
            >   > 用 Bottom-up Recursive 演算法即可
            >
            >   > **Basis Case:** <br> 當 ∂L(θ)/∂z_n 發生在 Output Layer (就是最後一層 Hidden Layer)， <br> 則 ∂L(θ)/∂z_n 就等於 ∂L(θ)/∂y_n，<br> 也就是 **∂∑_n( l_n(θ) )/∂y_n** <br> 可以直接秒算  
            >
            >   > **Recurrence:** <br> 若 Neuron 並非位於 Output Layer <br> 則遞迴解 ∂L(θ)/∂a_n
            > 
            >   > 從底部開始到回去算比較容易，這個算法在 Algorithm 裡就是稱為 Backward Reasoning (Backward Chaining)
            >
        * 實務上先算出每層的 Forward Pass 之後，只要用同個 Network 架構，但是將 Neuron 連接方向反轉，就能輕鬆計算 Backward Pass
3. 總結
    1. Neural Network 欲求 ∂L(θ)/∂w
    2. 先用 Forward Pass 秒算所有 Neurons 的 **∂z/∂w** 及 **a** (就是把 z 再通過 Activation Function)
    3. 再用 Backward Pass 對反轉的 Neural Network 計算出 **∂L(θ)/∂z**
    4. **∂z_n/∂w * ∂L(θ)/∂z_n** 便可得到特定 Output 的 ∂l_n(θ)/∂w
    5. 最後**加總**所有的 ∂l_n(θ)/∂w 便可得到 ∂L(θ)/∂w
            
        
    