# Tips for Deep Learning - 2
Date: 2020/12/03-04
Src: https://youtu.be/xki61j7z-30

----
1. Train 不好時，調整結構__壹：變更 Activation Function
    * 最原始的作法是使用 Sigomoid Function
    * 然而在手寫辨識上，結構加到 9、10 層的時候，Model 就 Train 壞了，這稱為 **Vanishing Gradient Problem (梯度消失)**。當結構疊很深時，靠近 Input 的地方，參數的 Gradient (對最後Loss Function 的微分) 是比較小的；而靠近 Output 的地方，Loss 的微分值會是比較大的。因此設定同樣 Learning Rate 的時候，靠近 Input的參數更新很慢 (參數幾乎還是random)；靠近 Output 的參數更新較快 (已經根據這些 Random 的結果找到一個 Local Minima，然後就 Converge(收斂))。
    * Sigomoid Function 做的事情就是把任意值都壓在 0~1 之間，並每通過一個 Sigomoid Function 則 Gradient 值的影響力就衰弱了一些；因此即使最初的參數有很大的變化，通過非常多層 Sigomoid Function 之後，最初的變化量的 Gradient 值對最終 Output 的改變是微乎其微的，因此難以透過迭代運算有效率地更新參數。

    * 修正提出 **ReLU (Rectified Linear Unit, 整流線性單元函數，又稱修正線性單元)**
        * ReLU 跟 Sigmoid function 相比，運算快很多
        * ReLU 的想法結合生物上的觀察 (Pengel)
        * 無窮多 Bias 不同的 Sigmoid Function 疊加之後也會變成 ReLU
        * ReLU 解決 Vanishing Gradient 的問題 (the most important thing)
            * ReLU 的 Output 不是 0，就是等於 Input
            * 當 Output = Input 的時候，Activation Function 就是 Linear；而 Output = 0 的 Neuron 對整個 Network 是沒有任何作用的，因此可以把它們從 Network 中去掉
            * 整個 Network 變成一個瘦長的 Neural Network
            * **使用 ReLU 的 Network 整體來說還是 Non-linear。**
            * 若只對 Input 做小小的改變，不改變 Neuron 的 Operation Region，Network 就是一個Linear；若對 Input 做比較大的改變，導致 Neuron Operation Region 被改變，比如從 Output = 0 轉變為 Output = Input，Network 整體上就變成 Non-linear，這裡的 Region 是指 Input z<0 和 Input z>0 的兩個範圍。
            * 當 Region 為 z>0 時，Gradient 值就是1；當 Region 處於 z<0 時，Gradient 值就是0；當 z 為 0 時，相當於把它從 Network 裡面拿掉
    * Maxout Network (ReLU 只是 Maxout 的特例)
        * 使 Network 自動學習自己的 Activation Function
        * 首先自定義為 Layer 的 Outputs 分組 (Group)，多少個 Neurons 要分一組需自行設定，接著從每個 Group 裡面挑選最大的值作此 Layer 的最終輸出。只要參數適當的話是可以產生 ReLU Function 的，但也可以生成更多變化。
        * Maxout 可以實現任何 Piecewise Linear Convex Activation Function (分段線性凸激活函数)
        * 當只選擇最大值的 Neuron 時，相當於把 Network 中沒選到的 Neurons 去掉，最後也是得到一個細長、刪去部分 Neurons 的 Linear Function Network (如同 ReLU)，因此就算概念上有 Max Pooling 的 Neuron，其實仍是可以作  Backpropagation
2. Train 不好時，調整結構__貳：變更 Adaptive Function
    * 此段內容助教課時已說明過