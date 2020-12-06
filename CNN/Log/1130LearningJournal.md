# Convolutional Neural Network part2
Date: 2020/11/30-31
Src: https://sakura-gh.github.io/ML-notes/ML-notes-html/12_Convolutional-Neural-Network-part2.html

----
## This note can avoid you forget what is Deep-learning.

1. What does CNN learn?
    * What is intelligent? 試圖探究這個黑盒子究竟智能在哪？
        > 延續前一章 (https://sakura-gh.github.io/ML-notes/ML-notes-html/11_Convolutional-Neural-Network-part1.html) 所講述的例子：<br>
        > 分析第一個 Convolution 的 Filter 比較容易，每一個 Filter 就是 3 * 3 的 Matrix，它對應到 3 * 3 範圍內的 9 個 Pixel，所以只要看這個 Filter 的值，就可以知道它在 Detect 什麼東西<br>
        > 但是你比較沒有辦法想像第二層的 Filter 的行為，它們是 50 個同樣為 3 * 3 的 Filter，但是這些 Filter 的 Input 已不是 Pixels，而是計算 Convolution 之後又再作 Max pooling 的結果，因此 Filter 考慮的範圍實際上並不是 3 * 3 = 9 個值，而是一個長寬為 3 * 3，高為 25 的 Cubic，Filter 實際在 Image 上看到的範圍是遠大於 9 個 Pixel，所以你就算把它的Weight 拿出來，也不知道它在做什麼
    * What does filter do?
        > 分析一個 Filter 行為：<br>
        > 我們知道在第二個 Convolution Layer 裡面的 50 個 Filter，每一個 Filter 的 Output 是 11 * 11 的 Matrix，假設我們現在把第 k 個 Filter 的 Output 拿出來，這個 Matrix 裡的每一個 Element，我們叫它 a^k_ij，上標 k 表示這是第 k 個 Filter，下標 ij 表示它在這個Matrix 裡的第 i 個 Row，第 j 個 Column<br>
        > 接下來我們 Define 一個 a^k 為 Degree of the activation of the k-th filter，這個值表示現在的第 k 個 Filter，它有多被 Activate，有多被“啟動”，直觀來講就是描述現在 Input 的東西跟第 k 個 Filter 有多接近，它對 Filter 的激活程度有多少
        > 第 k 個 Filter 被啟動的 Degree ak 就定義成它與 Input 進行卷積所輸出的 Output 裡所有Element 的 Summation，就是這 11*11 的 Output Matrix 裡所有元素之和<br>
        > 也就是說，我們 Input 一張 Image，然後把 Filter 和 Image 進行卷積所 Output 的 11 * 11 個值全部加起來，當作現在這個 Filter 被 Activate 的程度<br>
        > 接下來我們想要知道第 k 個 Filter 的作用是什麼，那我們就要找一張 Image，讓第 k 個Filter 被 Activate 的程度最大
    * What does neuron do?
        > 我們定義第 j 個 Neuron 的 Output 就是 a_j，接下來同樣用 Gradient Ascent 的方法去找一張 Image x，把它丟到 Neural Network 裡面就可以讓 a_j 的值被 Maximize
    * What about output?
        > 接下來我們考慮的是 CNN 的 Output，由於是手寫數字識別，因此 Output 是 10 維，我們把某一維拿出來，然後同樣去找一張 Image x，使這個維度的 Output 值最大<br>
        > 然而反推圖形之後發現，Neural Network 所學到的東西跟我們人類一般的想像認知是不一樣的。不過只要對 x 作一些 Regularization，即 Constraint(限制約束)，意義上是告訴 Machine，雖然有一些 x 可以讓 y 很大，但是它們不是數字。<br>
        > 這裡的 Constraint 最簡單的想法是畫圖的時候，假設圖片裡白色代表的是有墨水、有筆劃的地方，而對於一個 Digit 來說，整張 Image 上塗白的區域是有限的，整張圖都是白白的，它一定不會是數字。<br>
        > 假設 Image 裡的每一個 Pixel 都用 x i j 表示，我們把所有 Pixel 值取絕對值並求和，也就是 ∑_ij |x_ij|，這一項其實就是之前提到過的 **L1 的 Regularization**，再用 y_i 減去這一項。<br>
        > 這次我們希望再找一個 Input x，它可以讓 y_i 最大的同時，也要讓 |x_ij | 的 Summation 越小越好，也就是說我們希望找出來的 Image，大部分的地方是沒有塗顏色的，只有少數數字筆劃在的地方才有顏色出現。加上這個 Constraint 得到的結果隱約有些可以看出來是數字的形狀了。如果再加上一些額外的 Constraint，比如你希望相鄰的 Pixel 是同樣的顏色等等，你應該可以得到更好的結果
2. Deep Dream
    * 其實這就是 Deep Dream 的精神：如果給 Machine 一張 Image，它會在這個 Image 裡面加上它看到的東西
    * Regularization: Make CNN exaggerates what it has seen
3. Deep Style
    * Deep Dream 還有一個進階的版本，就叫做 Deep Style：Input 一張 Image，Deep Style 做的事情就是讓 Machine 去修改這張圖，讓它有另外一張圖的風格
    * Filter 和 Filter Output 之間的 Correlation 代表了一張 Image 的 Style
4. More Application——Playing Go
    * What does CNN do in Playing Go
        > Input 是棋盤當前局勢，Output是你下一步根據這個棋盤的盤勢而應該落子的位置，這樣其實就可以讓 Machine學會下圍棋了，所以用 Fully-connecte d 的Feedforward Network 可以做到讓Machine 下圍棋這件事情<br>
        > 但實際上如果我們採用 CNN 的話，會得到更好的 Performance，棋盤其實可以很自然地表示成一個 19 * 19 的 Matrix，對 CNN 來說就是一個 Image
    * Why CNN for Playing Go
        > 什麼時候我們可以用 CNN 呢？首要條件就是要有 Image 的特性，CNN 所提到的三個 Property 就是依據 Image 的特性所建構的<br>
        > 不過在圍棋這項應用上，第三項 Property 縮小 Image 比較難想像，圍棋真的能使用 Max Pooling 縮小圖片嗎？<br>
        > 其實 Alpha Go 沒有使用 Max Pooling，不過 Alpha Go 把每一個位置都用 48 個 value 來描述，業就是額外加上 Domain Knowledge，除了檢查黑子白子，也觀察這個位置是不是處於叫吃的狀態等等<br>
        > AlphaGo 的 Network Structure 一直在用 Convolution，根本就沒有使用 Max Pooling，原因是圍棋的特性不需要使用 Max pooling 這樣的構架
5. More Application——Speech、Text
    * Speech
        > 將聲音轉成頻譜 (Spectrogram) 也就是圖片即可，比較神奇的地方是通常只考慮在 Frequency(頻率) 方向上移動的 Filter 而非時間軸上移動。這是因為在語音裡面，CNN 的 Output 後面都還會再接別的東西，比如接 LSTM 之類，它們都已經有考慮 Typical 的 Information，所以再考慮一次時間其實沒有什麼幫助<br>
        > 相同的單字，即使聲線不同，但頻率的 Pattern 其實是一樣的，它們的差別可能只是所在的頻率範圍不同而已，因此考慮 Frequency 是有效的
        * 當你把 CNN 用在一個 Application 的時候，永遠要想一想這個 Application 的特性是什麼，根據這個特性你再去 Design Network 的 Structure，才會真正在理解的基礎上去解決問題
    * Text
        > 假設 Input 是 Word Sequence，你要做的事情是讓 Machine 偵測這個 Word Sequence 代表的意思是 Positive 還是 Negative<br>
        > 首先 Word Sequence 的每一個 Word 都用一個 Vector 表示，Vector 代表這個 Word 本身的 Semantic (語義)，如果兩個 Word 本身含義越接近的話，它們的 Vector 在高維的空間上就越接近，這個東西就叫做 **Word Embedding**<br>
        > 把一個 Sentence 裡面所有 Vector 排在一起，它就變成了一張 Image<br>
        > **文字處理時，Filter 只在時間的序列 (按照word的順序) 上移動，而不在這個 Embedding 的 Dimension 上移動**；因為 Word Eembedding 不同 Dimension 是 Independent，不會出現有兩個相同 Pattern 的情況，所以在這個方向上面移動 Filter 是沒有意義的
6. Conclusion
 * CNN 的三個屬性是
    * Some patterns are much smaller than the whole image. —— property 1
    * The same patterns appear in different regions. —— property 2
    * Subsampling the pixels will not change the object. —— property 3
 * 針對三個屬性的架構是
    * Convolution：針對 property 1 和 property 2
    * Max pooling：針對 property 3
 * 但使用 CNN 最重要的是事情是針對不同的 Application 要設計符合它特性的 Network Structure