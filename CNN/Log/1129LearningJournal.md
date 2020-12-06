# Convolutional Neural network(part 1)
Date: 2020/11/29
Src: https://sakura-gh.github.io/ML-notes/ML-notes-html/11_Convolutional-Neural-Network-part1.html

----
1. Why 使用 CNN 作影像辨識？
    * 使用一般的 Neural Network、Fully-connected，會導致參數過於龐大。假設一張 Picture 是 100x100 pixels，以彩圖而言每個 Pixel 就要存 3 個 values (RGB)，如此一來就有 3 萬維度的參數了；若 Hidden Layer 有 1000 個 Neurons，那僅僅是第一層 Hidden Layer 的參數就高達 30,000 * 1,000 個。
    * 希望每個 Neuron，就是一個最基本的 Classifier，每加一層 Layer 就分別更困難、複雜的事情，例如針對圖片第一層只分辨顏色、第二層分辨紋理或形狀、第三層區分物品種類等等。
    * 雖然 CNN 貌似運作複雜，但事實上 CNN 模型是比 DNN 還要簡單，就是用 Prior Knowledge 將 Fully-connected Layer 的一些參數去掉，DNN 就變成 CNN

2. Three Property for CNN Theory Base
    1. Some patterns are much smaller than the whole image
        * 以影像處理為例，第一層 Hidden Layer 的 Neurons 是偵測有沒有特定的 Pattern (圖案) 出現，若 Pattern 存在則不必看完整張圖
    2. The same patterns appear in different regions
        * 相同的 Pattern 可能會出現在不同圖片的不同位置上，例如辨識鳥喙時，每張圖片鳥喙的位置可能不同，但都是相同的形狀，因此判別特定 Pattern 的 Neuron 應設法重複使用，共用相同參數，以此降低參數數量
    3. Subsampling the pixels will not change the object
        * 我們可以對同張 Image 做 Subsampling (二次抽樣) 以降低 Data Size，例如將圖片奇數行、偶數列的 Pixels 去掉，Image 大小就變成原來的十分之一，且並不太會影響人類對這張 Image 的理解

3. The Whole CNN Structure
    * 首先要制定 CNN 的架構，也就是先決定好 Convolution Layer 及 Max Pooling 的次數，將 Input Datas 迭代反覆通過 Convolution Layer 及 Max Pooling，接著再經過 Flatten 程序，最後將 Outputs 丟到普通的 Fully-connected Structure 即可，這就是 CNN Structure

4. Convolution 的時作細節說明
    * 假設 Input Image 為 6 * 6，黑白色 (每個 Pixel 只需用一個 Value 儲存)
    * Convolution Layer 是一堆 Filter，每一個 Filter 其實就是 Fully-connected Layer 的一個 Neuron
    * Property 1:
        * 每個 Filter 其實就是一個 Matrix，裡面每一個 Element 的值作為 Network 的 Parameters，如同以前計算 Neuron 的 Weight 和 Bias，這些值都是通過 Training Data 學出來的，而不是人去設計
        * 假設每個 Filter Size 是 3 * 3，意味著偵測一個 3 * 3 的 pattern，它不會去看整張 Image，而是只看 3 * 3 範圍內的 Pixels 就可以判斷 pattern 有沒有出現，這就是考慮Property 1 的方式
    * Property 2:
        * 第二種 Filter 是從 Image 的左上角開始，做一個 Slide Window。每次向右挪動一定的距離，這個距離就叫做 Stride，由自己設定。每次 Filter 停下的時候就跟 Image 中 3 * 3 的 Matrix 做一個內積 (相同位置的值相乘並累計求和)。
        * 假設 Stride = 1，那麼 Filter 每次移動一格，當它碰到 Image 最右邊的時候，就從下一行的最左邊開始重複進行上述操作，經過整個 Convolution Process，最終得到 4 * 4 Matrix (https://gitee.com/Sakura-gh/ML-notes/raw/master/img/filter1.png)
        * 同個 Pattern 出現在 Image 左上角的位置和左下角的位置，並不需要用到不同的 Filter，我們用同個 Filter 就可以偵測出來，這就考慮了 property 2
    * Feature Map:
        * Convolution 的 Layer 中不一樣的 Filter 會有不一樣的參數，計算完卷積之後會得到許多 4 * 4 Matrixs，這矩陣合稱 Feature Map (特徵映射)，有多少個 Filter，就有多少個映射後的 Image
        * 然而 CNN 處理不同 Scale 的 Pattern 比較困難，例如：形狀相同但大小不同的鳥喙，雖然都是鳥喙應用相同的參數處理但 CNN 無法自動辨識這個問題。不過可以在 CNN 前面，再接另外一個Network，讓它 Output 一些 Scalar，將 Image 的裡面的某些位置做旋轉、縮放，然後再丟到 CNN 裡面，這樣會得到比較好的 Performance
    * Colorful Image
        * 這時候 Filter 變成一個立方體，如果今天是 RGB 來儲存一個 Pixel 的話，Input 就是3 * 6 * 6，Filter 是 3 * 3 * 3
        * 計算 Convolution 時，把 Filter 的 9 個值跟 Image 裡面的 9 個值作內積，可以想像成 Filter 的每一層都分別跟 Image 的三層做內積，得到的也是一個三層的 Output，每個 Filter 同時考慮了不同顏色所代表的 Channel

5. Convolution Vs Fully-connected
    * Filter 是特殊的 Neuron
    * Feature Map 的 Output，其實就是 Hidden Layer 的 Neurons 的 Output
    * Convolution 雖然是計算 Filter Matrix 與 Image 的 Inner Product 並得到一個純數，其實相當於 DNN 時，將 Filter 與 Image 轉乘向量，經過線性映射之後得到一的純數，這兩件事情本質上是一樣的
    * 每個 Neuron 只檢測 Image 的部分區域：對應到 DNN 時，其實就是制定一種連接 Neuron 的方式，第一層連接到第二層某個 Neuron 時，只有某些 Layer 1 的 Neuron 有連上，而非 Fully-connected，這樣就相當於 Filter 只檢視部分的 Image 
    * Neuron 之間共享參數: 在 Convolution 中是透過 Stride 的設定來移動 Filter，並持續使用範圍內曾使用到的參數，達到共用參數的效果；其實對應到 DNN 就只是類似 Full-connected 時，Laayer 1 Neurons 不只連接到 Layer 2 Neuron 1，亦連接到 Layer 2 Neuron 2，重點就是 Neuron 的 Output 會被重複連接的現象，這就是 DNN 共享參數的原理

