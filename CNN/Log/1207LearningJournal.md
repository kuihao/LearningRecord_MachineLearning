# ML Lecture 10: Convolutional Neural Network
Date: 2020/12/07
Src: https://youtu.be/FrKWiRv254g

----
1. Neural Network 偵測同樣的東西時，只要使用相同的 Detector (Neuron) 即可
2. 以 CNN 的視角來解讀 Neural Network:
    * 機器會自動學習每個 Filter 是長什麼樣子
    * Convolution: 將 Filter 都對圖片進行內積運算，並依據自定義的 Stride 位移量進行位移再進行內積，並對所有的 Filter 皆反覆執行這個動作。最後會過濾出一疊圖，稱為 Features Map，其中每層 Features 中最大值者就是該 Filter 想過濾出來的特徵 
    * 每個 Filter 其實就是減少許多 Weight 連接的 Connected Network
    * 計算 Convolution 時有兩個意義，一是去掉一些 Neuron 所連接 Weight (若以 Fully-connected 的觀點來看待架構的話)，二是兩個 Filters (Neurons) 之間強制規定有使用共用的參數
3. Max Pooling 只保留每層 Features Map 中最大值的 Neuron
    * 舉例：如此便能將 6x6 的 Image 變成 2x2 的 Image
    * **每層** Features Map 更正的說法較叫做 **深度**
4. Convolution + Max Pooling 得到新的、比較小的 Image
    * Convolution + Max Pooling 可以不斷重複執行
    * 重複執行，Filter 數量是不會變的，深度也不會變
5. Flatten
    * 將最後迭代執行多次 Convolution + Max Pooling 所得的小圖片拉直扔進 Feedforward Network 一切訓練工作就結束了

6. 分析黑盒子 (分析 Filter)
    * 假設
        > Input: 1 個 28\*28 的 Image (1\*28\*28)<br>
        > Filter Size: 3\*3<br>
        > Filter Number: 25 (以影像而言又可視為 25 的 Channels)<br>
        > Image 通過 Filter Output: 25\*26\*26 的 Matrixes<br>
    * 第一層 Filter 就是線性函數，把原圖的部分 Pixels 挑出
    * 然而第二層 Filter 以後事情就複雜了，第二層 Filter 是拿原本挑出的 Pixels 再進行過濾<br>
    所以實際上第二層 Filter ((25+25)\*11\*11) 所看過的 Pixels 範圍有包含第一層 (25\*11\*11)。
    * Gradient Descent 反轉: 之前是固定 Input 用 Gradient Descent 找參數；現在想分析 Input (Filter) 為何，因此是固定參數並用 Gradient Ascent 找 Input。
    * 最後將最 Activation 的部分 Image 還原之後，此例會看到每個 Filter 最 Active (Activation 值越大，表示該部分圖片越是該 Filter 想過濾的 Pattern) 的圖案是各種線條。
    * 使用相同方法就能查看 任意層的 Filter (較大的 Pattern) 或 Flattern 之後扔進 Fully-connection Network 的任一個 Neuron (更大的 Pattern) 或是查看任一個最終 Output (整個 Pattern 例如:完整的手寫數字)。<br><br>然而有趣的是，我們反推應算出來的圖案往往跟我們的原先預期差很多，例如此處反推的數字 1~8 的圖案都很像是電視機雜訊，但直接丟回 CNN 卻能被辨識為 1~8，顯然 CNN 學到的東西跟我們的認知是有落差的。
    * 當然我們會更想知道如何讓反算出來的圖案更接近我們想要的樣子，所以方法就是在使用 Gradient Descent/ Ascent 時，加上一些 Constraint (限制)，來把不可能是我們所要的圖片給過濾掉，這就是還原字體 (或創造新字體 (Generate))。而這個在 Gradient 時附加 Constraint 的想法，實作上就是 Regularization (正規化)，也就是實際上對 Loss Function 加上特定參數值再去做 Gradient 計算。
    * 此例的想法是圖片裡白色的部分相當於筆跡墨水的 Pattern，但應該只有少部分地方有塗白色才對，因此對 Activation Function 所檢測的 X_ij 取絕對值再加起來並附於 Activation Function 之中，意義就是雖然有些 X_ij 很 Active，但不是人類所要找的 X_ij (圖案)，而取決對值再相加其實就是 L1 Regularization。<br><br>最後會得到一個似乎比較像數字的圖片，當然也有其他方法能讓他生成更接近數字的東西。
    * 這個方法就是 Deep Dream 的精神，我們希望機器在看完圖片之後，能自動加上他所看到的東西，這樣能幫助我們分析它到底 Filter 了什麼、怎麼判斷。<br><br>我們輸入一張原圖至 CNN 之後，再對 CNN Hidden Layer 參數跨大化 (正值更正、負值更負)，這樣 CNN 就會更認為原圖是他每個 Filter 想要找的東西。
    * Deep Style: 進階版 Deep Dream，也就是改變畫風的例子。都是用同個 CNN，先把原圖丟進 CNN，並找出每個 Filter 的 Output 值，這作為改畫風圖片的 Content (內容)；把一個欲仿造畫風的圖 (例如吶喊) 丟進 CNN，並找出 Filter 之間的 Correlation 值，這作為改畫風圖片的 Style (風格)。<br>接著對 CNN 進行反推，設定此 CNN 的 Filters 的 Content 為原圖，Correlation 為欲仿 Style 的 Correlation 值，則找出來的圖片就會是帶有仿化風格改造後的原圖。
 
    
