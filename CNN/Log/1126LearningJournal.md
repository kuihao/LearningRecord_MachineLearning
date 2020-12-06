# ML Note: “Hello world” of deep learning
Date: 2020/11/26
Src: https://www.youtube.com/watch?v=Lx3l4lOrquw
Src: https://youtu.be/5BJDJd-dzzg

----
1. Tensorflow、Theano 皆可想成微分器，然而他們是比較 Flexible Toolkit，學習較為複雜，因此上課改教能快速上的 Keras

2. Keras 其實就是 Tensorflow 和 Theano 的 Interface，現在 Keras 也加入成為 Tensorflow 的 API，並且 Keras 已有許多現成的 Network Structure、Function 可以使用，想要重新自定義也沒問題，靈活性非常高

3. Data Info
    * 資料集選用**數字 1~10 手寫辨識**資料集
    * Dataset: **THE MNIST DATABASE of handwritten digits**
        > Src: http://yann.lecun.com/exdb/mnist/ <br>
        > 亦可用 Keras 直接下載： http://keras.io/datasets/

4. 導入
    ```
    from keras.models import Sequential
    ```

5. Implement Network 
    * Step 1： Define a Set of Function
        * 建立一個 Model
            ```
            model = Sequential()
            ```
        * 加入一個 Fully Connected Layer
            ```
            model.add(Dense(input_dim=28 * 28, units=500, activation='sigmoid'))
            ```
            * Dense 就是使用 Fully Connected Layer 的連接方式，若要使用 Convolution Layer 則是使用 Convolution 2D
            * 第一層 Layer 需要給定 input_dim 及 units
            * Keras 也有別的 Activation Function，例如 Softplus、Softsign、Relu、Tanh、Hard_sigmoid、Linear
        * 加入第二層 Layer
            ```
            model.add(Dense(units=500, activation='sigmoid'))
            ```
            * 第二層之後的新增只要給予 units 即可
        * 最後加入 Output Layer
            ```
            model.add(Dense(units=10, activation='softmax'))
            ```
            * 要辨識阿拉伯數字 1~10，因此 Output 維度是 10
            * 此處將 Output Layer 視為 Multi-class classifier，因此 Activation Function 選擇 Softmax
    * Step 2： Goodness of Function
        * 定義 Loss Function
        * 採用 Cross Entropy 則參數為 **categorical_crossentropy**
            ```
            model.compile(loss='categorical_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'])
            ```
    * Step 3： Pick the Best Function
        1. Configuration
            * 決定 Gradient Descent 的方式
            * 大部分方法可以不用預設 Learning Rate
            * Keras 已有 SGD、RMSprop、Adagrad、Adadelta、Adam、Adamax、Nadam
            ```
            model.compile(loss='categorical crossentropy',
                            optimizer='adam',
                            metrics=['accuracy'])
            ```
        2. Find the Optimal Network Parameters
            ```
            model.fit(x_train, y_train, batch_size = 100, nb_epoch = 20)
            ```
            * x_train: 
                > Train Data -- Image
            * y_train:
                > Labels (Digits)
            * batch_size:
                > 設定 Mini-Batch Size，若此值設 1 相當於使用 SGD
            * nb_epoch:
                > 對所有 Batchs 要迭代執行的次數
6. We don't really minimized total loss! 在實作數字辨識時，由於可以使用 GPU 加速，若 Mini-Batch Size = 1 則相同時間的運算效能並不會比 Mini-Batch Size = 20 來得好，而 Minimal Size 是多少可以從 Size 與時間之變化找尋

7. Batch Size 不該設太大，因為
    * 硬體資源有限，無限大 (過大) 的資料量無法進行平行運算
    * Batch Size 越大，則理論上的 Gradient Descent 下降曲線就越平滑，然而遇到平坦高原時就容易卡住，Batch Size 越小則 Gradient 的隨機性越大，越不容易卡住
    * 違反使用 MSGD 的初衷，Batch Size 小才能夠用相同的時間內更新多次 Weight

8. Testing
    * Case 1: Evaluation (訓練時，依 Loss值 及 Accuracy 判別好壞)
        * 當 Input Data 有 Label 或正解，則函式 **model.evaluate** 可以自動計算正確率
        ```
        score = model.evaluate(x_test,y_test)
        print('Total loss on Testing Set:',score[0])
        print('Accuracy of Testing Set:',score[1])
        ```
    * Case 2: Prediction (Model 正式上線進行預測，無標準答案)
        * 當 Input Data 無 Label 時，函式 **model.predict** 可以直接輸出預測結果
        ```
        result = model.predict(x_test)
        ```



