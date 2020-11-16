ML Homework: Classification (客戶年收入預測: 程式碼架構篇)
===
Date: 2020/11/16<br>
參考來源：https://colab.research.google.com/drive/1JaMKJU7hvnDoUfZjvUKzm9u-JLeX6B2C

---
## Logistic Regression
1. Preparing Data

        Load and normalize (正規化) data, and then split training data into training set (訓練集) and development set (發展集) .
2. Define Some Useful Functions

        定義在訓練迴圈中被重複使用的函數
    *   def _shuffle(X, Y)
                # This is the logistic regression function, parameterized by w and b
    
    *   def _sigmoid(z)
    *   def _f(X, w, b)

            This is the logistic regression function, parameterized by w and b
            Arguements:
                X: input data, shape = [batch_size, data_dimension]
                w: weight vector, shape = [data_dimension, ]
                b: bias, scalar
            Output:
                predicted probability of each row of X being positivel labeled, shape = [batch_size, ]
    *   def _predict(X, w, b)
            
            This function returns a truth value prediction for each row of X 
            by rounding the result of logistic regression function.
    *   def _accuracy(Y_pred, Y_label)

            This function calculates prediction accuracy
3. Functions about Gradient and Loss
    *   def _cross_entropy_loss(y_pred, Y_label)
    *   def _gradient(X, Y_label, w, b)
4. Training

        Everything is prepared, let's start training!

        Mini-batch gradient descent is used here, in which training data are split into several mini-batches and each batch is fed into the model sequentially for losses and gradients computation. Weights and bias are updated on a mini-batch basis.

        Once we have gone through the whole training set, the data have to be re-shuffled and mini-batch gradient desent has to be run on it again. We repeat such process until max number of iterations is reached.

5. Plotting Loss and accuracy curve
    > import matplotlib.pyplot as plt
    > 
    > \# *Loss curve*
    > plt.plot(train_loss)
    > plt.plot(dev_loss)
    > plt.title('Loss')
    > plt.legend(['train', 'dev'])
    > plt.savefig('loss.png')
    > plt.show()
    > 
    > \# *Accuracy curve*
    > plt.plot(train_acc)
    > plt.plot(dev_acc)
    > plt.title('Accuracy')
    > plt.legend(['train', 'dev'])
    > plt.savefig('acc.png')
    > plt.show()

6. Predicting testing labels

        Predictions are saved to output_logistic.csv.

----
## Porbabilistic Generative Model

    In this section we will discuss a generative approach to binary classification.

1. Preparing Data

        訓練集與測試集的處理方法跟 logistic regression 一模一樣，然而因為 generative model 有可解析的最佳解，因此不必使用到 development set.
2. Mean and Covariance

        In generative model, in-class (類別內的) mean (資料平均值) and covariance (共變異數) are needed.
3. Computing weights and bias
    
        權重矩陣與偏差向量可以直接被計算出來
4. Predicting Testing Labels

        Predictions are saved to output_generative.csv.