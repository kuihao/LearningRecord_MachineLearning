Logistic Regression
===
2020/11/14

---
* **Make a Funcion Set**<br>
令 P_w,b 為 σ(z)
令 z = w * x + b = Sum( w_i * x_i ) + b
> **f_w,b = P_w,b(C1|x)**
* **Goodness of a function**
> Training Data: (x_n, y_n^head)<br>
> 令 y_n^head 為 訓練資料的真值
> 令 Class 1 的 y_n^head = 1 <br>
> 令 Class 2 的 y_n^head = 0 <br>
> 令 C( ) 為 Cross entropy

> Loss(f(x)) = Sum( C(f(x_n), y_n^head) )<br>
> ![Loss function derive](https://github.com/kuihao/Learning-record__Machine-learning/blob/main/Classification/log/LogisticRegresssion.jpg "Loss function derive")<br>
* **Find the best function**
* 對 Loss function 作微分
* Cross entropy 會比 Gradient descent 更容易、快速找到最佳解