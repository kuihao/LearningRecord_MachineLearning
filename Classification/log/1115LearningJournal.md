ML Homework: Classification (客戶年收入預測)
===
* 2020/11/15<br>
* 程式碼參考來源：https://colab.research.google.com/drive/1JaMKJU7hvnDoUfZjvUKzm9u-JLeX6B2C <br>
* 課程目的：二元分類是機器學習中最基礎的問題之一，在這份教學中，你將學會如何實作一個線性二元分類器，來根據人們的個人資料，判斷其年收入是否高於 50,000 美元。我們將以兩種方法: logistic regression 與 generative model，來達成以上目的，你可以嘗試了解、分析兩者的設計理念及差別。

---
0. Colab 檔案讀取
    * 檔案共用連結
        > !gdown --id '**file_id**' --output Example.jpg
    
        * 從 Google Drive 下載特定 file_id 的檔案，並將它命名為 Example.jpg
        * Google Drive 雲端網址 'open?id=' 後面的亂碼即為 **file_id**

        > !ls
        * 列出目前目錄下所有的檔案

    * 掛載自己的 Google Drive<br> 

        可以讓 Colab 上的程式直接讀取自己的雲端硬碟。這個方法的好處是只要檔案存在於自己的雲端硬碟，就隨時都可以直接存取；相對地，缺點就是使用者得手動將檔案加入，並且在程式運行時要輸入連結 Google Drive 所需要的授權碼。
        
        Step 1. 導入 Google 套件
        > from google.colab import drive<br>
        > import os<br>
        > drive.mount('/content/gdrive')
        
        Step 2. 點擊 URL 進行授權，並複製授權碼至 colab 的輸入框

        Step 3. 檢視掛載成功
        > 從側邊File已經可以看到 **/content/gdrive/My Drive** 已經被載入

        * G-Suite 也可以透過以下路徑到達小組資料夾
            > **/content/gdrive/Team Drives/{小組名稱}**
        
        * 更改路徑
            > import os<br>
            > os.chdir("/content/gdrive/My Drive") #填入欲更改路徑<br>
            > os.getcwd() #查看當前路徑

1. Colab 解壓縮指令
> !tar -zxvf 檔案.tar.gz

2. 課程提供的資料已初步洗清
    * 移除部分與預測目的顯然無關的項目資料
    * 對離散型資料進行 One-hot encoding
    * 稍微調整使資料的正負值比例相近

3. 檢視資料集
    * 本作業只需用到 X_train、Y_train 及 X_test 
    * 另外的檔案 (sample_submission.csv, test_no_label.csv, train.csv) 則可以提供一些額外的資訊
    * X_train
        > 第一行為表頭 (id, age, 國籍...etc)<br>
        > 共有 54,256 筆資料(列)
    * Y_train
        > 第一行為表頭 (id, label) <br>
        > 共有 54,256 筆資料(列)<br>
        > label >= 1 表示該用戶收入大於 50,000<br>
        > 否則表示該用戶收入小於 50,000
    * X_test
        > 第一行表頭同 X_train.csv
        > 共有 27,622 筆資料(列)<br>

4. Machine Learning 架構
    * Input: 金融用戶的個資
    * Output: 預測該用戶的年收入 (大於 50,000 則顯示 1，否則顯示 0)
    * Model:  Logistic Regression 及 Generative Model
    
 

