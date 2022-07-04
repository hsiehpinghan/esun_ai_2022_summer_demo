# README
## 程式架構
    .
    ├── README.md
    ├── data_preprocess
    │ └── data_preprocess.py
    ├── inference
    │ └── inference.py
    ├── model
    │ ├── data.py
    │ ├── model.py
    │ └── training.py
    ├── api.py
    ├── app.py
    └── requirements.txt

## 程式環境
* 系統平台：  
  模型開發平台：Colab Pro  
  推論服務平台：GCP n1-standard-4 * 1 + T4 GPU * 1  
* 程式語言：  
  Python  
* 函式庫：  
  yacs==0.1.8  
  jiwer==2.3.0  
  torch==1.11.0  
  torchmetrics==0.9.2  
  pytorch-lightning==1.6.4  
  datasets==2.2.1  
  transformers==4.19.2  

## 執行範例 <a href="https://colab.research.google.com/github/hsiehpinghan/esun_ai_2022_summer_demo/blob/master/model-spec/demo.ipynb"><img data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" src="https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667"></a>