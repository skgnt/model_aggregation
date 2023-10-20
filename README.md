# model_aggregation
分類モデルのトレーニングからテスト、混同行列、ROC(2分類のみ)の作成などの一連のプロセスを自動化する。


## 共通事項
* 提供機能(それぞれについてon/offが可能。ただし、一部on/offがまとまっている場合がある)  
✅test、val、train機能  
✅自動解析機能  
✅データ自動分割機能

* 使用可能モデルについて  
  available_model.pyをご覧ください。  

解析内容(2分類)
* Sensitivity  
* Specificity  
* Accuracy  
* Precision  
* F1-score  
* Balanced accuracy
* auc
* cutoff(制限値内)

解析内容(3分類以上)
* Accuracy
<br>
※解析内容はlog直下のdbファイルに書き込まれるため、sqliteのviewrのアプリまたはvscode拡張機能を使用することを推奨する。

## model_aggreation.py
単一のモデルについて評価を行います。  
<br>

<br>


## various_model.py
available_model.pyに存在する複数のモデルについて学習を行います。　　
