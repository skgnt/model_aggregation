# model_aggregation
分類モデルのトレーニングからテスト、混同行列、ROC(2分類のみ)の作成などの一連のプロセスを自動化する。



## 共通事項
* 必要ライブラリ
```
pyyaml
tensorboard
pandas
matplotlib
torch
torchvision
timm
```

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
※解析内容はlog直下のdbファイルに書き込まれるため、sqliteのviewrのアプリまたはvscode拡張機能を使用することを推奨します。

## model_aggreation.py
単一のモデルについて評価を行います。  



## various_model.py
available_model.pyに存在する複数のモデルについて学習を行います。　　


# use_transform.py
modelのトレーニングに使うtransformの設定が行なえます。

# available_model_pytorch.py
pytorchのプレトレーニングモデルを確認でき、available_model_pytorchに含まれるモデルがvarious_model.pyを実行したときに使用されるモデルで、含まれていないモデルは,model_aggreation.pyで実行することができません。ただし、available_model_pytorchにすべてのmodelが含まれていない可能性があり、任意で追加することが可能です。
また、出力層の設定も行うことができます。

# available_model_timm.py
timmのプレトレーニングモデルを一部を確認でき、available_model_timmに含まれるモデルがvarious_model.pyを実行したときに使用されるモデルで、含まれていないモデルは,model_aggreation.pyで実行することができません。ただし、available_model_timmにすべてのモデルは含まれていません。
全てのモデルを確認するためには以下のプログラムを実行します。
```
import timm
model_list=timm.list_models(pretrained=True)
print("\n".join(model_list))
```

