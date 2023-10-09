import parameter_record as pr
from main_sequence import main_sequence
import pandas as pd
import os
import torch
from available_model import model_kind_list

args = pr.ParameterWR()
run_name = "byouri_5wari"
args.cache=True#キャッシュを残すかどうか
args.sys_split=False#システムによるtrain,validation,testの自動分割を行うかどうか
args.save_wip_model=None #モデルを途中で保存するかどうか.保存する場合は保存するepoch数を指定する,保存しない場合はNone
args.test=True#testを行うかどうか
args.test_best_model=False#testを行う際に最も良いモデルを使用するかどうか(Falseの場合はトレーニングの最後のモデルを使用する)
args.validation=False#validationを行うかどうか
args.data_dir = r"C:\Users\teralab\Documents\byouri_5wari"#データセットのパス
args.weight_only=True#重みのみを保存するかどうか
args.weight_freeze = (True, 5)  # (モデルの最終層を除く重みを固定するか,途中のepochで固定を解除するか)
args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

# parameter設定
args.num_classes = 2
args.batch_size = 16
args.pretrained = True
args.epoch = 60
args.lr = 10e-5
# lossの種類を選択
# 0:CrossEntropyLoss
# 1:BCELoss
# 2:MultiMarginLoss
# 3:BCEWithLogitsLoss
# 4:CosineEmbeddingLoss
# 5:CTCLoss
args.loss_num = 0

flg=True
for model_kind in model_kind_list:
    for model in model_kind:
        args.run_name = f"{run_name}_{model}"
        args.model_name = model
        args.write_yaml(record="./record")
        main_sequence(pr_y=f"./record/{args.run_name}.yaml")

        if flg:
            if args.sys_split:
                args.sys_split=False
                args.data_dir = rf"cache/{args.run_name}"
            flg=False

        if args.num_classes==2:
            #log\{args.run_name}\test.csvからBest Accを取得する
            df_roc = pd.read_csv(rf"log\{args.run_name}\test_roc.csv")
            #df_rocを通常のリストに変換
            df_roc=df_roc.values.tolist()
            cutoff_ll=0.4
            cutoff_ul=0.6
            max_acc=0
            best_cutoff=0
            for i in df_roc:
                
                #cutOffがcutoff_ll以上,cutoff_ul以下の時のみ
                if cutoff_ll<=i[0]<=cutoff_ul:
                    if max_acc<i[3]:
                        max_acc=i[3]
                        best_cutoff=i[0]
            os.makedirs(f"record", exist_ok=True)
            with open(f"record/model_acc.txt", "a") as f:
                
                f.write(f"{model}_cutoff,{max_acc},cutoff,{best_cutoff}\n")

        sp=""
        #log\{args.run_name}\{args.run_name}_status.txtを一行ずつ読み込む
        with open(f"log\{args.run_name}\{args.run_name}_status.txt", "r") as f:
            status=f.readlines()
            #:の前後で分割し、リストに格納する
            for i in status:
                i=i.split(":")
                if i[0]=="Test Accuracy":
                    sp=((i[1].split("("))[1].split("%"))[0]
                    
        with open(f"record/model_acc.txt", "a") as f:
            f.write(f"{model},{sp}\n")

        print("----------")
        
        

