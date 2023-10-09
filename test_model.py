import torch
from parameter_record import ParameterWR as pr
from main_sequence import main_sequence

def test():
    num=[6,8]
    kaisuu=6

    args = pr()
    args.cache=True#キャッシュを残すかどうか
    args.sys_split=False#システムによるtrain,validation,testの自動分割を行うかどうか
    args.save_wip_model=None #モデルを途中で保存するかどうか.保存する場合は保存するepoch数を指定する,保存しない場合はNone
    args.test=True#testを行うかどうか
    args.test_best_model=False#testを行う際に最も良いモデルを使用するかどうか(Falseの場合はトレーニングの最後のモデルを使用する)
    args.validation=False#validationを行うかどうか
    args.weight_only=True#重みのみを保存するかどうか
    args.weight_freeze = (True, 5)  # (モデルの最終層を除く重みを固定するか,途中のepochで固定を解除するか)
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # parameter設定
    args.model_name = "wide_resnet50_2"
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
    for i in num:
        for j in range(kaisuu):
            args.run_name = f"auto-byouri_{i}w_wrn50_{j+1}"
            args.data_dir = rf"C:\Users\teralab\Documents\byouri_data\byouri_{i}wari"
            args.write_yaml(record="./record")
            main_sequence(pr_y=f"./record/{args.run_name}.yaml")

            kiroku=""
            with open(f"log\{args.run_name}\{args.run_name}_status.txt", "r") as f:
                status=f.readlines()
                #:の前後で分割し、リストに格納する
                for k in status:
                    sp=k.split(":")
                    if sp[0]=="Test Accuracy":
                        kiroku=k
                        
            with open(f"record/test_model_kihon.csv", "a") as f:
                f.write(f"{args.run_name},{kiroku}")          
    



if __name__ == "__main__":
    test()