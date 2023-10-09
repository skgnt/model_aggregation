import parameter_record as pr
from model_aggregation import traing_sequence
import os
import torch
from untils import data_vtt,csv_analyze
import shutil
from torch.utils.tensorboard import SummaryWriter

def main_sequence(pr_y=None):
    args = pr.ParameterWR()

    if pr_y==None:
        args.run_name = "byouri_10w-pre_plus"
        args.cache=True#キャッシュを残すかどうか
        args.sys_split=False#システムによるtrain,validation,testの自動分割を行うかどうか
        args.save_wip_model=None #モデルを途中で保存するかどうか.保存する場合は保存するepoch数を指定する,保存しない場合はNone
        args.test=True#testを行うかどうか
        args.test_best_model=False#testを行う際に最も良いモデルを使用するかどうか(Falseの場合はトレーニングの最後のモデルを使用する)
        args.validation=False#validationを行うかどうか
        args.data_dir = r"C:\Users\teralab\Documents\byouri_data\byouri_10wari"#データセットのパス
        args.weight_only=True#重みのみを保存するかどうか
        args.weight_freeze = (False, None)  # (モデルの最終層を除く重みを固定するか,途中のepochで固定を解除するか)
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # parameter設定
        args.model_name = "wide_resnet50_2"
        args.num_classes = 2
        args.batch_size = 16
        args.pretrained = True
        args.epoch = 30
        args.lr = 10e-5
        # lossの種類を選択
        # 0:CrossEntropyLoss
        # 1:BCELoss
        # 2:MultiMarginLoss
        # 3:BCEWithLogitsLoss
        # 4:CosineEmbeddingLoss
        # 5:CTCLoss
        args.loss_num = 0

        
    else:
        args.load_yaml(pr_y)
    
    
    
    

    if not args.pretrained:
        args.weight_freeze=(False,None)
    

    #警告を無視する
    if True:
        import warnings
        warnings.filterwarnings("ignore")
    
    #yamlに同様の名前があれば、エラーを送出する。
    if os.path.exists(f"log\{args.run_name}\{args.run_name}.yaml"):
        raise Exception(f"run_name {args.run_name} is already exists.")

    try:
        tt_split=0
        vtr_split=1
        if args.test:
            tt_split=0.2
        if args.validation:
            vtr_split=0.8

        if args.sys_split:
            print("---cache記録中---")
            new_data_dir=rf"cache/{args.run_name}"
            data_vtt(args.data_dir,result_folder=new_data_dir,tt_split=tt_split,vtr_split=vtr_split)
            print("---cache記録完了---")
            args.data_dir=new_data_dir
    except:
        import traceback
        #logフォルダがなければ作成
        os.makedirs(f"log\{args.run_name}",exist_ok=True)
        #エラーをlogに記録
        with open(f"log\{args.run_name}\error.txt","w") as f:
            traceback.print_exc(file=f)

        print("エラーが発生しました。")
        print("キャッシュを削除します。")
        print("詳細はlogフォルダを確認してください。")
        shutil.rmtree(rf"cache/{args.run_name}")
        exit()

    try:
        yaml_path=args.write_yaml(record=f"log\{args.run_name}")
        traing_sequence(yaml_path)
        if args.test:
            print("---auto analyze---")
            csv_analyze(f"log/{args.run_name}/test.csv",run_name=args.run_name)
            print("---auto analyze end---")
    except:
        import traceback
        #logフォルダがなければ作成
        os.makedirs(f"log\{args.run_name}",exist_ok=True)
        #エラーをlogに記録
        with open(f"log\{args.run_name}\error.txt","w") as f:
            traceback.print_exc(file=f)

        print("エラーが発生しました。")
        print("詳細はlogフォルダを確認してください。")
        if args.sys_split:
            if input("キャッシュを削除しますか？(y/n)")=="y" :
                shutil.rmtree(new_data_dir)
        exit()
    if args.sys_split:
        if args.cache==False:
            #キャッシュの中身を削除
            shutil.rmtree(new_data_dir)

if __name__ == "__main__":
    main_sequence()