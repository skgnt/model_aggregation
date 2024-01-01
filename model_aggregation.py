# これはpytorchのpretrained modelをファインチューニングして、画像分類を行うプログラムです。
# このプログラムは、学習済みモデルを読み込み、最終層を変更して、新しいデータセットに対して学習を行います。
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import parameter_record as pr
import os
from untils import train_model,test_model
from available_model_pytorch import model_output_plus,available_model_pytorch
from available_model_timm import FineTuning,available_model_timm,change_transform
import timm
from use_transform import use_transform


def traing_sequence(yaml_path=None):
    args = pr.ParameterWR()
    args.load_yaml(yaml_path)

    # userが選択できるようにlossの配列を作成
    loss_func_choices = [nn.CrossEntropyLoss(),
                        nn.BCELoss(), 
                        nn.MultiMarginLoss(), 
                        nn.BCEWithLogitsLoss(),
                        nn.CosineEmbeddingLoss(),
                        nn.CTCLoss()]
    loss_func = loss_func_choices[args.loss_num]
    learning_model=['train', 'val'] if args.validation else ['train']
    transform = use_transform
    

    # モデル名を全て小文字に変換
    model_name = args.model_name.lower()
    print("you select model is", model_name)
    os.makedirs(f"{args.log_folder}/{args.run_name}",exist_ok=True)
    with open(f"{args.log_folder}/{args.run_name}/{args.run_name}_status.txt","a") as f:
        f.write(f"you select model is {model_name}\n")

    if args.source == "timm":
        flg = False
        for model_kind in available_model_timm:
            if model_name in model_kind:
                flg = True
        if flg == False:
            raise Exception("Cannot find model name. Please check the model name.")

        model = timm.create_model(model_name, pretrained=args.pretrained, num_classes=0,in_chans=args.channel)
        model = FineTuning(model=model, num_features=model.num_features, num_classes=args.num_classes)
        transform = change_transform(transform=transform, model_name=model_name)
        model.device = args.device

    elif args.source == "pytorch":
        flg = False
        for model_kind in available_model_pytorch:
            if model_name in model_kind:
                flg = True
        if flg == False:
            raise Exception("Cannot find model name. Please check the model name.")

        model = torch.hub.load('pytorch/vision', model_name, pretrained=args.pretrained)
        model,transform = model_output_plus(model=model, model_name=model_name, num_classes=args.num_classes, transform=transform)
        model.device = args.device

    # 2. モデルの最終層以外の重みを固定する
    if args.weight_freeze[0] == True:
        for param in model.parameters():
            param.requires_grad = False
        # 最終層の重みのみを固定解除
        last_layer = list(model.children())[-1]
        for param in last_layer.parameters():
            param.requires_grad = True

    #モデルのアーキテクチャをlogに記録
    with open(f"{args.log_folder}/{args.run_name}/{args.model_name}_architecture.txt","w") as f:
        f.write(str(model))

    # 4. データセットの読み込み
    image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(args.data_dir, x), transform[x])
                    for x in learning_model}
    datasize = {x: len(image_datasets[x]) for x in learning_model}

    # 5. DataLoaderの作成
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True,drop_last=True)
                for x in learning_model}

    # 6. optimizerの設定
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    

    # 7. 学習
    print("---start training---")
    train_model(model=model, dataloaders=dataloaders, loss_func=loss_func, optimizer=optimizer,run_name=args.run_name,vt=learning_model,
                   dataset_sizes=datasize, device=args.device, num_epochs=args.epoch, scheduler=None, save_wip_model=args.save_wip_model,
                   freeze=args.weight_freeze[1],weight_only=args.weight_only,log_folder=args.log_folder)
    print("---end training---")

    # 9. モデルの評価
    if args.test:
        print("---start test---")
        if args.test_best_model:
            model.load_state_dict(torch.load(f"{args.log_folder}/{args.run_name}/best_model_weight.pth"))
        else:
            model.load_state_dict(torch.load(f"{args.log_folder}/{args.run_name}/last_model_weight.pth"))
        #モデルの評価
        test_model(model=model,data_path=os.path.join(args.data_dir, "test"),transform=transform["val"],device=args.device,run_name=args.run_name,log_folder=args.log_folder)
        print("---end test---")


if __name__ == "__main__":
    traing_sequence()
