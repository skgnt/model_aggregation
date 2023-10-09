# これはpytorchのpretrained modelをファインチューニングして、画像分類を行うプログラムです。
# このプログラムは、学習済みモデルを読み込み、最終層を変更して、新しいデータセットに対して学習を行います。
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import parameter_record as pr
import os
from untils import train_model,test_model
from available_model import *

def traing_sequence(yaml_path=None):
    args = pr.ParameterWR()
    if yaml_path == None:
        args.run_name = "kouzou"
        args.model_name = "vit_h_14"
        args.num_classes = 2
        args.batch_size = 32
        args.pretrained = False
        args.epoch = 100
        args.lr = 10e-5
        args.loss_num = 0
        args.data_dir = ""
        args.weight_freeze = (True, None)  # (モデルの最終層を除く重みを固定するか,途中のepochで固定を解除するか)
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        args.validation = True
        args.write_yaml()
    else:
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
    transform = {"train": torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ]), "val": torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])}
    

    # モデル名を全て小文字に変換
    model_name = args.model_name.lower()
    print("you select model is", model_name)
    os.makedirs(f"log/{args.run_name}",exist_ok=True)
    with open(f"log/{args.run_name}/{args.run_name}_status.txt","a") as f:
        f.write(f"you select model is {model_name}\n")

    # #全てのモデルの最終層を表示する
    # for model_kind in model_kind_list:

    #     for model_s in model_kind:
    #         model=torch.hub.load('pytorch/vision', model_s, pretrained=args.pretrained)
    #         os.makedirs(f"log/{args.run_name}",exist_ok=True)
    #         print(f"---{model_s}---")
    #         with open(f"log/{args.run_name}/{model_s}_architecture.txt","w") as f:
    #             f.write(str(model))

    # return 0


    flg = False
    # モデルの種類が存在するか確認
    for model_kind in model_kind_list:
        if model_name in model_kind:
            flg = True
    if flg == False:
        raise Exception("Cannot find model name. Please check the model name.")

    # 1. モデルの読み込み
    model = torch.hub.load('pytorch/vision', model_name,
                        pretrained=args.pretrained)
    model.device = args.device

    if model_name in efficientnet:
        in_features=list(model.children())[-1][1].in_features
    elif model_name in alexnet:
        in_features=9216
    else:
        try:
            in_features = list(model.children())[-1][0].in_features
        except:
            in_features = list(model.children())[-1].in_features

    if in_features < 1024:
        if in_features < 512:
            one_liner_out = in_features
        else:
            one_liner_out = 512
    else:
        one_liner_out = 1024

    #層追加用変数
    #全結合層の追加
    #ReLU層の追加
    #Dropout層の追加
    #全結合層の追加
    torch.manual_seed(0)

    add_layer = nn.Sequential(
        nn.Linear(in_features=in_features, out_features=one_liner_out),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=one_liner_out, out_features=args.num_classes),
    )

    


    # 3. 最終層の変更(指定されたモデルによって最終層の形式が異なるため)
    if model_name in vgg:
        model.classifier = add_layer
    elif model_name in resnet:
        model.fc = add_layer

    elif model_name in wide_resnet:
        model.fc = add_layer

    elif model_name in efficientnet:
        model.classifier = add_layer

    elif model_name in alexnet:
        model.classifier = add_layer
    elif model_name in inception:
        model.fc=add_layer
        model.aux_logits = False 
        transform = {"train": torchvision.transforms.Compose([
                              torchvision.transforms.Resize((299, 299)),
                              torchvision.transforms.RandomHorizontalFlip(),
                              torchvision.transforms.RandomVerticalFlip(),
                              torchvision.transforms.ToTensor(),
                              torchvision.transforms.Normalize((0.5,), (0.5,))
                ]), "val":  torchvision.transforms.Compose([
                            torchvision.transforms.Resize((299, 299)),
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize((0.5,), (0.5,))
                ])}
    

    elif model_name in googlenet:
        model.fc = add_layer

    elif model_name in swin:
        model.head = add_layer

    elif model_name in vit:
        model.heads = add_layer


    # if model_name in vgg:
    #     model.classifier[6] = nn.Linear(
    #         in_features=in_features, out_features=args.num_classes)
    # elif model_name in resnet:
    #     model.fc = nn.Linear(in_features=in_features,out_features=args.num_classes)

    # elif model_name in wide_resnet:
    #     model.fc = nn.Linear(in_features=in_features,out_features=args.num_classes)

    # elif model_name in efficientnet:
    #     model.fc = nn.Linear(in_features=in_features,out_features=args.num_classes)

    # elif model_name in alexnet:
    #     model.classifier[6] = nn.Linear(
    #         in_features=in_features, out_features=args.num_classes)
    # elif model_name in inception:
    #     model.fc = nn.Linear(in_features=in_features,out_features=args.num_classes)

    # elif model_name in googlenet:
    #     model.fc = nn.Linear(in_features=in_features,out_features=args.num_classes)

    # elif model_name in swin:
    #     model.head = nn.Linear(in_features=in_features,out_features=args.num_classes)

    # elif model_name in vit:
    #     model.head = nn.Linear(in_features=in_features,out_features=args.num_classes)
    load=torch.load(r"C:\Users\teralab\Documents\model_aggregation\log\byouri_10w-pre\last_model_weight.pth")
    model.load_state_dict(load)
    
        # 2. モデルの最終層以外の重みを固定する
    if args.weight_freeze[0] == True:
        for param in model.parameters():
            param.requires_grad = False
        # 最終層の重みのみを固定解除
        last_layer = list(model.children())[-1]
        for param in last_layer.parameters():
            param.requires_grad = True

    #モデルのアーキテクチャをlogに記録
    with open(f"log/{args.run_name}/{args.model_name}_architecture.txt","w") as f:
        f.write(str(model))

    # 重みを表示して重みが固定されているか確認
    # for param in model.parameters():
    #     print(param.requires_grad)

    # 4. データセットの読み込み
    # print({x:os.path.join(args.data_dir, x) for x in learning_model})
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
                   freeze=args.weight_freeze[1])
    print("---end training---")

    # 9. モデルの評価
    if args.test:
        print("---start test---")
        if args.test_best_model:
            model.load_state_dict(torch.load(f"log/{args.run_name}/best_model_weight.pth"))
        else:
            model.load_state_dict(torch.load(f"log/{args.run_name}/last_model_weight.pth"))
        #モデルの評価
        test_model(model=model,data_path=os.path.join(args.data_dir, "test"),transform=transform["val"],device=args.device,run_name=args.run_name)
        print("---end test---")


if __name__ == "__main__":
    traing_sequence()
