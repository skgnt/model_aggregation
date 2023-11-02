import glob
import random
import os
import shutil
import time
import copy


class Cross_Validation:
    def __init__(self,glob_path,val=5,class_name=None,train_validation_ratio="1/5",result_folder="root"):
        """Cross_Validation

        Args:
            glob_path (str): globで使用するパス。これに合致するファイルは全て同じ処理がされます。
                             つまり、二つのクラスで分けたいときはこのクラスを二回つからなければならないです。
            val (int, optional): 分割数です。 Defaults to 5.
            class_name (_type_, optional): 複数のクラスが存在するときに区別できるように設定します。 Defaults to None.
            train_validation_ratio (str, optional): trainとvalidationの割合を設定します。割り算表記の整数で記述して、文字列としてください。 Defaults to "1/5".
        """
        self.val=val
        self.glob_path=glob_path
        self.class_name=f"/{class_name}" if class_name!=None else ""
        self.molecule=int((train_validation_ratio.split("/"))[0])
        self.denominator=int((train_validation_ratio.split("/"))[1])
        self.result_folder=result_folder

    def process(self):
        path_all=glob.glob(self.glob_path)
        path_group=[[]]*(self.val)
        class_num=0
        print(path_all)

        print("処理開始")
        start=time.time()
        while len(path_all)!=0:
            path=path_all.pop(random.randint(0,len(path_all)-1))
            path_group[class_num]=path_group[class_num]+[path]
            class_num+=1
            class_num=class_num%self.val

        for i in range(self.val):
            for k in range(len(path_group[i])):
                make_test_path=f"{self.result_folder}split{i+1}/test{self.class_name}"
                os.makedirs(make_test_path,exist_ok=True)
                make_img_path=f"{make_test_path}/{os.path.basename(path_group[i][k])}"
                shutil.copy(path_group[i][k],make_test_path)

            train_val=[]
            for j in range(self.val):
                if j!=i:
                    train_val=train_val+path_group[j]

            count=0
            make_train_path=f"{self.result_folder}split{i+1}/train{self.class_name}"
            make_val_path=f"{self.result_folder}split{i+1}/val{self.class_name}"
            os.makedirs(make_train_path,exist_ok=True)
            os.makedirs(make_val_path,exist_ok=True)

            rand=random.sample([i for i in range(0,self.denominator)],5)

            spl=[]
            while len(train_val)!=0:
                tv_path=train_val.pop(random.randint(0,len(train_val)-1))
                if len(spl)==0:
                    spl=copy.deepcopy(rand)

                if spl.pop(0)>=self.molecule:
                    make_img_path=f"{make_train_path}"
                else:
                    make_img_path=f"{make_val_path}"
                shutil.copy(tv_path,make_img_path)
                count+=1
                count=count%5
        end=time.time()
        print(f"処理終了\ntime:{round(end-start,2)}秒")




