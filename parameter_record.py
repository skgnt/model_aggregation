import os
# yamlがない場合はpip install pyyamlでインストールしてください
try:
    import yaml
except:
    print("Please install pyyaml by pip install pyyaml")
    exit()


class ParameterWR:
    def __init__(self):
        self.params = {}

    def __setattr__(self, name, value):
        if name != "params":
            add_param=value
            if isinstance(value,tuple):
                add_param=list(value)
            self.params[name] = add_param
        super().__setattr__(name, value)

    def load_yaml(self, path):
        with open(path, "r") as yaml_file:
            self.params = yaml.safe_load(yaml_file)
        for key, value in self.params.items():
            self.__setattr__(key, value)

    def write_yaml(self,record="./record"):
        if not hasattr(self, "run_name"):
            raise Exception("Cannot find run_name which is an instance variable.")
        else:
            if not os.path.exists(record):
                os.makedirs(record)
            
            yaml_path = os.path.join(record, f"{self.run_name}.yaml")

            with open(yaml_path, "w") as yaml_file:
                yaml.dump(self.params, yaml_file)
            return yaml_path