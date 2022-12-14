import yaml
from .repr_learning import RepresentationLearning
from .class_learning import ClassificationLearning


def from_yaml(yaml_path):
    """
    see config.yaml
    """
    with open(yaml_path) as f:
        config = dict(yaml.load(f, Loader=yaml.FullLoader))
    learning_method = config["learning"] 
    return from_config(learning_method)
    
def from_config(config:dict):
    """
    see config.yaml
    """
    config = dict(config)
    method = config.pop("name")
    if method == 'representation':
        train_module = RepresentationLearning
    elif method == 'classification':
        train_module = ClassificationLearning
    else:
        raise ValueError(f"Invalid learning method '{method}'")

    return train_module(**config)
