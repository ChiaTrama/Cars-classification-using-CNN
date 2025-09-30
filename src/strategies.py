from src.model import SimpleResNet, get_resnet18, SimpleResNetLarge, get_inception
from src.model import SimpleSiamese, SiameseResNet18 , SiameseEfficientNetB0, SiameseResNet50_30M, SimpleSiameseResNet
from src.dataset import get_albu_transform

def get_strategies(NUM_CLASSES):
    return {
        "SimpleResNetLarge_224x224": {
            "name": "SimpleResNetLarge 224x224",
            "target_size": (224, 224),
            "use_bbox": True,
            "augmentations": get_albu_transform(target_size=(224, 224), train=True),
            "val_augmentations": get_albu_transform(target_size=(224, 224), train=False),
            "model": lambda embedding_dim=None: SimpleResNetLarge(num_classes=NUM_CLASSES),
            "use_dataset_mean": True
        },
        "SimpleResNet_224x224": {
            "name": "SimpleResNet 224x224",
            "target_size": (224, 224),
            "use_bbox": True,
            "augmentations": get_albu_transform(target_size=(224, 224), train=True),
            "val_augmentations": get_albu_transform(target_size=(224, 224), train=False),
            "model": lambda embedding_dim=None: SimpleResNet(num_classes=NUM_CLASSES),
            "use_dataset_mean": False
        },
        "ResNet18_FineTuning_224x224": {
            "name": "ResNet18 Fine-tuning",
            "target_size": (224, 224),
            "use_bbox": True,
            "augmentations": get_albu_transform(target_size=(224, 224), train=True),
            "val_augmentations": get_albu_transform(target_size=(224, 224), train=False),
            "model": lambda embedding_dim=None: get_resnet18(num_classes=NUM_CLASSES),
            "use_dataset_mean": False
        },
        "InceptionV3_299x299": {
            "name": "InceptionV3 299x299",
            "target_size": (299, 299),
            "use_bbox": True,
            "augmentations": get_albu_transform(target_size=(299, 299), train=True),
            "val_augmentations": get_albu_transform(target_size=(299, 299), train=False),
            "model": lambda embedding_dim=None: get_inception(num_classes=NUM_CLASSES, pretrained=True),
            "use_dataset_mean": True
        },
        "SiameseResNet18_192x192": {
            "name": "Siamese ResNet18 192x192",
            "embedding_dim": 512,
            "model": lambda embedding_dim=None: SiameseResNet18(embedding_dim=embedding_dim if embedding_dim is not None else strat["embedding_dim"], pretrained=True),
            "target_size": (192, 192),
            "augmentations": get_albu_transform(target_size=(192, 192), train=True),
            "val_augmentations": get_albu_transform(target_size=(192, 192), train=False),
            "use_bbox": False,
        },
        "SiameseEfficientNet_B0_192x192": {
            "name": "SiameseEfficientNet_B0 192x192",
            "target_size": (192, 192),
            "embedding_dim": 512,
            "use_bbox": False,
            "augmentations": get_albu_transform(target_size=(192, 192), train=True),
            "val_augmentations": get_albu_transform(target_size=(192, 192), train=False),
            "model": lambda embedding_dim=None: SiameseEfficientNetB0(embedding_dim=embedding_dim if embedding_dim is not None else strat["embedding_dim"], pretrained=True),
            "use_dataset_mean": True
        },
        "SiameseResNet50_30M_192x192_pretrained": {
            "name": "Siamese ResNet50 30M",
            "embedding_dim": 512,
            "model": lambda embedding_dim=None: SiameseResNet50_30M(embedding_dim=embedding_dim if embedding_dim is not None else strat["embedding_dim"], pretrained=True),
            "target_size": (192, 192),
            "augmentations": get_albu_transform(target_size=(192, 192), train=True),
            "val_augmentations": get_albu_transform(target_size=(192, 192), train=False),
            "use_bbox": False,
        },
        "SiameseResNet50_30M_192x192": {
            "name": "Siamese ResNet50 30M",
            "embedding_dim": 512,
            "model": lambda embedding_dim=None: SiameseResNet50_30M(embedding_dim=embedding_dim if embedding_dim is not None else strat["embedding_dim"], pretrained=False),
            "target_size": (192, 192),
            "augmentations": get_albu_transform(target_size=(192, 192), train=True),
            "val_augmentations": get_albu_transform(target_size=(192, 192), train=False),
            "use_bbox": False,
        },
        "SiameseResNet18_192x192_scratch": {
            "name": "Siamese ResNet18 192x192 Scratch",
            "embedding_dim": 512,
            "model": lambda embedding_dim=None: SiameseResNet18(embedding_dim=embedding_dim if embedding_dim is not None else strat["embedding_dim"], pretrained=False),
            "target_size": (192, 192),
            "augmentations": get_albu_transform(target_size=(192, 192), train=True),
            "val_augmentations": get_albu_transform(target_size=(192, 192), train=False),
            "use_bbox": False,
            "use_dataset_mean": True
        }
    }

# Serialize a strategy for saving in results
def serialize_strategy(strat):
    return {
        "name": strat.get("name"),
        "target_size": strat.get("target_size"),
        "use_bbox": strat.get("use_bbox"),
        "use_dataset_mean": strat.get("use_dataset_mean"),
        "augmentations": str(strat.get("augmentations")),
        "val_augmentations": str(strat.get("val_augmentations")),
        "embedding_dim": strat.get("embedding_dim") if "embedding_dim" in strat else None,
        # model is not included because it's a lambda/function and not serializable
    }