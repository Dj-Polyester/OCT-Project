# %%
from copy import deepcopy
from pathlib import Path

from sklearn.model_selection import train_test_split

from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
    resnet34,
    ResNet34_Weights,
    resnet50,
    ResNet50_Weights,
    resnet101,
    ResNet101_Weights,
    resnet152,
    ResNet152_Weights,
    densenet121,
    DenseNet121_Weights,
    densenet161,
    DenseNet161_Weights,
    densenet169,
    DenseNet169_Weights,
    densenet201,
    DenseNet201_Weights,
    mobilenet_v2,
    MobileNet_V2_Weights,
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
    mobilenet_v3_large,
    MobileNet_V3_Large_Weights,
)
from torchvision.transforms import v2
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, Timer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from torchinfo import summary
import segmentation_models_pytorch as smp
from utils import (
    OCTDataset, 
    IterableOCTDataset, 
    plotimg, 
    LitSupervised, 
    OCTRandAugment, 
    TransferLearningModel,
    TransferLearning,
    GLOBAL_CONFIG,
    product_dict, 
    LOGDIR, 
    OCTCollate,
)
from core_utils import stringify_map, get_normal_statistics

# %%
dataset = OCTDataset.from_path("OCTDataMendeley_resized")

# %%
train_paths, test_paths, train_labels, test_labels = train_test_split(
    dataset.paths, 
    dataset.targets,
    stratify=dataset.targets, 
    test_size=1000, 
    random_state=42
)
train_paths, val_paths = train_test_split(
    train_paths, 
    stratify=train_labels, 
    test_size=0.2, 
    random_state=42
)

# %%
train_dataset = OCTDataset(train_paths)
val_dataset = OCTDataset(val_paths)
test_dataset = OCTDataset(test_paths)
del dataset
len(train_dataset), len(val_dataset), len(test_dataset)

# %%
#get_normal_statistics(IterableOCTDataset(train_dataset))
#(tensor([48.6167, 48.6167, 48.6167]), tensor([56.4110, 56.4110, 56.4110]))

# %%
OCTDataset.classes()

# %%
train_dataset.class_dist(), val_dataset.class_dist() 

# %%
train_dataset.paths[400], train_dataset.targets[400]

# %%
from collections import Counter
from pprint import pprint

counts = dict(Counter(train_dataset.paths))
with open("../test/counts.txt", "w") as f:
    pprint(counts, stream=f)

# %%
40736/10184, 29689/7423, 9194/2298, 7028/1757 

# %%
train_dataset.oversample()

# %%
train_dataset.class_dist(), val_dataset.class_dist() 

# %%
_index = 6
path, instance, label = test_dataset[_index]

# %%
plotimg(instance, f"{OCTDataset.classes()[label]} {path}")

# %%
train_loader = DataLoader(
    train_dataset, 
    batch_size=GLOBAL_CONFIG["batch_size"], 
    shuffle=True,
    num_workers=7,
    collate_fn=OCTCollate()
)
validation_loader = DataLoader(
    val_dataset, 
    batch_size=GLOBAL_CONFIG["batch_size"], 
    num_workers=7,
    collate_fn=OCTCollate()
)

# %%
# for batch in validation_loader:
#     path, inp, target = batch
#     print(path, inp.shape, target.shape)

# %%
TransferLearning.models = {
    "mobilenet_v2": TransferLearningModel(mobilenet_v2, MobileNet_V2_Weights.IMAGENET1K_V2),
    "mobilenet_v3_small": TransferLearningModel(mobilenet_v3_small, MobileNet_V3_Small_Weights.IMAGENET1K_V1),
    "mobilenet_v3_large": TransferLearningModel(mobilenet_v3_large, MobileNet_V3_Large_Weights.IMAGENET1K_V2),
    "resnet18": TransferLearningModel(resnet18, ResNet18_Weights.IMAGENET1K_V1),
    "resnet34": TransferLearningModel(resnet34, ResNet34_Weights.IMAGENET1K_V1),
    "resnet50": TransferLearningModel(resnet50, ResNet50_Weights.IMAGENET1K_V1),
    "resnet101": TransferLearningModel(resnet101, ResNet101_Weights.IMAGENET1K_V2),
    #"resnet152": TransferLearningModel(resnet152, ResNet152_Weights.IMAGENET1K_V1),
    #"densenet121": TransferLearningModel(densenet121, DenseNet121_Weights.IMAGENET1K_V1),
    #"densenet161": TransferLearningModel(densenet161, DenseNet161_Weights.IMAGENET1K_V1),
    #"densenet169": TransferLearningModel(densenet169, DenseNet169_Weights.IMAGENET1K_V1),
    #"densenet201": TransferLearningModel(densenet201, DenseNet201_Weights.IMAGENET1K_V1),
}

# %%
def train_model(config, ckpt_config = None):
   current_name = stringify_map(config, ", ")
   version_name = stringify_map(config)
   logdir = config["logdir"]
   print(current_name)
   logger = TensorBoardLogger(
      save_dir=".",
      name=logdir,
      version=version_name,
   )
   litmodel = LitSupervised(config, OCTDataset.classes())
   metric_names = litmodel.metric_names()
   timer = Timer()
   trainer = L.Trainer(
      logger=logger,
      accelerator="gpu",
      callbacks=[
         timer,
         EarlyStopping(
            monitor=f"{LitSupervised.SET_NAME_VAL}_loss", 
            mode="min",
            min_delta=1e-3, 
            patience=8,
         ),
      ] + [
         ModelCheckpoint(
            dirpath=None,  # Uses logger's default directory
            monitor=metric_name,  # Metric to monitor
            filename="top-"+metric_name+"-epoch:{epoch:d}-step:{step:d}-{"+metric_name+"}",
            save_top_k=1,  # Save top 1 checkpoints
            mode="max",  # Save highest
            save_last=i==0,  # Also save the last checkpoint
         ) for i, metric_name in enumerate(metric_names)
      ],
   )
   ckpt_path = None
   if isinstance(ckpt_config, dict):
      ckpt_version_name = stringify_map(ckpt_config)
      ckpts_path = Path(logdir, ckpt_version_name, "checkpoints")
      ckpt_path = sorted(list(ckpts_path.iterdir()))[-1]
   trainer.fit(
      model=litmodel, 
      train_dataloaders=train_loader, val_dataloaders=validation_loader,
      ckpt_path=ckpt_path,
   )
   train_time = timer.time_elapsed("train")
   val_time = timer.time_elapsed("validate")
   total_fit_time = train_time + val_time
   print(f"Training completed in {train_time} secs")
   print(f"Validation completed in {train_time} secs")
   print(f"Total time is {total_fit_time} secs")

# %%
def predict_samplewise(config, dataloader: DataLoader):
    current_name = stringify_map(config, ", ")
    version_name = stringify_map(config)
    logdir = config["logdir"]
    print(current_name)
    logger = TensorBoardLogger(
       save_dir=".",
       name=logdir,
       version=version_name,
    )
    ckpt_path = Path(logdir, version_name, "checkpoints", "last.ckpt")
    litmodel = LitSupervised.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        config=config,
        classes=OCTDataset.classes(),
    )
    trainer = L.Trainer(
      logger=logger,
      accelerator="gpu"
    )
    predictions = trainer.predict(
        model=litmodel, 
        dataloaders=dataloader,
    )
    return predictions

# %%
predict_samplewise(
    {
           "run": "step2", 
           "logdir": "logs_oversampled",
           "model": "resnet34", 
           "magnitude": 0, 
           "lr": 1e-4
    },
    validation_loader,
)

# %%
for model_name in TransferLearning.models:
    print(f"Running {model_name}:")
    train_model(
        {
            "run": "step1", 
            "logdir": "logs_oversampled",
            "model": model_name, 
            "magnitude": 0, 
            "lr": 1e-3
        },
    )
    train_model(
        {
            "run": "step2", 
            "logdir": "logs_oversampled",
            "model": model_name, 
            "magnitude": 0, 
            "lr": 1e-4
        },
        {
            "run": "step1", 
            "logdir": "logs_oversampled",
            "model": model_name, 
            "magnitude": 0, 
            "lr": 1e-3
        }
    )


