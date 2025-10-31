from torch import nn
from dataclasses import dataclass

transforms_in_the_original_paper = [ 
    "Identity", 
    "AutoContrast", 
    "Equalize", 
    "Rotate", 
    "Solarize", 
    "Color", 
    "Posterize", 
    "Contrast", 
    "Brightness", 
    "Sharpness", 
    "ShearX", 
    "ShearY", 
    "TranslateX", 
    "TranslateY"
]  

transforms_in_the_oct_paper = [
    "TranslateX", 
    "TranslateY",
    "horizontal flip (Rotate 180)",
    "Rotate",
    "scale (ShearX, ShearY)",
]

def randaugment(N, M): 
    """Generate a set of distortions.  
    Args: 
        N: Number of augmentation transformations to apply sequentially. 
        M: Magnitude for all the transformations. 
    """  
    sampled_ops = np.random.choice(transforms, N) 
    return [(op, M) for op in sampled_ops]

class TransferLearning:

    @dataclass
    class TransferLearningHead:
        label: str
        input_size: int
        
    BASE_OUTPUT_SINGLE_LAYER: dict[str, TransferLearningHead] = {
        "resnet152": TransferLearningHead("fc", 2048),
        "densenet169": TransferLearningHead("classifier", 1664),
    }

    @staticmethod
    def replace_head(model_name, model, num_classes):
        if model_name in TransferLearning.BASE_OUTPUT_SINGLE_LAYER:
            tlh = TransferLearning.BASE_OUTPUT_SINGLE_LAYER[model_name]
            setattr(model, tlh.label, nn.Linear(tlh.input_size, num_classes))
        else: 
            raise ValueError(f"Invalid model name {model_name}")