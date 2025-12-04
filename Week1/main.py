import os
from typing import *
from PIL import Image
import glob
from grid_search import gridsearch


def Dataset(ImageFolder:str = "data/MIT_split/train") -> List[Tuple[Type[Image.Image], int]]:

    """
    Expected Structure:

        ImageFolder/<cls label>/xxx1.png
        ImageFolder/<cls label>/xxx2.png
        ImageFolder/<cls label>/xxx3.png
        ...

    Example:
        ImageFolder/cat/123.png
        ImageFolder/cat/nsdf3.png
        ImageFolder/cat/[...]/asd932_.png
    
    """

    map_classes = {clsi: idx for idx, clsi  in enumerate(os.listdir(ImageFolder))}
    dataset :List[Tuple] = []

    for _, cls_folder in enumerate(os.listdir(ImageFolder)):
        image_path = os.path.join(ImageFolder, cls_folder)
        images: List[str] = glob.glob(image_path+"/*.jpg")
        for img in images:
            img_pil = Image.open(img).convert("RGB")
            dataset.append((img_pil, map_classes[cls_folder]))

    return dataset


if __name__ == "__main__":
    data_train = Dataset(ImageFolder="./data/MIT_split/train")
    data_test = Dataset(ImageFolder="./data/MIT_split/test") 

    best_config = gridsearch(data_train,data_test)
    
    print("BEST CONFIGURATION FOUND ACROSS ALL CLASSIFIERS:")
    print(best_config)



