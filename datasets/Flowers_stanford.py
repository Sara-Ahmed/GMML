import torchvision 
from PIL import Image

import numpy as np

from typing import Any, Tuple

from typing import Optional

from datasets.datasets_utils import getItem

class Flowers(torchvision.datasets.ImageFolder):

    def __init__( self, root, transform=None, target_transform=None, 
                 loader=torchvision.datasets.folder.default_loader, training_mode = 'SSL'
    ) -> None:
        super(Flowers, self).__init__(root, loader=loader,
                                          transform=transform,
                                          target_transform=target_transform)
        
        self.training_mode = training_mode

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        target = self.targets[index]
        with open(self.imgs[index][0], 'rb') as f:
            img = Image.open(f).convert('RGB')
                    
        return getItem(-1, img, target, self.transform, self.training_mode)
