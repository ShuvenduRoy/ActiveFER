import numpy as np
from PIL import Image
import os
import os.path
from typing import Any, Callable, Optional, Tuple

from torchvision.datasets.vision import VisionDataset


class AffectDataset(VisionDataset):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:

        super(AffectDataset, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set
        split = 'train' if self.train else 'val'

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays

        self.data = np.load(os.path.join(self.root, '{}_data.npy'.format(split)))
        # self.data = self.data.transpose((0, 3, 1, 2))  # convert to HWC
        # print(self.data.shape)
        self.targets = np.load(os.path.join(self.root, '{}_label.npy'.format(split)))
        # print(self.targets.shape)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)
