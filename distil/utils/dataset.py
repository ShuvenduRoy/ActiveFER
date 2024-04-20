import os
from PIL import Image
from torchvision.datasets.vision import VisionDataset

class ImageDataset(VisionDataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
    ) -> None:
        super(ImageDataset, self).__init__(root, transform=transform)

        self.train = train  # training set or test set
        split = 'train' if self.train else 'val'

        self.data: Any = []
        self.targets = []

        # Load images from folder
        folders = os.listdir(os.path.join(self.root, split))
        for idx, folder in enumerate(folders):
            files = os.listdir(os.path.join(self.root, split, folder))
            for file in files:
                self.data.append(os.path.join(self.root, split, folder, file))
                self.targets.append(idx)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = Image.open(self.data[index])

        if self.transform is not None:
            img = self.transform(img)

        return img, self.targets[index]

    def __len__(self) -> int:
        return len(self.data)