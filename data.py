import os

import torch
from pycocotools.coco import COCO
from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.transforms.v2 import functional as F


class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, idx):
        coco = self.coco
        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info["file_name"])
        img = read_image(img_path)

        num_objs = len(coco_annotation)
        boxes = torch.zeros((num_objs, 4), dtype=torch.float32)
        masks = torch.zeros((num_objs, img.shape[-2], img.shape[-1]), dtype=torch.uint8)
        masks.to_sparse_coo()
        #TODO: polygon 转sparse优化

        for i, ann in enumerate(coco_annotation):
            # COCO的边界框格式为[x_min, y_min, width, height]
            # 转换为[x_min, y_min, x_max, y_max]
            x_min, y_min, width, height = ann["bbox"]
            boxes[i] = torch.tensor([x_min, y_min, x_min + width, y_min + height])
            masks[i] = torch.tensor(coco.annToMask(ann))
        labels = torch.tensor(
            [ann["category_id"] for ann in coco_annotation], dtype=torch.int64
        )
        image_id = torch.tensor([img_id])
        area = torch.tensor(
            [ann["area"] for ann in coco_annotation], dtype=torch.float32
        )
        iscrowd = torch.tensor(
            [ann["iscrowd"] for ann in coco_annotation], dtype=torch.int64
        )

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(
            boxes, format="XYXY", canvas_size=F.get_size(img)
        )
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)


if __name__ == "__main__":
    data = CocoDataset("./data/img", "./data/coco/train.json")
