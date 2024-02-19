import torch
from torchvision.transforms import v2 as T

import utils
from data import CocoDataset
from engine import evaluate, train_one_epoch
from model import get_model_instance_segmentation


def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
num_classes = 2
dataset = CocoDataset(
    "./data/img/", "./data/coco/train.json", get_transform(train=True)
)
# dataset_test = CocoDataset(
#     "./data/img/", "./data/coco/train.json", get_transform(train=False)
# )
# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=4,
    collate_fn=utils.collate_fn,
)

# data_loader_test = torch.utils.data.DataLoader(
#     dataset_test,
#     batch_size=1,
#     shuffle=False,
#     num_workers=4,
#     collate_fn=utils.collate_fn,
# )

# get the model using our helper function
model = get_model_instance_segmentation(num_classes)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# let's train it just for 2 epochs
num_epochs = 2

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    # evaluate(model, data_loader_test, device=device)

print("That's it!")
