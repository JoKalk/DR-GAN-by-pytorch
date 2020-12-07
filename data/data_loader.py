import collections
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

def CreateDataLoader(opt, mask):
    """
    Return the dataloader according to the opt.
    """
    import sys
    sys.path.append('/home/zhangjunhao/data')
    from dataset_custom import FDDataset
    transform = transforms.Compose([
        transforms.Scale((100, 100)),       #Switch to the transforms.Resize on the service # 100, 100, 96
        transforms.RandomCrop(96),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)) # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    single = True if opt.model=='single' else False
    dataset = FDDataset(root=opt.dataroot, train=opt.is_Train, transform=transform, single=single, mask=mask)
    def my_collate(batch):
        if isinstance(batch[0], collections.Sequence):
            return [default_collate(b) for b in batch]
        return default_collate(batch)

    dataloader = DataLoader(dataset, batch_size=opt.batchsize, shuffle=opt.is_Train, num_workers=0, collate_fn=my_collate)
    return dataloader
