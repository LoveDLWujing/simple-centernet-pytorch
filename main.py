from dataset import UODAC
from losses import CenterNetLoss
from model import resnet, mobilenet, dlav0
from decode import ctdet_decode
import utils

import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    #######################
    # config
    #######################
    dataset_dir = '/home/ubuntu/raid/513wj/train/train'
    train_dataset = UODAC(dataset_dir, mode='train')
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=20,
        shuffle=True,
        num_workers=10,
        pin_memory=False
    )
    val_dataloader = torch.utils.data.DataLoader(
        UODAC(dataset_dir, mode='val'),
        batch_size=1,
        shuffle=False,
        num_workers=10,
        pin_memory=False
    )

    model = dlav0(34, {'hm': train_dataset.class_num, "wh": 2, "reg": 2}, 256)
    # model.load_state_dict(torch.load("resnet18.pt"))
    if torch.cuda.is_available():
        model = model.cuda()

    ########################
    # train
    ########################
    loss_fn = CenterNetLoss()
    optim = torch.optim.Adam(model.parameters(), lr=3e-4)
    for epoch in range(1, 70):
        total_loss = 0.
        for img, label in tqdm.tqdm(train_dataloader):
            if torch.cuda.is_available():
                img = img.cuda()
                label = label.cuda()
            output = model(img)[0]
            optim.zero_grad()
            loss = loss_fn(output, label)
            total_loss = total_loss + loss
            loss.backward()
            optim.step()
        print(total_loss / len(train_dataloader))
        torch.save(model.state_dict(), 'dlav0.pt')

    #######################
    # test
    #######################
    # with torch.no_grad():
    #     for img, _ in val_dataloader:
    #         if torch.cuda.is_available():
    #             img = img.cuda()
    #         output = model(img)[0]
    #         detections = ctdet_decode(output['hm'], output['wh'], output['reg'])
    #         detections[..., :-2] *= 4
    #         utils.draw_detections(((img[0].permute(1, 2, 0).cpu().detach().numpy() + 0.5) * 255).astype(np.uint8),
    #                               detections[0, :10].cpu().detach().numpy(),
    #                         train_dataset.class_names)
    #         plt.show()

    #######################
    # export onnx model
    #######################
    # torch.onnx.export(
    #     model=model.cpu(),
    #     args=(torch.randn(1, 3, 512, 512),),
    #     f='resnet18.onnx',
    #     export_params=True,
    #     verbose=True,
    #     training=False,
    #     input_names=["input"],
    #     output_names=['hm', 'reg', 'wh']
    # )
