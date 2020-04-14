import argparse
import yaml
import torch
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import matplotlib.pyplot as plt

from datasets.images import ImageLabel
from modelzoo.classifiers import ConvClassifier

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as yml:
        config = yaml.load(yml, Loader=yaml.FullLoader)

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    transforms = transforms.Compose([
        transforms.RandomRotation(90),
        transforms.ToTensor()  # HxWxC -> CxHxW
    ])
    dataset = ImageLabel(df_loc=config['df_loc'], transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=config['parameters']['batch_size'], shuffle=True)

    model = ConvClassifier(3,64,64)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=config['parameters']['lr'])
    criterion = torch.nn.NLLLoss()

    writer = SummaryWriter()
    i = 0

    for e in range(config['parameters']['epochs']):
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            fig = plt.figure()
            plt.title(
                'Probability Apple {}/Banana {}'.format(
                    torch.exp(out[0][0]).detach().cpu(), 
                    torch.exp(out[0][1]).detach().cpu()
            ))
            plt.imshow(imgs[0].detach().cpu().numpy().transpose(2,1,0))
            writer.add_figure('probability', fig, i)

            writer.add_scalar('train/loss', loss.item(), i)
            i += 1
