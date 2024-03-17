import torch
from torchvision.datasets import CIFAR10
from google.colab.patches import cv2_imshow
from torchvision import transforms
from model.DDPM_total import UNet, train, sample


device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = CIFAR10(
    root="./data", train=True, download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ]) 
)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=32, shuffle=True, num_workers=8
)

model = UNet(128, 3, 3, 128).to(device)
optim = torch.optim.Adam(model.parameters(), lr=0.0002)

for e in range(100):
    model.train()
    for i, (x, _) in enumerate(dataloader, 1):
        optim.zero_grad()
        x = x.to(device)
        loss = train(model, x)
        loss.backward()
        optim.step()
        print("\r[Epoch: {} , Iter: {}/{}]  Loss: {:.3f}".format(e, i, len(dataloader), loss.item()), end='')
    model.eval()
    with torch.no_grad():
        x_T = torch.randn(5, 3, 32, 32).to(device)
        x_0 = sample(model, x_T)
        x_0 = x_0.permute(0, 2, 3, 1).clamp(0, 1).detach().cpu().numpy() * 255
        for i in range(5):
            cv2_imshow(x_0[i])