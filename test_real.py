from codes.models.modules.arch_spkdeblur_transformer import SpkDeblurNet
from codes.utils.spike_utils import load_vidar_dat,middleTFI
import torch
import cv2

model = SpkDeblurNet(S_in_chs=56).cuda()
model.load_state_dict(torch.load("./pretrained/gopro_100000_SpikeDeblur.pth"))

x = cv2.imread("./000.jpg")
x = cv2.resize(x, (800, 500))/255

y = load_vidar_dat("./000.dat", width=400, height=250)
x = torch.tensor(x).permute(2,0,1).unsqueeze(0).float().cuda()
y = torch.tensor(y).unsqueeze(0).cuda()
y = y[:,100-28:100+28]

tfi = middleTFI(y[0].cpu().numpy(), 28)
tfi = torch.tensor(tfi).unsqueeze(0).unsqueeze(0).cuda()
print(x.shape, y.shape, tfi.shape)

out = model(x,y,tfi)
sharp = out[0].detach().cpu().numpy()

# save
sharp = sharp[0].transpose(1,2,0)
sharp = sharp * 255
cv2.imwrite("sharp.jpg", sharp)