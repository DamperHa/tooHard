import torch
import torchvision.models as models


model = models.vgg16(weights='IMAGENET1K_V1')


# 保存模型的操作
torch.save(model.state_dict(), 'model_weights.pth')


# 读取模型的操作
model = models.vgg16() # we do not specify ``weights``, i.e. create untrained model
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()