import torch as t
from torchvision import transforms as ts
from PIL import Image
from resnet import resnet50
from deepdream import dream

img_ts = ts.Compose([
    ts.ToTensor(),
    ts.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

input_img = Image.open('./sky.jpg')
input_tensor = img_ts(input_img).unsqueeze(0)
input_np = input_tensor.numpy()

model = resnet50(pretrained=True)
if t.cuda.is_available():
    model = model.cuda()
for param in model.parameters():
    param.requires_grad = False

dream(model, input_np)
