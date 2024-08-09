
import cv2
from PIL import Image
from torchvision import transforms

INPUT_SIZE = (256, 256)

def get_transform():
    transform_image_list = [
        # transforms.Resize((256, 256), 3),
        # transforms.Resize(INPUT_SIZE, interpolation=transforms.InterpolationMode.BICUBIC), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(transform_image_list)

def preprocess(img_path, data_transform):
    img_cv = cv2.imread(img_path)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_cv = cv2.resize(img_cv, INPUT_SIZE,  cv2.INTER_NEAREST)
    image = Image.fromarray(img_cv, 'RGB')
    # ori_img = Image.open(img_path)
    img = data_transform(image)
    return img.unsqueeze(dim=0).cpu().numpy(), img_cv