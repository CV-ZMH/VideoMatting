import cv2
import torch
import torch.nn as nn

from modnet.models.modnet import MODNet


class Predictor:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(self, weight_path):
        self.model = self.load_model(weight_path)

    def set_input_size(self, image):
        h, w = image.shape[:2]
        if w >= h:
            rh = 512
            rw = int((w / h) * 512)
        else:
            rw = 512
            rh = int((h / w) * 512)
        self.inp_h = rh - rh % 32
        self.inp_w = rw - rw % 32

    def load_model(self, weight_path):
        checkpoint = torch.load(weight_path, map_location='cpu')
        model = MODNet(backbone_pretrained=False) # create architecture
        model = nn.DataParallel(model)
        model.load_state_dict(checkpoint) #
        return model.eval().to(self.device)

    @torch.no_grad()
    def predict(self, image):
        """ Prediction on image"""
        if not hasattr(self, 'inp_w'): self.set_input_size(image)
        image = cv2.resize(image, (self.inp_w, self.inp_h), cv2.INTER_AREA)
        inp_tensor = self._preprocess(image)
        *_, matte_tensor = self.model(inp_tensor, True)
        matte_tensor = matte_tensor.repeat(1, 3, 1, 1)
        matte = matte_tensor[0].data.cpu().numpy().transpose(1, 2, 0)
        return matte

    def _preprocess(self, image):
        inp_tensor = torch.as_tensor(image, dtype=torch.float32)
        inp_tensor = inp_tensor.permute(2, 0, 1).to(self.device)
        inp_tensor = (inp_tensor - 127.5) / 127.5 # normalize (mean=0.5 / std=0.5)
        return inp_tensor.unsqueeze_(0)
 
