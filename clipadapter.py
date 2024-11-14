import warnings
import torch
import clip
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn import metrics
import copy
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class TextEncoder(nn.Module):

    def __init__(self, classnames, clip_model):
        super().__init__()
        self.classnames = classnames
        self.clip_model = clip_model
        self.dtype = clip_model.dtype
    
    def forward(self):
        temp = 'a photo of a {}.'
        prompts = [temp.format(c.replace('_', ' ')) for c in self.classnames]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to('cuda')
        text_features = self.clip_model.encode_text(prompts)
        x = text_features
        return x
    
class CustomCLIP(nn.Module):

    def __init__(self, clip_model,classnames,visual_ratio=0.8,text_ratio=0.0):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(classnames, clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.adapter = Adapter(512, 2).to(clip_model.dtype)
        self.adapter_text = Adapter(512, 2).to(clip_model.dtype)
        self.visual_ratio=visual_ratio
        self.text_ratio = text_ratio

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        x = self.adapter(image_features)

        ratio = self.visual_ratio
        image_features = ratio * x + (1 - ratio) * image_features

        text_features = self.text_encoder()
        #
        ratio_text = self.text_ratio
        text_features_ad = self.adapter_text(text_features)
        text_features = ratio_text * text_features_ad + (1 - ratio_text) * text_features
        #
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits