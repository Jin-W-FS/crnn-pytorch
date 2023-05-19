#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

def custom(pretrained, **kwargs):
    from model import CRNN
    from lpr_dataset import LPRDataset
    from ctc_decoder import ctc_decode

    class Options:
        cuda = True
        decode_method = 'beam_search'
        beam_size = 10
        img_shape = (32, 140, 3)    # h, w, c
        map_to_seq_hidden = 64
        rnn_hidden = 256
        leaky_relu = False

    opt = Options()
    opt.__dict__.update(kwargs)
    if not torch.cuda.is_available(): opt.cuda = False

    img_height, img_width, img_channel = opt.img_shape
    num_class = len(LPRDataset.CHARS)
    model = CRNN(img_channel, img_height, img_width, num_class,
                 map_to_seq_hidden=opt.map_to_seq_hidden,
                 rnn_hidden=opt.rnn_hidden,
                 leaky_relu=opt.leaky_relu)
    if opt.cuda:
        model = model.cuda()
    if pretrained:
        model.load_state_dict(
            torch.load(pretrained, map_location=(None if opt.cuda else 'cpu'))
        )

    class Detector(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.model.eval()

        def _apply(self, fn):
            # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
            self = super()._apply(fn)
            return self

        def forward(self, img, **kw):
            with torch.no_grad():
                return self._forward(img, **kw)

        def _forward(self, img, **kw):
            img = LPRDataset.loadImage(img, opt.img_shape)
            img = LPRDataset.transform(img)
            img = torch.from_numpy(img)
            img = img.unsqueeze(0)
            if opt.cuda: img = img.cuda()

            logits = self.model(img)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)
            preds = ctc_decode(log_probs, method=opt.decode_method, beam_size=opt.beam_size)
            label = LPRDataset.decode(preds[0])
            return label

    return Detector(model)


if __name__ == '__main__':
    import os, sys, time
    from imutils import paths

    args = {
        'pretrained' : 'weights/crnn_lpr-230518.pt',
        'cuda' : True,
        'decode_method' : 'beam_search',
        'beam_size' : 10,
    }
    for i in range(1, len(sys.argv)):
        s = sys.argv[i]
        if '=' not in s: break
        k, v = s.split('=')
        try:
            v = eval(v)
        except:
            v = str(v)
        args[k] = v
    images = sys.argv[i:]

    print(f'Init model with {args} on {images}')
    model = torch.hub.load(os.getcwd(), 'custom', source='local', **args)

    img_paths = []
    for p in images:
        if os.path.splitext(p)[-1] in paths.image_types:
            img_paths.append(p)
        else:
            img_paths.extend(paths.list_images(p))

    for f in img_paths:
        t0 = time.time()
        rlt = model(f)
        t1 = time.time()
        print(f'{f} {rlt} {(t1-t0)*1000:.1f}ms')

