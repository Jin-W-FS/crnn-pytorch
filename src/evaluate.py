import os, sys, time

import torch
from torch.utils.data import DataLoader
from torch.nn import CTCLoss
from tqdm import tqdm

from lpr_dataset import LPRDataset
from model import CRNN
from ctc_decoder import ctc_decode
from config import evaluate_config as config

torch.backends.cudnn.enabled = False


def evaluate(crnn, dataloader, criterion=None,
             max_iter=None, decode_method='beam_search', beam_size=10,
             logfile=None):
    crnn.eval()

    tot_loss = 0
    Tp = Tn_1 = Tn_2 = 0
    test_results = []
    t1 = time.time()

    pbar_total = max_iter if max_iter else len(dataloader)
    pbar = tqdm(total=pbar_total, desc="Evaluate")

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if max_iter and i >= max_iter:
                break
            device = 'cuda' if next(crnn.parameters()).is_cuda else 'cpu'

            images, targets, target_lengths = [d.to(device) for d in data[:3]]
            fnames = data[3]

            logits = crnn(images)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)

            batch_size = images.size(0)
            input_lengths = torch.LongTensor([logits.size(0)] * batch_size)

            if criterion:
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
                tot_loss += loss.item()

            preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size)
            reals = targets.cpu().numpy().tolist()
            target_lengths = target_lengths.cpu().numpy().tolist()

            target_length_counter = 0
            for i, (pred, target_length) in enumerate(zip(preds, target_lengths)):
                real = reals[target_length_counter:target_length_counter + target_length]
                target_length_counter += target_length
                fname = fnames[i]

                if len(pred) != len(real):
                    Tn_1 += 1
                elif pred == real:
                    Tp += 1
                else:
                    Tn_2 += 1

                lb = LPRDataset.decode(pred)
                tg = LPRDataset.decode(real)
                test_results.append(('FT'[lb==tg], tg, lb, fname))

            pbar.update(1)
        pbar.close()

    t2 = time.time()
    Total = (Tp + Tn_1 + Tn_2)
    Acc = Tp * 1.0 / Total
    Loss = tot_loss / Total

    print("[Info] Test Accuracy: {} [{}:{}:{}:{}]".format(Acc, Tp, Tn_1, Tn_2, Total))
    print("[Info] Test Speed: {}s 1/{}]".format((t2 - t1) / Total, Total))

    if logfile:
        with open(logfile, 'w') as outf:
            for v in sorted(test_results): print(*v, sep='\t', file=outf)

    return {'loss':Loss, 'acc':Acc}

def main():
    eval_batch_size = config['eval_batch_size']
    cpu_workers = config['cpu_workers']
    reload_checkpoint = config['reload_checkpoint']
    tests_data_dir = config['tests_data_dir']

    img_height = config['img_height']
    img_width = config['img_width']
    img_channel = config['img_channel']
    img_shape = (img_height, img_width, img_channel)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    test_dataset = LPRDataset(img_dir=tests_data_dir, mode='test', img_shape=img_shape, lpr_max_len=8)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=cpu_workers,
        collate_fn=LPRDataset.collate_fn)

    num_class = len(LPRDataset.CHARS)
    crnn = CRNN(img_channel, img_height, img_width, num_class,
                map_to_seq_hidden=config['map_to_seq_hidden'],
                rnn_hidden=config['rnn_hidden'],
                leaky_relu=config['leaky_relu'])
    crnn.load_state_dict(torch.load(reload_checkpoint, map_location=device))
    crnn.to(device)

    criterion = CTCLoss(reduction='sum')
    criterion.to(device)

    evaluation = evaluate(crnn, test_loader, criterion,
                          decode_method=config['decode_method'],
                          beam_size=config['beam_size'],
                          logfile='test_results.log')


if __name__ == '__main__':
    main()
