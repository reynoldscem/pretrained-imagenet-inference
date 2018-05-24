from torchvision.models import resnet
import numpy as np
import pickle
import torch

BATCH_SIZE = 128


def process_batch(X):
    mean = np.array([0.485, 0.456, 0.406])
    mean = mean[np.newaxis, :, np.newaxis, np.newaxis]

    std = np.array([0.229, 0.224, 0.225])
    std = std[np.newaxis, :, np.newaxis, np.newaxis]

    X = X / 255.
    X = X - mean
    X = X / std

    return torch.Tensor(X)


def main():
    with open('./val224.pkl', 'rb') as fd:
        data = pickle.load(fd)

    splits = data['target'].shape[0] // BATCH_SIZE
    loader = zip(*(
        np.array_split(data[key], splits)
        for key in ('data', 'target')
    ))

    model = resnet.resnet18(pretrained=True).cuda()

    correct = 0
    seen = 0
    for X, y in loader:
        X = process_batch(X)
        y = y.astype(np.int64)
        y = torch.from_numpy(y)
        X, y, = X.cuda(), y.cuda()

        output = model(X)
        pred = torch.argmax(output, dim=-1)

        correct += (pred == y).cpu().sum().item()
        seen += y.size()[0]
        acc = 100 * (correct / seen)

        print('{} / {},\t{:.6f}%'.format(correct, seen, acc))


if __name__ == '__main__':
    main()
