import argparse
import datetime
from tkinter import N
from turtle import color
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.cuda.amp import autocast as autocast
from dataloaders import *
from SEMC_network import SEMC_Net,GumbelSparseGate
from utils import *
from torch.utils.data import DataLoader
from datetime import datetime


normalize = transforms.Normalize(
    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

def apply_clahe(img):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    """
    img_np = np.array(img)
    if len(img_np.shape) == 2:
        pass
    else:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_np)
    img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(img_clahe)

data_transforms = transforms.Compose([
    transforms.Lambda(lambda img: apply_clahe(img)),
    transforms.Resize((512, 512)),
    transforms.CenterCrop((512, 512)),
    transforms.ToTensor(),
])

parser = argparse.ArgumentParser(description='PyTorch SEMC_Liver Test')

parser.add_argument('--epochs', default=200, help='epoch for augmented training')

parser.add_argument('--batch_size', default=16)

parser.add_argument('--learning_rate', default=0.001)

parser.add_argument('--seed', default=123, help='keep all seeds fixed')

args = parser.parse_args()

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

gate = GumbelSparseGate(in_channels=2048, num_experts=3).to(device)

def main():
    """
    Main function for testing the SEMC model on the dataset.
    Loads the test set, model, and runs evaluation.
    """
    set_seed(args.seed)

    test_set = UltrasoundDataset(root_dir='./datasets/liver/test', transform=data_transforms)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=16,drop_last=True)
    model=SEMC_Net().to(device)
    test(test_loader, model)
          
def test(val_loader, model):
    """
    Evaluate the model on the validation/test dataset.
    Computes accuracy, precision, recall, and F1-score.
    """
    all_preds = []
    all_targets = []
    checkpoint = torch.load('checkpoint/liver.pth.tar', map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    with torch.no_grad():
        for _,(images, target) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            *_,logits_list, deep_out, _ = model(images,target)
            gated_out = gate(deep_out, logits_list, hard=False)
           
            preds = gated_out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
    accuracy = 100.0 * accuracy_score(all_targets, all_preds)
    precision = 100.0 * precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = 100.0 * recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = 100.0 * f1_score(all_targets, all_preds, average='macro', zero_division=0)

    print(f"Test  Accuracy {accuracy:.3f}  Precision {precision:.3f}  Recall {recall:.3f}  F1 {f1:.3f}")  

if __name__ == '__main__':
    clock_start = datetime.now()
    main()
    clock_end = datetime.now()
    print(clock_end - clock_start)
