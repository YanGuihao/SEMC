import argparse
import datetime
import os
from tkinter import N
from turtle import color
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torch.cuda.amp import autocast as autocast, GradScaler
from dataloaders import *
from SEMC_network import SEMC_Net,GumbelSparseGate,AdaptiveLossWeight 
from utils import *
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
from loss import GPaCoLoss
import glob


normalize = transforms.Normalize(
    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))



def apply_clahe(img):
    img_np = np.array(img)
    if len(img_np.shape) == 2:
        pass
    else:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_np)
    img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(img_clahe)

data_transforms = {
    'advanced_train': transforms.Compose([
    transforms.Lambda(lambda img: apply_clahe(img)),              
    transforms.RandomHorizontalFlip(p=0.5),                         
    transforms.RandomRotation(degrees=10),                         
    transforms.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),         
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),   
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.4, scale=(0.02, 0.08))             
    ]),
    'test': transforms.Compose([
    transforms.Lambda(lambda img: apply_clahe(img)),
    transforms.Resize((512, 512)),
    transforms.CenterCrop((512, 512)),
    transforms.ToTensor(),
])
}


parser = argparse.ArgumentParser(description='PyTorch SEMC_Liver Training')
parser.add_argument('--outf', default='./outputs/', help='folder to output images and model checkpoints')
parser.add_argument('--pre_epoch', default=0, help='epoch for pre-training')
parser.add_argument('--epochs', default=200, help='epoch for augmented training')
parser.add_argument('--batch_size', default=16)
parser.add_argument('--learning_rate', default=0.001)
parser.add_argument('--seed', default=123, help='keep all seeds fixed')
parser.add_argument('--re_train', default=True, help='implement cRT')
parser.add_argument('--cornerstone', default=180)
parser.add_argument('--num_exps', default=3, help='exps')
parser.add_argument('-e',
                    '--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-p',
                    '--print-freq',
                    default=10,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')
args = parser.parse_args()

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
samples_per_class = torch.tensor([652, 239, 688, 303, 663, 580, 2940], dtype=torch.float)
weights = 1.0 / samples_per_class
weights = weights / weights.sum()

criterion = nn.CrossEntropyLoss(weight=weights.to(device)).to(device)
Gpacoloss = GPaCoLoss(alpha=0.05, temperature=0.2, K=2048, smooth=0.1).to(device)
gate = GumbelSparseGate(in_channels=2048, num_experts=3).to(device)
adaptive_weight= AdaptiveLossWeight(128).to(device)
def main():

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if not os.path.exists(args.outf):
        os.makedirs(args.outf)   

    set_seed(args.seed)

    train_set = UltrasoundDataset(root_dir='./datasets/liver2/train', transform=data_transforms['advanced_train'])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=16,drop_last=True)

    test_set = UltrasoundDataset(root_dir='./datasets/liver2/test', transform=data_transforms['test'])
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=16,drop_last=True)

    best_acc = .0

    model=SEMC_Net().to(device)

    optimizer_feat = optim.SGD(
        model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    optimizer_crt = optim.SGD(
        model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler_feat = CosineAnnealingLRWarmup(
        optimizer=optimizer_feat,
        T_max=args.epochs - 20,
        eta_min=0.0,
        warmup_epochs=5,
        base_lr=args.learning_rate,
        warmup_lr=0.005
    )
    scheduler_crt = CosineAnnealingLRWarmup(
        optimizer=optimizer_crt,
        T_max=20,
        eta_min=0.0,
        warmup_epochs=5,
        base_lr=args.learning_rate,
        warmup_lr=0.01
    )

    scaler = GradScaler()
    train_losses = []
    for epoch in range(0, args.epochs):  # args.start_epoch

        train(train_loader, model, scaler, optimizer_feat, epoch, train_losses)
        acc = validate(test_loader, model, criterion, epoch)

        if epoch >= args.cornerstone:
            scheduler_crt.step()
        else:
            scheduler_feat.step()

        # Best accuracy
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        if(is_best):
            print("Best epoch: ",epoch+1)

        save_checkpoint(
            {
                'epoch': epoch + 1,
                'architecture': "resnet50",
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
            }, is_best, current_time=current_time,epoch=epoch)
    plt.figure()
    plt.plot(train_losses, label="Train Loss", color='b')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)

    loss_dir = os.path.join('loss', current_time)
    os.makedirs(loss_dir, exist_ok=True) 

    plt.savefig(os.path.join(loss_dir, "train_loss_curve.png"))
    print("训练损失图已保存为 train_loss_curve.png")
    print("Training Finished, TotalEPOCH=%d" % args.epochs)



def train(train_loader, model, scaler, optimizer, epoch, train_losses):
    model.train()
    all_preds = []
    all_targets = []
    epoch_loss = 0.0

    for _, (images, target) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad()
        features, targets, sup_logits, logits_list, deep_out,proj_feat = model(images,target)
      #  proto_moco_contrast.update_prototypes(feat_expert.detach(), target)
        gated_out= gate(deep_out, logits_list, hard=False)
        loss1 = criterion(gated_out, target)
        loss2 = Gpacoloss(features,targets,sup_logits)
     
        loss = adaptive_weight(loss1, loss2, proj_feat)  

        epoch_loss += loss.item()

        preds = gated_out.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    #  sklearn 
    accuracy = 100.0 * accuracy_score(all_targets, all_preds)
    precision = 100.0 * precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = 100.0 * recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = 100.0 * f1_score(all_targets, all_preds, average='macro', zero_division=0)

    print(f"Epoch Train [{epoch+1}]  Accuracy {accuracy:.2f}  Precision {precision:.2f}  Recall {recall:.2f}  F1 {f1:.2f}")

    epoch_loss /= len(train_loader)
    train_losses.append(epoch_loss)     

def validate(val_loader, model, criterion, epoch):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0

    with torch.no_grad():
        for _,(images, target) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            *_,logits_list, deep_out,_ = model(images,target)
            gated_out= gate(deep_out, logits_list, hard=False)
            loss = criterion(gated_out, target)

            total_loss += loss.item()

            preds = gated_out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    accuracy = 100.0 * accuracy_score(all_targets, all_preds)
    precision = 100.0 * precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = 100.0 * recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = 100.0 * f1_score(all_targets, all_preds, average='macro', zero_division=0)

    print(f"Epoch Test [{epoch+1}]  Accuracy {accuracy:.3f}  Precision {precision:.3f}  Recall {recall:.3f}  F1 {f1:.3f}")

    return accuracy

def save_checkpoint(state, is_best, current_time, epoch, filename='liver'):
    checkpoint_dir = os.path.join('checkpoint', current_time)
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_model_path = os.path.join(checkpoint_dir, f'{filename}_best_epoch_{epoch+1}.pth.tar')

    if is_best:
        old_files = glob.glob(os.path.join(checkpoint_dir, f'{filename}_best_epoch_*.pth.tar'))
        for f in old_files:
            try:
                os.remove(f)
            except Exception as e:
                print(f"Warning: failed to remove old checkpoint {f}: {e}")

        torch.save(state, best_model_path)



if __name__ == '__main__':
    clock_start = datetime.now()
    main()
    clock_end = datetime.now()
    print(clock_end - clock_start)
