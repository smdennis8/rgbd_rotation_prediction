import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import scipy

import image
from rotation_dataset import RotationDataset
from rotation_model import RotationNet
#from segmentation_helper import check_dataset, check_dataloader, show_mask


def iou(prediction, target):
    """
    In:
        prediction: Tensor [batchsize, rotmatrix ], predicted rotation
        target: Tensor [batchsize,rotmatix], ground truth rotation.
    Out:
        batch_ious: a list of floats, storing IoU on each batch.
    Purpose:
        Compute IoU on each data and return as a list.
    """
    #_, pred = torch.max(prediction, dim=1)
    batch_num = prediction.shape[0]
    pred = prediction.detach().numpy()
    lbl = target.detach().numpy()
    dist_list = []
    batch_ious = list()
    for batch_id in range(batch_num):
        pred_single = np.array(pred[batch_id].flatten().reshape(3,3))
        lbl_single = np.array(lbl[batch_id].flatten().reshape(3,3))
        pred_single = torch.from_numpy(pred_single).unsqueeze(dim=0)
        lbl_single = torch.from_numpy(lbl_single).unsqueeze(dim=0)
        dist = geodesic_dist(pred_single, lbl_single)  # Input Shape: (1, 3, 3)
        dist_list.append(dist.item())
        
   
    return dist


def adjoint(A, E, f):
    A_H = A.T.conj().to(E.dtype)
    n = A.size(0)
    M = torch.zeros(2*n, 2*n, dtype=E.dtype, device=E.device)
    M[:n, :n] = A_H
    M[n:, n:] = A_H
    M[:n, n:] = E
    return f(M)[:n, n:].to(A.dtype)


def logm_scipy(A):
    return torch.from_numpy(scipy.linalg.logm(A.cpu(), disp=False)[0]).to(A.device)


class Logm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        assert A.ndim == 2 and A.size(0) == A.size(1)  # Square matrix
        assert A.dtype in (torch.float32, torch.float64, torch.complex64, torch.complex128)
        ctx.save_for_backward(A)
        return logm_scipy(A)
    @staticmethod
    def backward(ctx, G):
        A, = ctx.saved_tensors
        return adjoint(A, G, logm_scipy)

# logm = Logm.apply

# -----------------------


def geodesic_dist(mata: torch.Tensor, matb: torch.Tensor) -> torch.Tensor:
    """Calculates the geodesic distance between 2 3x3 rotation matrices
​
​
    Args:
        mata: Rotation Matix. Shape: [B, 3, 3]
        matb: Rotation Matix. Shape: [B, 3, 3]
​
    Returns:
        torch.float: Geodesic distance
​
    Reference:
        Mahendran, Siddharth, Haider Ali, and René Vidal. "3d pose regression using convolutional neural networks."
        Proceedings of the IEEE International Conference on Computer Vision Workshops. 2017.
    """
    assert len(mata.shape) == 3 and mata.shape[1] == 3 and matb.shape[2] == 3
    assert len(matb.shape) == 3 and mata.shape[1] == 3 and matb.shape[2] == 3
    dists = 0
    for mata_, matb_ in zip(mata, matb):
        d2 = Logm.apply(mata_ @ matb_.T)
        dist = torch.linalg.norm(d2.flatten(), ord=2) / torch.sqrt(torch.tensor(2).float())
        dists += dist
    return dists / mata.shape[0]  # div by batch size


def save_chkpt(model, epoch, val_miou):
    """
    In:
        model: MiniUNet instance in this homework, trained model.
        epoch: int, current epoch number.
        val_miou: float, mIoU of the validation set.
    Out:
        None.
    Purpose:
        Save parameters of the trained model.
    """
    state = {'model_state_dict': model.state_dict(),
             'epoch': epoch,
             'model_miou': val_miou, }
    torch.save(state, 'checkpoint.pth.tar')
    print("checkpoint saved at epoch", epoch)
    return


def load_chkpt(model, chkpt_path):
    """
    In:
        model: MiniUNet instance in this homework to accept the saved parameters.
        chkpt_path: string, path of the checkpoint to be loaded.
    Out:
        model: MiniUNet instance in this homework, with its parameters loaded from the checkpoint.
        epoch: int, epoch at which the checkpoint is saved.
        model_miou: float, mIoU on the validation set at the checkpoint.
    Purpose:
        Load model parameters from saved checkpoint.
    """
    checkpoint = torch.load(chkpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    model_miou = checkpoint['model_miou']
    print("epoch, model_miou:", epoch, model_miou)
    return model, epoch, model_miou


def save_learning_curve(train_loss_list, train_miou_list, val_loss_list, val_miou_list):
    """
    In:
        train_loss, train_miou, val_loss, val_miou: list of floats, where the length is how many epochs you trained.
    Out:
        None.
    Purpose:
        Plot and save the learning curve.
    """
    epochs = np.arange(1, len(train_loss_list)+1)
    plt.figure()
    lr_curve_plot = plt.plot(epochs, train_loss_list, color='navy', label="train_loss")
    plt.plot(epochs, train_miou_list, color='teal', label="train_mIoU")
    plt.plot(epochs, val_loss_list, color='orange', label="val_loss")
    plt.plot(epochs, val_miou_list, color='gold', label="val_mIoU")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.xticks(epochs, epochs)
    plt.yticks(np.arange(10)*0.1, [f"0.{i}" for i in range(10)])
    plt.xlabel('epoch')
    plt.ylabel('mIoU')
    plt.grid(True)
    plt.savefig('learning_curve.png', bbox_inches='tight')
    plt.show()


def save_prediction(model, device, dataloader, dataset_dir):
    """
    In:
        model: MiniUNet instance in this homework, trained model.
        device: torch.device instance.
        dataloader: Dataloader instance.
        dataset_dir: string, path of the val or test folder.

    Out:
        None.
    Purpose:
        Visualize and save the predicted masks.
    """
    pred_dir = dataset_dir + "pred/"
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    print(f"Save predicted masks to {pred_dir}")

    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch_id, sample_batched in enumerate(dataloader):
            data = sample_batched['input'].to(device)
            output = model(data)
            _, pred = torch.max(output, dim=1)
            for i in range(pred.shape[0]):
                scene_id = batch_id * dataloader.batch_size + i
                mask_name = pred_dir + str(scene_id) + "_pred.png"
                mask = pred[i].cpu().numpy()
                #show_mask(mask)
                image.write_mask(mask, mask_name)


def train(model, device, train_loader, criterion, optimizer):
    """
    Loop over each sample in the dataloader.
    Do forward + backward + optimize procedure. Compute average sample loss and mIoU on the dataset.
    """
    model.train()
    train_loss, train_iou = 0, 0
    batches = len(train_loader)
    
    #first dim of input gives number of samples
    num_batches = 0
    for i, data in enumerate(train_loader, 0):
         # get the inputs; data is a list of [inputs, labels]
        inputs = data.get('input')
        labels = data.get('target')
       
        
        # forward + backward + optimize (feed a batch of input into the model to get the output)
        btchsize = inputs.shape[0]
        inputs = inputs.reshape(btchsize,-1)
        labels = labels.reshape(btchsize,3,3)
        outputs = model(inputs)
       
        # compute average loss of the batch using criterion()
        loss = criterion(outputs, labels)
        #compute mIoU of the batch using iou()
        mIoU = iou(outputs,labels)
        #store loss and mIoU of the batch for computing statistics
        
        #train_iou += sum(mIoU)/len(inputs)
        train_loss += loss.item() #avg over a batch so times batch size gives total loss per batch
        print(loss.item())
        # zero the parameter gradients
        optimizer.zero_grad()
        #compute the gradients using loss.backward()
        loss.backward()
        #updates the parameters of the model using optimizer.step()
        optimizer.step()
        num_batches +=1
    #compute the average loss and mIoU of the dataset to return  
    #train_iou = train_iou/batches #divide by number of samples
    train_loss = train_loss/num_batches # by dataset size (size 300)
    
    train_iou = 0
    
    return train_loss, train_iou


def val(model, device, val_loader, criterion):
    """
    Similar to train(), but no need to backward and optimize.
    """
    model.eval()
    val_loss, val_iou = 0, 0
    batches = len(val_loader)
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
         # get the inputs; data is a list of [inputs, labels]
            inputs = data.get('input')
            labels = data.get('target')
            
            # forward + backward + optimize (feed a batch of input into the model to get the output)
            outputs = model(inputs)
            
            # compute average loss of the batch using criterion()
            loss = criterion(outputs, labels)
           
            #compute mIoU of the batch using iou()
            #mIoU = iou(outputs,labels)
            
            #store loss and mIoU of the batch for computing statistics
            #val_iou += sum(mIoU)/len(inputs)
            val_loss += loss.item()
      
    #compute the average loss and mIoU of the dataset to return  
    val_iou = val_iou/batches
    val_loss = val_loss/5
    return val_loss, val_iou


def main():
    # Check if GPU is being detected
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # Define directories
    root_dir = './dataset/'
    train_dir = root_dir + 'train1/'
    #val_dir = root_dir + 'val/'
    #test_dir = root_dir + 'test/'

    #Create Datasets.
    train_dataset = RotationDataset(train_dir, True )
    #check_dataset(train_dataset)
    #val_dataset = RotationDataset(val_dir, True )
    #check_dataset(val_dataset)
    #test_dataset = RotationDataset(test_dir, False )
    #check_dataset(test_dataset)

    #Prepare Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=0)
    #check_dataloader(train_loader)
    #val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0)
    #check_dataloader(val_loader)
    #test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)
    #check_dataloader(test_loader)

    # Prepare model
    model = RotationNet()

    # Define criterion and optimizer
    #criterion = torch.nn.MSELoss()
    criterion = geodesic_dist #gives pointer to function
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9) #best was .001

    # Train and validate the model
    train_loss_list, train_miou_list, val_loss_list, val_miou_list = list(), list(), list(), list()
    epoch, max_epochs = 1, 15 # TODO: you may want to make changes here
    best_miou = float('-inf')
    while epoch <= max_epochs:
        print('Epoch (', epoch, '/', max_epochs, ')')
        train_loss, train_miou = train(model, device, train_loader, criterion, optimizer)
        #val_loss, val_miou = val(model, device, val_loader, criterion)
        train_loss_list.append(train_loss)
        train_miou_list.append(train_miou)
       # val_loss_list.append(val_loss)
        #val_miou_list.append(val_miou)
        print('Train loss & mIoU: %0.2f %0.2f' % (train_loss, train_miou))
        #print('Validation loss & mIoU: %0.2f %0.2f' % (val_loss, val_miou))
        print('---------------------------------')
        #if val_miou > best_miou:
        #    best_miou = val_miou
        #    save_chkpt(model, epoch, val_miou)
        epoch += 1

    # Load the best checkpoint, use save_prediction() on the validation set and test set
    model, epoch, best_miou = load_chkpt(model, 'checkpoint.pth.tar')
    #save_prediction(model, device, val_loader, val_dir)
    #save_prediction(model, device, test_loader, test_dir)
    save_learning_curve(train_loss_list, train_miou_list, val_loss_list, val_miou_list)


if __name__ == '__main__':
    main()
