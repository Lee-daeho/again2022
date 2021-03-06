import cv2.cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
from collections import Counter
import os
import warnings
import random
# Custom
from config import *
from models.query_models import VAE, Discriminator, GCN, CVAE, C_Discriminator
from load_dataset import CamDataset, NewDataset
from data.sampler import SubsetSequentialSampler

from kcenterGreedy import kCenterGreedy
from cam_override import GradCAM
import matplotlib.pyplot as plt
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.preprocessing import minmax_scale


def BCEAdjLoss(scores, lbl, nlbl, l_adj):
    lnl = torch.log(scores[lbl])
    lnu = torch.log(1 - scores[nlbl])
    labeled_score = torch.mean(lnl) 
    unlabeled_score = torch.mean(lnu)
    bce_adj_loss = -labeled_score - l_adj*unlabeled_score
    return bce_adj_loss


def aff_to_adj(x, y=None):
    x = x.detach().cpu().numpy()
    adj = np.matmul(x, x.transpose())
    adj +=  -1.0*np.eye(adj.shape[0])
    adj_diag = np.sum(adj, axis=0) #rowise sum
    adj = np.matmul(adj, np.diag(1/adj_diag))
    adj = adj + np.eye(adj.shape[0])
    adj = torch.Tensor(adj).cuda()

    return adj


def read_data(dataloader, labels=True):
    if labels:
        while True:
            for img, label,_ in dataloader:
                yield img, label
    else:
        while True:
            for img, _, _ in dataloader:
                yield img


def vae_loss(x, recon, mu, logvar, beta):
    mse_loss = nn.MSELoss()
    MSE = mse_loss(recon, x)
    KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)
    # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # beta = 1
    # print('mse : ',MSE)
    KLD = KLD * beta
    return MSE + KLD


def flatten_vae_loss(x, recon, mu, logvar, beta):
    mse_loss = nn.MSELoss(reduction='none')
    MSE = mse_loss(recon, x)
    div = MSE.shape[1] * MSE.shape[2] * MSE.shape[3]
    MSE = torch.sum(MSE, dim=(1,2,3)) / div
    # KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)
    KLD = -0.5 * torch.sum((1 + logvar - mu.pow(2) - logvar.exp()),dim=1)
    # beta = 1
    KLD = KLD * beta
    return MSE + KLD


def train_vaal(models, optimizers, labeled_dataloader, unlabeled_dataloader, cycle):
    
    vae = models['vae']
    discriminator = models['discriminator']
    vae.train()
    discriminator.train()
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        vae = vae.cuda()
        discriminator = discriminator.cuda()
    
    adversary_param = 1
    beta          = 1
    num_adv_steps = 1
    num_vae_steps = 2

    bce_loss = nn.BCELoss()

    labeled_data = read_data(labeled_dataloader)
    unlabeled_data = read_data(unlabeled_dataloader)

    train_iterations = int( (ADDENDUM*cycle+ SUBSET) * EPOCHV / BATCH )
    for iter_count in range(train_iterations):
        labeled_imgs, labels = next(labeled_data)
        unlabeled_imgs = next(unlabeled_data)[0]

        with torch.cuda.device(CUDA_VISIBLE_DEVICES):
            labeled_imgs = labeled_imgs.cuda()
            unlabeled_imgs = unlabeled_imgs.cuda()

        # VAE step
        for count in range(num_vae_steps): # num_vae_steps
            recon, _, mu, logvar = vae(labeled_imgs)
            unsup_loss = vae_loss(labeled_imgs, recon, mu, logvar, beta)
            unlab_recon, _, unlab_mu, unlab_logvar = vae(unlabeled_imgs)
            transductive_loss = vae_loss(unlabeled_imgs, 
                    unlab_recon, unlab_mu, unlab_logvar, beta)
        
            labeled_preds = discriminator(mu)
            unlabeled_preds = discriminator(unlab_mu)
            
            lab_real_preds = torch.ones(labeled_imgs.size(0))
            unlab_real_preds = torch.ones(unlabeled_imgs.size(0))
                
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                lab_real_preds = lab_real_preds.cuda()
                unlab_real_preds = unlab_real_preds.cuda()

            dsc_loss = bce_loss(labeled_preds[:,0], lab_real_preds) + \
                       bce_loss(unlabeled_preds[:,0], unlab_real_preds)
            total_vae_loss = unsup_loss + transductive_loss + adversary_param * dsc_loss
            
            optimizers['vae'].zero_grad()
            total_vae_loss.backward()
            optimizers['vae'].step()

            # sample new batch if needed to train the adversarial network
            if count < (num_vae_steps - 1):
                labeled_imgs, _ = next(labeled_data)
                unlabeled_imgs = next(unlabeled_data)[0]

                with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                    labeled_imgs = labeled_imgs.cuda()
                    unlabeled_imgs = unlabeled_imgs.cuda()

        # Discriminator step
        for count in range(num_adv_steps):
            with torch.no_grad():
                _, _, mu, _ = vae(labeled_imgs)
                _, _, unlab_mu, _ = vae(unlabeled_imgs)
            
            labeled_preds = discriminator(mu)
            unlabeled_preds = discriminator(unlab_mu)
            
            lab_real_preds = torch.ones(labeled_imgs.size(0))
            unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0))

            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                lab_real_preds = lab_real_preds.cuda()
                unlab_fake_preds = unlab_fake_preds.cuda()
            
            dsc_loss = bce_loss(labeled_preds[:,0], lab_real_preds) + \
                       bce_loss(unlabeled_preds[:,0], unlab_fake_preds)

            optimizers['discriminator'].zero_grad()
            dsc_loss.backward()
            optimizers['discriminator'].step()
            # sample new batch if needed to train the adversarial network
            if count < (num_adv_steps-1):
                labeled_imgs, _ = next(labeled_data)
                unlabeled_imgs = next(unlabeled_data)[0]

                with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                    labeled_imgs= labeled_imgs.cuda()
                    unlabeled_imgs = unlabeled_imgs.cuda()
            if iter_count % 100 == 0:
                print("Iteration: " + str(iter_count) + "  vae_loss: " + str(total_vae_loss.item()) + " dsc_loss: " +
                      str(dsc_loss.item()))


def train_cvaal(models, optimizers, labeled_dataloader, unlabeled_dataloader, cycle, cls):
    vae = models['vae']
    # discriminator = models['discriminator']
    vae.train()
    # discriminator.train()
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        vae = vae.cuda()
        # discriminator = discriminator.cuda()

    adversary_param = 1
    beta = 1
    num_adv_steps = 1
    num_vae_steps = 2

    bce_loss = nn.BCELoss()

    # labeled_data = read_data(labeled_dataloader)
    unlabeled_data = read_data(unlabeled_dataloader)

    train_iterations = int((ADDENDUM * cycle) * EPOCHV / BATCH)
    epoch = int((ADDENDUM * cycle) * EPOCHV / BATCH * 0.4)
    epoch = 1         #changingggg
    print('epoch : ',epoch)
    # for iter_count in range(train_iterations ):
    for e in range(epoch):
        cnt = 0
        for i, (labeled_imgs, labels) in enumerate(labeled_dataloader):
            # labeled_imgs, labels = next(labeled_data)
            # unlabeled_imgs = next(unlabeled_data)[0]

            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                labeled_imgs = labeled_imgs.cuda()
                # unlabeled_imgs = unlabeled_imgs.cuda()
                labels = labels.cuda()

            # VAE step
            # for count in range(num_vae_steps):  # num_vae_steps
            recon, _, mu, logvar = vae(labeled_imgs)
            # nbeta = beta * (epoch-e)/epoch
            unsup_loss = vae_loss(labeled_imgs, recon, mu, logvar, beta)
            # print('unsup_loss : ',unsup_loss)

            total_vae_loss = unsup_loss#  + adversary_param * dsc_loss #transductive_loss +

            optimizers['vae'].zero_grad()
            total_vae_loss.backward()
            optimizers['vae'].step()

                # sample new batch if needed to train the adversarial network
                # if count < (num_vae_steps - 1):
                #     labeled_imgs, _ = next(labeled_data)
                #
                #     with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                #         labeled_imgs = labeled_imgs.cuda()
                #         labels = labels.cuda()
            # if e == epoch:
            # PATH + str(cycle)

            # warnings.filterwarnings("ignore")
            #IMG SAVE
            #if e % 299 == 0 and cnt <= 3:#e > epoch - 100 and e%10 == 0:
            # if e == epoch-1 and cnt<=3:
            #     cnt += 1
            #     if not os.path.exists(PATH + str(cycle) + '/org_class_' + str(cls) + '/' + str(e)):
            #         os.makedirs(PATH + str(cycle) + '/org_class_' + str(cls) + '/' + str(e))
            #     for j in range(len(labeled_imgs)):
            #         title = str(i) + '_' +str(j) + '.png'
            #         plt.title(title)
            #         plt.subplot(1,2,1)
            #         plt.imshow(labeled_imgs[j].cpu().permute(1, 2, 0).numpy().squeeze(), cmap='gray')
            #         plt.subplot(1,2,2)
            #         plt.imshow(recon[j].detach().cpu().permute(1, 2, 0).numpy().squeeze(), cmap='gray')
            #         plt.savefig(PATH + str(cycle) + '/org_class_' + str(cls) + '/' + str(e) + '/' + title)
            #         plt.clf()

            if e % 100 == 0:
                print(
                   "epoch : " + str(e) + "  Iteration: " + str(i) + "  vae_loss: " + str(total_vae_loss.item()))  # + " dsc_loss: " + str(
                # dsc_loss.item()))


def get_uncertainty(models, unlabeled_loader):
    models['backbone'].eval()
    models['module'].eval()
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        uncertainty = torch.tensor([]).cuda()

    with torch.no_grad():
        for inputs, _, _ in unlabeled_loader:
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                inputs = inputs.cuda()
            _, _, features = models['backbone'](inputs)
            pred_loss = models['module'](features) # pred_loss = criterion(scores, labels) # ground truth loss
            pred_loss = pred_loss.view(pred_loss.size(0))
            uncertainty = torch.cat((uncertainty, pred_loss), 0)
    
    return uncertainty.cpu()

def get_features(models, unlabeled_loader):
    models['backbone'].eval()
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        features = torch.tensor([]).cuda()    
    with torch.no_grad():
            for inputs, _, _ in unlabeled_loader:
                with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                    inputs = inputs.cuda()
                    _, features_batch, _ = models['backbone'](inputs)
                features = torch.cat((features, features_batch), 0)
            feat = features #.detach().cpu().numpy()
    return feat

def get_kcg(models, labeled_data_size, unlabeled_loader):
    models['backbone'].eval()
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        features = torch.tensor([]).cuda()

    with torch.no_grad():
        for inputs, _, _ in unlabeled_loader:
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                inputs = inputs.cuda()
            _, features_batch, _ = models['backbone'](inputs)
            features = torch.cat((features, features_batch), 0)
        feat = features.detach().cpu().numpy()
        new_av_idx = np.arange(SUBSET,(SUBSET + labeled_data_size))
        sampling = kCenterGreedy(feat)
        batch = sampling.select_batch_(new_av_idx, ADDENDUM)
        other_idx = [x for x in range(SUBSET) if x not in batch]
    return  other_idx + batch


def get_camresult(model, data_loader, num_cls):
    cam = GradCAM(model=model, target_layers = [model.layer4[-1]], use_cuda=True)
    all_graycam = None
    all_targets = None
    gray_transform = transforms.Compose([transforms.Grayscale()])
    for i, (input, target, _) in enumerate(data_loader):  # calculate grayscale cam
        iter_graycam = None
        for cls in range(num_cls):

            grayscale_cam = cam(input_tensor=input, target_category=cls)
            grayscale_cam = np.expand_dims(grayscale_cam, axis=1)

            gray_img = gray_transform(input)
            img_cam = gray_img.numpy() * grayscale_cam

            if iter_graycam is None:
                # iter_graycam = grayscale_cam
                iter_graycam = img_cam

            else:
                # iter_graycam = np.concatenate((iter_graycam, grayscale_cam), axis=1)
                iter_graycam = np.concatenate((iter_graycam, img_cam), axis=1)

        if all_graycam is None:
            all_graycam = iter_graycam
            all_targets = target

        else:
            all_graycam = np.concatenate((all_graycam, iter_graycam), axis=0)
            all_targets = torch.cat((all_targets, target))

    return all_graycam, all_targets


# Select the indices of the unlablled data according to the methods
def query_samples(model, method, data_unlabeled, subset, labeled_set, cycle, args):

    if method == 'Random':
        #arg = np.random.randint(SUBSET, size=SUBSET)
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH,
                                      sampler=SubsetSequentialSampler(subset), pin_memory=True)
        if args.dataset == 'cifar100':
            num_cls = 100
        else:
            num_cls = 10

        data_list = []
        for i in range(num_cls):
            data_list.append(np.array([]))

        for idx, (_, targets, _) in enumerate(unlabeled_loader):
            for i in range(len(targets)):
                data_list[targets[int(i)]] = np.append(data_list[targets[int(i)]], idx * BATCH + i)

        arg = []

        for i in range(num_cls):
            random.shuffle(data_list[i])
            arg += list(data_list[i][:int(ADDENDUM / num_cls)])

        arg = np.array(arg)

    if (method == 'UncertainGCN') or (method == 'CoreGCN'):
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH, 
                sampler=SubsetSequentialSampler(subset+labeled_set), # more convenient if we maintain the order of subset
                                    pin_memory=True)
        binary_labels = torch.cat((torch.zeros([SUBSET, 1]),(torch.ones([len(labeled_set),1]))),0)


        features = get_features(model, unlabeled_loader)
        features = nn.functional.normalize(features)
        adj = aff_to_adj(features)

        gcn_module = GCN(nfeat=features.shape[1],
                         nhid=args.hidden_units,
                         nclass=1,
                         dropout=args.dropout_rate).cuda()
                                
        models      = {'gcn_module': gcn_module}

        optim_backbone = optim.Adam(models['gcn_module'].parameters(), lr=LR_GCN, weight_decay=WDECAY)
        optimizers = {'gcn_module': optim_backbone}

        lbl = np.arange(SUBSET, SUBSET+(cycle+1)*ADDENDUM, 1)
        nlbl = np.arange(0, SUBSET, 1)
        
        ############
        for _ in range(200):

            optimizers['gcn_module'].zero_grad()
            outputs, _, _ = models['gcn_module'](features, adj)
            lamda = args.lambda_loss 
            loss = BCEAdjLoss(outputs, lbl, nlbl, lamda)
            loss.backward()
            optimizers['gcn_module'].step()


        models['gcn_module'].eval()
        with torch.no_grad():
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                inputs = features.cuda()
                labels = binary_labels.cuda()
            scores, _, feat = models['gcn_module'](inputs, adj)
            
            if method == "CoreGCN":
                feat = feat.detach().cpu().numpy()
                new_av_idx = np.arange(SUBSET,(SUBSET + (cycle+1)*ADDENDUM))
                sampling2 = kCenterGreedy(feat)  
                batch2 = sampling2.select_batch_(new_av_idx, ADDENDUM)
                other_idx = [x for x in range(SUBSET) if x not in batch2]
                arg = other_idx + batch2

            else:

                s_margin = args.s_margin 
                scores_median = np.squeeze(torch.abs(scores[:SUBSET] - s_margin).detach().cpu().numpy())
                arg = np.argsort(-(scores_median))

            print("Max confidence value: ",torch.max(scores.data))
            print("Mean confidence value: ",torch.mean(scores.data))
            preds = torch.round(scores)
            correct_labeled = (preds[SUBSET:,0] == labels[SUBSET:,0]).sum().item() / ((cycle+1)*ADDENDUM)
            correct_unlabeled = (preds[:SUBSET,0] == labels[:SUBSET,0]).sum().item() / SUBSET
            correct = (preds[:,0] == labels[:,0]).sum().item() / (SUBSET + (cycle+1)*ADDENDUM)
            print("Labeled classified: ", correct_labeled)
            print("Unlabeled classified: ", correct_unlabeled)
            print("Total classified: ", correct)
    
    if method == 'CoreSet':
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH, 
                                    sampler=SubsetSequentialSampler(subset+labeled_set), # more convenient if we maintain the order of subset
                                    pin_memory=True)

        arg = get_kcg(model, ADDENDUM*(cycle+1), unlabeled_loader)

    if method == 'lloss':
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH, 
                                    sampler=SubsetSequentialSampler(subset), 
                                    pin_memory=True)

        # Measure uncertainty of each data points in the subset
        uncertainty = get_uncertainty(model, unlabeled_loader)
        arg = np.argsort(uncertainty)

    if method == 'VAAL':
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH, 
                                    sampler=SubsetSequentialSampler(subset), 
                                    pin_memory=True)
        labeled_loader = DataLoader(data_unlabeled, batch_size=BATCH, 
                                    sampler=SubsetSequentialSampler(labeled_set), 
                                    pin_memory=True)

        if args.dataset == 'fashionmnist':
            vae = VAE(28,1,3)
            discriminator = Discriminator(28)
        else:
            vae = VAE()
            discriminator = Discriminator(32)
        models = {'vae': vae, 'discriminator': discriminator}

        optim_vae = optim.Adam(vae.parameters(), lr=5e-4)
        optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)
        optimizers = {'vae': optim_vae, 'discriminator':optim_discriminator}

        train_vaal(models,optimizers, labeled_loader, unlabeled_loader, cycle+1)
        
        all_preds, all_indices = [], []

        for images, _, indices in unlabeled_loader:                       
            images = images.cuda()
            with torch.no_grad():
                _, _, mu, _ = vae(images)
                preds = discriminator(mu)

            preds = preds.cpu().data
            all_preds.extend(preds)
            all_indices.extend(indices)

        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        # need to multiply by -1 to be able to use torch.topk 
        all_preds *= -1
        # select the points which the discriminator things are the most likely to be unlabeled
        _, arg = torch.sort(all_preds)

    if method == 'CAAL':
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH,
                                      sampler=SubsetSequentialSampler(subset),
                                      pin_memory=True)
        labeled_loader = DataLoader(data_unlabeled, batch_size=BATCH,
                                    sampler=SubsetSequentialSampler(labeled_set),
                                    pin_memory=True)

        if args.dataset == 'CIFAR100':
            num_cls = 100
        else:
            num_cls = 10

        labeled_graycam, labeled_targets = get_camresult(model['backbone'], labeled_loader, num_cls)

        unlabeled_graycam, unlabeled_targets = get_camresult(model['backbone'], unlabeled_loader, num_cls)

        if args.dataset == 'fashionmnist':
            vae = VAE(28, num_cls, 3)
            discriminator = Discriminator(28)
        elif args.dataset == 'cifar100':
            vae = VAE(nc=num_cls)
            discriminator = Discriminator(32)
        else:
            vae = VAE(nc=num_cls)
            discriminator = Discriminator(32)
        models = {'vae': vae, 'discriminator': discriminator}
        optim_vae = optim.Adam(vae.parameters(), lr=5e-4)
        optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)
        optimizers = {'vae': optim_vae, 'discriminator': optim_discriminator}

        cam_labeled_data = CamDataset(labeled_graycam, labeled_targets)
        cam_unlabeled_data = CamDataset(unlabeled_graycam, unlabeled_targets)

        cam_labeled_loader =  DataLoader(cam_labeled_data, batch_size=BATCH)
        cam_unlabeled_loader = DataLoader(cam_unlabeled_data, batch_size=BATCH)

        train_vaal(models, optimizers, cam_labeled_loader, cam_unlabeled_loader, cycle + 1)

        all_preds, all_indices = [], []
        labeled_true = 0
        unlabeled_true = 0

        for images, _, indices in cam_labeled_loader:
            images = images.cuda()
            with torch.no_grad():
                _, _, mu, _ = vae(images)
                preds = discriminator(mu)

            labeled_true += len(preds[preds>=0.5])

        for images, _, indices in cam_unlabeled_loader:
            images = images.cuda()
            with torch.no_grad():
                _, _, mu, _ = vae(images)
                preds = discriminator(mu)

            unlabeled_true += len(preds[preds<0.5])
            preds = preds.cpu().data
            all_preds.extend(preds)
            all_indices.extend(indices)

        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        # need to multiply by -1 to be able to use torch.topk
        all_preds *= -1
        # select the points which the discriminator things are the most likely to be unlabeled

        with open('newcam_discrim_results.txt', 'a') as f:
            f.write('{} | labeled_results : {}/{} | unlabeled_results : {}/{}'.
                    format(cycle+1, labeled_true, len(labeled_graycam), unlabeled_true ,len(unlabeled_graycam)))
            f.write('\n')


        _, arg = torch.sort(all_preds)

        return arg, labeled_graycam, unlabeled_graycam

    if method == 'daeho':
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH,
                                      sampler=SubsetSequentialSampler(subset),
                                      pin_memory=True)
        labeled_loader = DataLoader(data_unlabeled, batch_size=BATCH,
                                    sampler=SubsetSequentialSampler(labeled_set),
                                    pin_memory=True)
        if args.dataset == 'fashionmnist':
            vae = CVAE(28, 1, 3, 10)
            discriminator = Discriminator(28)
        elif args.dataset == 'cifar100':
            vae = CVAE(32, 3, 4, 100)
            discriminator = Discriminator(32)
        else:
            vae = CVAE()
            discriminator = Discriminator(32)
        models = {'vae': vae, 'discriminator': discriminator}

        optim_vae = optim.Adam(vae.parameters(), lr=5e-4)
        for name, param in vae.named_parameters():
            if param.requires_grad:
                print(name, param.data)

        optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)
        optimizers = {'vae': optim_vae, 'discriminator': optim_discriminator}

        train_cvaal(models, optimizers, labeled_loader, unlabeled_loader, cycle + 1)

        all_preds, all_indices = [], []

        for images, _, indices in unlabeled_loader:
            images = images.cuda()
            with torch.no_grad():
                _, _, mu, _, _ = vae(images)
                preds = discriminator(mu)

            preds = preds.cpu().data
            all_preds.extend(preds)
            all_indices.extend(indices)

        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        # need to multiply by -1 to be able to use torch.topk
        all_preds *= -1
        # select the points which the discriminator things are the most likely to be unlabeled
        _, arg = torch.sort(all_preds)
        return arg, vae

    if method == 'cam':
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=32,
                                    sampler=SubsetSequentialSampler(subset),
                                    pin_memory=True)
        if args.dataset == 'CIFAR100':
            num_cls = 100
        else:
            num_cls = 10
        cam = GradCAM(model=model['backbone'], target_layers = [model['backbone'].layer4[-1]], use_cuda=True)
        all_graycam = None

        for i, (input, target, indeces) in enumerate(unlabeled_loader): #calculate grayscale cam
            iter_graycam = None
            for cls in range(num_cls):

                grayscale_cam = cam(input_tensor= input, target_category = cls)
                grayscale_cam = np.expand_dims(grayscale_cam, axis=1)
                # print('grayscale : ', grayscale_cam)
                if iter_graycam is None:
                    iter_graycam = grayscale_cam

                else:
                    iter_graycam = np.concatenate((iter_graycam, grayscale_cam), axis=1)

            if all_graycam is None:
                all_graycam = iter_graycam

            else:
                all_graycam = np.concatenate((all_graycam, iter_graycam), axis=0)

        graycam_std = np.array([])

        for graycam in all_graycam:
            graycam_std = np.append(graycam_std, np.std(graycam))

        print('all result size : ',all_graycam.shape)
        tensor_graystd = torch.from_numpy(graycam_std)
        _, arg = torch.sort(-1 * tensor_graystd)
        print('all result size : ', graycam_std.shape)

    if method == 'ensemble':
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH,
                                      sampler=SubsetSequentialSampler(subset),
                                      pin_memory=True)
        labeled_loader = DataLoader(data_unlabeled, batch_size=BATCH,
                                    sampler=SubsetSequentialSampler(labeled_set),
                                    pin_memory=True)

        if args.dataset == 'cifar100':
            num_cls = 100
        else:
            num_cls = 10

        imgs_list = []
        unlab_imgs_list = []

        for i in range(num_cls):
            imgs_list.append(np.array([]))
            unlab_imgs_list.append(np.array([]))

        c = Counter()
        for i, (img, label, _) in enumerate(labeled_loader):
            for j in range(len(label)):
                if len(imgs_list[label[j]]) == 0:
                    imgs_list[label[j]] = img[j].unsqueeze(0).numpy()
                else:
                    imgs_list[label[j]] = np.concatenate((imgs_list[label[j]], img[j].unsqueeze(0).numpy()))
            c.update(label.numpy())

        with open('labels.txt', 'a') as f:
            f.write(str(cycle+1) + ':' + str(c) + '\n')

        for i, (img, label, _) in enumerate(unlabeled_loader):
            for j in range(len(label)):
                if len(unlab_imgs_list[label[j]]) == 0:
                    unlab_imgs_list[label[j]] = img[j].unsqueeze(0).numpy()
                else:
                    unlab_imgs_list[label[j]] = np.concatenate((unlab_imgs_list[label[j]], img[j].unsqueeze(0).numpy()))

        data_list = []
        unlab_data_list = []
        for i in range(num_cls):    #split datum
            data_list.append(NewDataset(imgs_list[i], np.ones(len(imgs_list[i])) * i))
            unlab_data_list.append(NewDataset(unlab_imgs_list[i], np.ones(len(unlab_imgs_list[i])) * i))

        if args.dataset == 'fashionmnist':
            # vae = VAE(28, 1, 3)

            for i in range(10):
                globals()['vae{}'.format(i)] = VAE(28,1,3)
            vae_list = [vae0, vae1, vae2, vae3, vae4, vae5, vae6, vae7, vae8, vae9]

            discriminator = Discriminator(28)
            for i in range(10):
                globals()['optim_vae{}'.format(i)] = optim.Adam(vae_list[i].parameters(), lr=5e-4)
            optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)
            optim_list = [optim_vae0, optim_vae1, optim_vae2, optim_vae3, optim_vae4, optim_vae5, optim_vae6, optim_vae7, optim_vae8, optim_vae9]

        elif args.dataset == 'cifar100':
            discriminator = Discriminator(32)
            optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)
            optim_list = []

        else:
            for i in range(10):
                globals()['vae{}'.format(i)] = VAE()
            vae_list = [vae0, vae1, vae2, vae3, vae4, vae5, vae6, vae7, vae8, vae9]
            discriminator = Discriminator(32)

            for i in range(10):
                globals()['optim_vae{}'.format(i)] = optim.Adam(vae_list[i].parameters(), lr=5e-4)
            optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)
            optim_list = [optim_vae0, optim_vae1, optim_vae2, optim_vae3, optim_vae4, optim_vae5, optim_vae6, optim_vae7, optim_vae8, optim_vae9]



        loss_thres = np.zeros(num_cls)
        loss_var_list = np.zeros(num_cls)
        loss_same_max = np.zeros(num_cls)

        loss_diff_thres = np.zeros(num_cls)
        loss_diff_var_list = np.zeros(num_cls)
        loss_diff_min = np.zeros(num_cls)

        for i in range(num_cls):
            total_loss = []
            new_dataloader = DataLoader(data_list[i], batch_size=BATCH)
            unl_new_dataloader = DataLoader(unlab_data_list[i], batch_size=BATCH)

            if args.dataset == 'cifar100':
                vae = VAE()
                models = {'vae': vae, 'discriminator':discriminator}
                optimizers = {'vae': optim.Adam(vae.parameters(), lr=5e-4), 'discriminator': optim_discriminator}
                train_cvaal(models, optimizers, new_dataloader, unlabeled_loader, cycle + 1 , i)
                torch.save(models['vae'].state_dict(), 'models/{}'.format(args.number) + args.dataset + '_vae_{}.pth'.format(i))

            else:
                models = {'vae': vae_list[i], 'discriminator': discriminator}
                optimizers = {'vae': optim_list[i], 'discriminator': optim_discriminator}

                train_cvaal(models, optimizers, new_dataloader, unlabeled_loader, cycle + 1 , i)

            loss_diff = 0
            total_diff = None
            # print('why?')
            for j in range(num_cls):
                if i == j:
                    # print('i : ',i)
                    # print('j : ',j)
                    loss_same = 0
                    total_num = 0
                    total_loss = None
                    for k, (labeled_imgs, labels) in enumerate(new_dataloader):
                        models['vae'].cuda()
                        labeled_imgs = labeled_imgs.cuda()
                        # unlabeled_imgs = unlabeled_imgs.cuda()
                        labels = labels.cuda()
                        with torch.no_grad():
                            models['vae'](labeled_imgs)

                            # VAE step
                            # for count in range(num_vae_steps):  # num_vae_steps
                            recon, _, mu, logvar = models['vae'](labeled_imgs)
                            unsup_loss = flatten_vae_loss(labeled_imgs, recon, mu, logvar, beta=1)
                            # loss_same += (unsup_loss * len(labels))
                            # total_num += len(labels)
                            if total_loss is None:
                                total_loss = unsup_loss.detach().cpu().numpy()
                            else:
                                total_loss = np.concatenate((total_loss, unsup_loss.detach().cpu().numpy()))
                    # print('total_num : ',total_num)
                    # loss_same /= total_num
                    loss_same = np.mean(total_loss)
                    loss_same_var = np.std(total_loss)
                    loss_same_mx = np.max(total_loss)
                    loss_thres[i] = loss_same
                    loss_var_list[i] = loss_same_var
                    loss_same_max[i] = loss_same_mx
                    # print('loss_same : ',loss_same)

                else:
                    bnew_dataloader = DataLoader(data_list[j], batch_size=BATCH)
                    for k, (labeled_imgs, labels) in enumerate(bnew_dataloader):
                        models['vae'].cuda()
                        labeled_imgs = labeled_imgs.cuda()
                        # unlabeled_imgs = unlabeled_imgs.cuda()
                        labels = labels.cuda()
                        with torch.no_grad():
                            models['vae'](labeled_imgs)

                            # VAE step
                            # for count in range(num_vae_steps):  # num_vae_steps
                            recon, _, mu, logvar = models['vae'](labeled_imgs)

                            unsup_loss = flatten_vae_loss(labeled_imgs, recon, mu, logvar, beta=1)

                            if total_diff is None:
                                total_diff = unsup_loss.detach().cpu().numpy()
                            else:
                                total_diff = np.concatenate((total_diff, unsup_loss.detach().cpu().numpy()))
                            # print('unsup_loss : ',unsup_loss)
                            # print('len labels : ',len(labels))

                            # loss_diff += unsup_loss * len(labels)
                            # total_diff += len(labels)

            # loss_diff /= total_diff
            loss_diff = np.mean(total_diff)
            loss_diff_var = np.std(total_diff)
            loss_diff_mn = np.min(total_diff)

            loss_diff_thres[i] = loss_diff
            loss_diff_var_list[i] = loss_diff_var
            loss_diff_min[i] = loss_diff_mn

            print('loss_same : ',loss_same)
            print('loss_diff : ',loss_diff)

        with open(args.dataset + '_' + args.method_type + '_' + str(args.number) + '_' + 'thres.txt', 'a') as f:
            f.write(str(cycle+1) + 'thres same : ' + str(loss_thres) + '\n')
            f.write(str(cycle + 1) + 'thres same var : ' + str(loss_var_list) + '\n')
            f.write(str(cycle + 1) + 'thres same max : ' + str(loss_same_max) + '\n')

            f.write(str(cycle + 1) + 'thres diff : ' + str(loss_diff_thres) + '\n')
            f.write(str(cycle + 1) + 'thres diff var : ' + str(loss_diff_var_list) + '\n')
            f.write(str(cycle + 1) + 'thres diff min : ' + str(loss_diff_min) + '\n')
        beta = 1

        # c = Counter()
        np_total_loss = None
        arg = np.array([])
        print('loss_thres : ',loss_thres)
        unlab_label_list = None
        for i, (images, labels, _) in enumerate(unlabeled_loader):
            total_loss = []
            images = images.cuda()

            if unlab_label_list is None:
                unlab_label_list = labels.detach().cpu().numpy()
            else:
                unlab_label_list = np.concatenate((unlab_label_list, labels.detach().cpu().numpy()))
            print('labels : ',labels)

            for vae_num in range(num_cls):
                if args.dataset == 'cifar100':
                    vae = VAE()
                    vae.load_state_dict(torch.load('models/{}'.format(args.number) + args.dataset + '_vae_{}.pth'.format(vae_num)))
                else:
                    vae = vae_list[vae_num]
                vae = vae.cuda()
                vae.eval()

                with torch.no_grad():
                    recon, _, mu, logvar = vae(images)
                    unsup_loss = flatten_vae_loss(images, recon, mu, logvar, beta)
                    # print('fffffffff unsup loss : ',unsup_loss)

                    bool_unsup_loss = unsup_loss.detach().cpu().numpy() < loss_thres[vae_num]  #(128)
                    total_loss.append(bool_unsup_loss)  #(num_cls, n)

            if np_total_loss is None:
                np_total_loss = np.array(total_loss)    #(num_cls, n)
            else:
                np_total_loss = np.hstack((np_total_loss, np.array(total_loss)))

        # print('np total loss : ',np_total_loss.shape)
        # print('shitttt : ',np_total_loss)
        prediction_check_list = np.zeros(num_cls)
        # print('np total loss : ',np_total_loss.shape)
        for idx in range(len(unlab_label_list)):    #unlab_label_list has true label
            if np_total_loss[unlab_label_list[idx]][idx]:
                prediction_check_list[unlab_label_list[idx]] += 1

        cnt = Counter()
        cnt.update(unlab_label_list)

        for i in range(len(prediction_check_list)):
            prediction_check_list[i] = prediction_check_list[i] / cnt[i]

        bools = sum(np_total_loss)
        # print('bools : ', sum(bools > 2))
        with open(args.dataset + '_' + args.method_type + '_' + str(args.number) + '_' + 'thres.txt', 'a') as f:
            f.write(str(cycle + 1) + 'was unlab right? : ' + str(prediction_check_list) + '\n')
            f.write(str(cycle + 1) + 'bools : ' + str(sum(bools>2)) + '\n')

        _, arg = torch.sort(torch.tensor(bools))

        #_, arg = torch.sort(-1*torch.tensor(total_arr))
        # for cls in range(10):
        #     unl_new_dataloader = DataLoader(unlab_data_list[cls], batch_size=BATCH)
        #     for vae_num in range(10):
        #         vae = vae_list[vae_num]
        #         vae = vae.cuda()
        #         for i, (images, labels) in enumerate(unl_new_dataloader):
        #             images = images.cuda()
        #             # c.update(labels)
        #             if i >= 3:
        #                 break
        #             with torch.no_grad():
        #                 recon, _, mu, logvar = vae(images)
        #                 unsup_loss = vae_loss(images, recon, mu, logvar, beta)
        #
        #                 # total_vae_loss = unsup_loss
        #
        #                 if len(unlab_recon_list[cls]) == 0:
        #                     unlab_recon_list[cls] = recon.detach().cpu().numpy()
        #                 else:
        #                     unlab_recon_list[cls] = np.concatenate((unlab_recon_list[cls], recon.detach().cpu().numpy()))

                        # if not os.path.exists(PATH + str(cycle+1) + '/class_' + str(cls) + '/newreconvae/'):
                        #     os.makedirs(PATH + str(cycle+1) + '/class_' + str(cls) + '/newreconvae/')
                        # for j in range(len(images)):
                        #     title = 'vae_' + str(vae_num) + 'cls_' + str(cls) + str(i) + '_' + str(j) + '.png'
                        #     plt.title(title)
                        #     plt.subplot(1, 2, 1)
                        #     plt.imshow(images[j].cpu().permute(1, 2, 0).numpy())
                        #     plt.subplot(1, 2, 2)
                        #     plt.imshow(recon[j].detach().cpu().permute(1, 2, 0).numpy())
                        #     plt.savefig(PATH + str(cycle+1) + '/class_' + str(cls) + '/newreconvae/' + title)
        # return 0


    return arg
