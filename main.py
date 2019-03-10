from dataloader import *
from model import *
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import random
import datetime
from tensorboardX import SummaryWriter
import torchvision
import torchvision.utils as vutils
from CX.CX.CX_helper import *
from CX.config import *

# General preferences
use_cuda = torch.cuda.is_available()
epochs = 3000
log_file = 'logs/log.log'
#


# configuration options
def set_cfg():
    cfg = {'architecture':None, 'optimizer':None, 'batch_size':None, 'scheduler_gamma':None, 'k':None, 'scale':None,
           'train_data':None, 'test_data':None, 'sy':None, 'my':None, 'n':None, 'sx':None, 'sg':None}
    cfg['architecture'] = 'SCN_Dy' # SCN / SCN_theta / SCN_Dy
    cfg['optimizer'] = random.choice(['Adam'])
    cfg['learning_rate'] = random.choice([1e-3]) #for SGD around 1e-7
    cfg['batch_size'] = random.choice([64])
    cfg['scheduler_gamma'] = random.choice([0.1])
    cfg['criterion'] = random.choice(['L1']) # L1 / MSE / CX
    cfg['k'] = random.choice([1])
    cfg['scale'] = random.choice(['2'])
    cfg['train_data'] = random.choice(['T91'])
    cfg['test_data'] = random.choice(['Set5'])
    cfg['sy'] = random.choice([9])
    cfg['my'] = random.choice([100])
    cfg['n'] = random.choice([128])
    cfg['sx'] = random.choice([5])
    cfg['sg'] = random.choice([5])
    return cfg


# Create train&test dataloaders
def set_data_sets(transform_train, transform_test, cfg):
    train_dir = 'data/SR_training_datasets/' + cfg['train_data']
    test_dir = 'data/SR_testing_datasets/' + cfg['test_data']
    scale = cfg['scale']
    batch_size = cfg['batch_size']
    train_dataset = SR_dataloader(Train=True, transform=transform_train, dir=train_dir, scale=scale, use_cuda=use_cuda)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print('Loaded %d of Training Images' % train_dataset.__len__())
    test_dataset = SR_dataloader(Train=False, transform=transform_test, dir=test_dir, scale=scale, use_cuda=use_cuda)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print('Loaded %d of Test Images' % test_dataset.__len__())
    return trainloader, testloader


# create the configured net
def init_net(cfg):
    if (cfg['architecture']=='SCN'):
        net = tensor_to_gpu(SCN(cfg), use_cuda)
    elif (cfg['architecture']=='SCN_theta'):
        net = tensor_to_gpu(SCN_theta(cfg), use_cuda)
    elif (cfg['architecture']=='SCN_Dy'):
        net = tensor_to_gpu(SCN_Dy(cfg), use_cuda)
    else:
        net = tensor_to_gpu(SCN(cfg), use_cuda)
    return net


# create the configured optimizer
def set_optimizer(cfg, net):
    if (cfg['optimizer']=='Adam'):
        optimizer = optim.Adam(net.parameters(), lr=cfg['learning_rate'])
    elif (cfg['optimizer']=='SGD'):
        optimizer = optim.SGD(net.parameters(), lr=cfg['learning_rate'])
    else:
        optimizer = optim.Adam(net.parameters(), lr=cfg['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=cfg['scheduler_gamma'])
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000, 2000], gamma=cfg['scheduler_gamma'])
    return optimizer, scheduler


def ycbcr_to_rgb(input):
    # input is mini-batch N x 3 x H x W of an YCbCr image
    output = Variable(input.data.new(*input.size()))
    # renormalize values
    input[:, 0, :, :] = (input[:, 0, :, :]*255.0-16)/(235-16)
    #output[:, 1:, :, :] = (input[:, 1:, :, :]*255.0-16)/(240-16)
    #                           Y (renormalized)                Cb                            Cr
    output[:, 0, :, :] = 1.164 * input[:, 0, :, :] + 0 * input[:, 1, :, :] + 1.59 * input[:, 2, :, :] - 222.921
    output[:, 1, :, :] = 1.164 * input[:, 0, :, :] - 0.391 * input[:, 1, :, :] - 0.813 * input[:, 2, :, :] + 135.576
    output[:, 2, :, :] = 1.164 * input[:, 0, :, :] + 2.017 * input[:, 1, :, :] + 0 * input[:, 2, :, :] - 276.836
    return output


def ycbcr_to_y(input):
    # input is mini-batch N x 3 x H x W of an YCbCr image
    output = Variable(input.data.new(*input.size()))
    # show Y only
    output[:, 0, :, :] = input[:, 0, :, :]
    output[:, 1, :, :] = input[:, 0, :, :]
    output[:, 2, :, :] = input[:, 0, :, :]
    return output


# visualize training - tensorboard
def visualize_training(tb_writer, loss_graph, iter, hr_img, bicub_img, output_img):
    tb_writer.add_scalar('Training Error', loss_graph["mse_train"][-1].item(), iter)
    tb_writer.add_scalar('Test Error', loss_graph["mse_test"][-1].item(), iter)
    tb_writer.add_scalar('PSNR', loss_graph["psnr"][-1].item(), iter)
    output_img = ycbcr_to_rgb(output_img)
    output_normalized = vutils.make_grid(output_img, normalize=True, scale_each=True, range=(0, 255))
    tb_writer.add_image('Output', output_normalized, iter)
    hr_img = ycbcr_to_rgb(hr_img)
    hr_normalized = vutils.make_grid(hr_img, normalize=True, scale_each=True, range=(0, 255))
    tb_writer.add_image('High Resolution', hr_normalized, iter)
    bicub_img = ycbcr_to_rgb(bicub_img)
    bicub_normalized = vutils.make_grid(bicub_img, normalize=True, scale_each=True, range=(0, 255))
    tb_writer.add_image('Bicubic', bicub_normalized, iter)
    output_img_y = ycbcr_to_y(output_img)
    output_normalized_y = vutils.make_grid(output_img_y, normalize=True, scale_each=True, range=(0, 255))
    tb_writer.add_image('Output_y', output_normalized_y, iter)
    hr_img_y = ycbcr_to_y(hr_img)
    hr_normalized_y = vutils.make_grid(hr_img_y, normalize=True, scale_each=True, range=(0, 255))
    tb_writer.add_image('High Resolution_y', hr_normalized_y, iter)
    bicub_img_y = ycbcr_to_y(bicub_img)
    bicub_normalized_y = vutils.make_grid(bicub_img_y, normalize=True, scale_each=True, range=(0, 255))
    tb_writer.add_image('Bicubic_y', bicub_normalized_y, iter)


def calc_psnr(y, gt):
    sum = 0.0
    for i in range(y.shape[0]):
        y_uint8 = np.rint(np.clip(y[i], 0, 255))
        gt_uint8 = np.rint(np.clip(gt[i], 0, 255))
        diff = y_uint8 - gt_uint8
        rmse = np.sqrt((diff**2).mean())
        psnr = 20*np.log10(255.0/rmse)
        sum += psnr
    return sum / y.shape[0]


def get_CX_loss(img1, img2):
    # CX loss
    vgg_netfull = tensor_to_gpu(torchvision.models.vgg19(pretrained=True), use_cuda)
    for layer, w in config.CX.feat_content_layers.items():
        if layer == 'conv4_2':
            layer = 22
        elif layer == 'conv3_2':
            layer = 14
        vgg_net = nn.Sequential(*list(vgg_netfull.features.children())[0:layer])
        vgg_output = vgg_net(img1.repeat(1, 3, 1, 1))
        vgg_gt = vgg_net(img2.repeat(1, 3, 1, 1))
        CX_loss_content_list = w * CX_loss_helper(vgg_output, vgg_gt, config.CX)
        CX_loss = np.sum(CX_loss_content_list)
        CX_loss = torch.tensor(CX_loss)
        CX_loss.requires_grad = True
    return CX_loss


# Perform training
def train_net(net, criterion, optimizer, scheduler, trainloader, testloader, tb_writer, loss_graph):
    err_vec = []
    err_test_vec = []
    epoch_idx = 0

    print('Start Training')
    for epoch in range(epochs):
        net.train()
        scheduler.step()
        running_loss = 0.0
        running_loss_test = 0.0


        for i, data in enumerate(trainloader, 0):
            # get inputs
            hr_img, bicub_img = data
            hr_img, bicub_img = tensor_to_gpu(Variable(hr_img),use_cuda), tensor_to_gpu(Variable(bicub_img),use_cuda)
            hr_img_y = hr_img[:, 0, :, :].unsqueeze(1)
            bicub_img_y = bicub_img[:, 0, :, :].unsqueeze(1)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            output_img_y = net(bicub_img_y)
            # backward + optimize
            loss = criterion(output_img_y[:,:,10:-10, 10:-10], hr_img_y[:,:,10:-10, 10:-10])
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.data

        # epoch statistics
        # training performance
        err_vec += [running_loss / trainloader.__len__()]
        running_loss = 0
        epoch_idx += 1
        print('[%d] loss train: %.3f' % (epoch_idx, err_vec[-1]))
        loss_graph["mse_train"].append(err_vec[-1])
        # test performance
        net.eval()
        for data in testloader:
            hr_img, bicub_img = data
            hr_img, bicub_img = tensor_to_gpu(Variable(hr_img),use_cuda), tensor_to_gpu(Variable(bicub_img),use_cuda)
            hr_img_y = hr_img[:, 0, :, :].unsqueeze(1)
            bicub_img_y = bicub_img[:, 0, :, :].unsqueeze(1)
            # forward
            output_img_y = net(bicub_img_y)
            loss = criterion(output_img_y[:,:,10:-10, 10:-10], hr_img_y[:,:,10:-10, 10:-10])
            # calc statistics
            running_loss_test += loss.data
        err_test_vec += [running_loss_test / testloader.__len__()]
        print('[%d] loss test: %.3f' % (epoch_idx, err_test_vec[-1]))
        loss_graph["mse_test"].append(err_test_vec[-1])
        bicub_psnr = calc_psnr(tensor_to_cpu(bicub_img_y[:, 0, 10:-10, 10:-10].data, use_cuda).numpy(), tensor_to_cpu(hr_img_y[:, 0, 10:-10,10:-10].data, use_cuda).numpy())
        SCN_psnr = calc_psnr(tensor_to_cpu(output_img_y[:,0,10:-10,10:-10].data, use_cuda).numpy(), tensor_to_cpu(hr_img_y[:,0,10:-10,10:-10].data, use_cuda).numpy())
        loss_graph["psnr"].append(SCN_psnr)
        print('bicubic psnr = %.3f, SCN psnr = %.3f' % (bicub_psnr, SCN_psnr))
        output_img = bicub_img.detach().clone()
        output_img[:,0,:,:] = output_img_y[:,0,:,:]
        visualize_training(tb_writer, loss_graph, epoch_idx, hr_img.detach(), bicub_img.detach(), output_img.detach())

    print('Finished Training')
    return net, err_vec, err_test_vec


def save_error_plot(cfg, err_vec, err_test_vec):
    plt.plot(err_vec, label='Training Data')
    plt.plot(err_test_vec, label='Test Data')
    plt.ylabel('MSE loss')
    plt.xlabel('Epoch')
    plt.title('Net Loss')
    plt.legend(loc='upper right')
    plt.suptitle('Arch: %s | Optimizer: %s %f | Batch S: %d' % (cfg['architecture'], cfg['optimizer'], cfg['learning_rate'], cfg['batch_size']))
    file_path = 'figs/%s_%s_%d' % (cfg['architecture'], cfg['optimizer'], cfg['batch_size'])
    plt.savefig(file_path)
    plt.clf()
    return (file_path + '.png')


def save_net(net, cfg, acc):
    str_acc = ('%.2f' % acc).replace('.', 'p')
    net_to_save = tensor_to_cpu(net,use_cuda)
    filename = 'net_files/' + '_' + cfg['architecture'] + '_' + cfg['optimizer'] + '_' + str(cfg['batch_size']) + '_' + str_acc + '.pkl'
    torch.save(net_to_save.state_dict(), filename)


def writelog(cfg, acc_train, acc_test, time_dif):
    str_acc_train = ('%.2f' % acc_train)
    str_acc_test = ('%.2f' % acc_test)
    line2write = (cfg['architecture'] + ', ' + cfg['optimizer'] + ', ' + str(cfg['batch_size']) + ', ' +
                   'acc training: ' + str_acc_train + ', acc test: ' + str_acc_test + ', time elapsed: ' +
                  str(time_dif))
    log = open(log_file,'a')
    log.write(line2write+'\n')
    log.close()


def set_criterion(cfg):
    if cfg['criterion'] == 'L1':
        criterion = tensor_to_gpu(nn.L1Loss(), use_cuda)
    elif cfg['criterion'] == 'MSE':
        criterion = tensor_to_gpu(nn.MSELoss(), use_cuda)
    elif cfg['criterion'] == 'CX':
        criterion = get_CX_loss
    else:
        criterion = tensor_to_gpu(nn.L1Loss(), use_cuda)
    return criterion


def main():
    tb_writer = SummaryWriter('logs/TB/')
    loss_graph = {
        "mse_train": [],
        "mse_test": [],
        "psnr": []
    }

    cfg = set_cfg()
    train_transforms_f = train_transforms
    test_transforms_f = test_transforms
    trainloader, testloader = set_data_sets(transform_train=train_transforms_f, transform_test=test_transforms_f, cfg=cfg)
    net = init_net(cfg)
    optimizer, scheduler = set_optimizer(cfg, net)
    criterion = set_criterion(cfg)
    start_time = datetime.datetime.now()
    net, err_vec, err_test_vec = train_net(net, criterion, optimizer, scheduler, trainloader, testloader, tb_writer, loss_graph)
    end_time = datetime.datetime.now()
    fig_path = save_error_plot(cfg, err_vec, err_test_vec)
    save_net(net, cfg, err_test_vec[-1])
    writelog(cfg, err_vec[-1], err_test_vec[-1], end_time-start_time)
    tb_writer.close()


if __name__ == "__main__":
    main()

