from __future__ import print_function
import _init_paths
import datetime
import argparse
import torch
import torch.optim as optim
from torch.autograd import Variable
import pytz
import scipy.misc
import os.path as osp
import os
from PIL import Image
from os.path import isfile, join
from os import listdir
import numpy as np
import random
import pdb
import copy
from shutil import copyfile

from lib.data_loader.color_mnist_tree_multi import COLORMNISTTREE
from lib.data_loader.clevr.clevr_tree import CLEVRTREE
from lib.config import load_config, Struct
from models.PNPNet.pnp_net import PNPNet
from models.PNPNet.simplified_pnp_net import PNPNet as SimplePNPNet
from trainers.pnpnet_trainer import PNPNetTrainer
from lib.weight_init import weights_init

parser = argparse.ArgumentParser(description='PNPNet - main model experiment')
parser.add_argument('--config_path', type=str, default='./configs/pnp_net_configs.yaml', metavar='C',
                    help='path to the configuration file')


def main():
    args = parser.parse_args()

    config_dic = load_config(args.config_path)
    configs = Struct(**config_dic)

    assert (torch.cuda.is_available())  # assume CUDA is always available

    print('configurations:', configs)

    torch.cuda.set_device(configs.gpu_id)
    torch.manual_seed(configs.seed)
    torch.cuda.manual_seed(configs.seed)
    np.random.seed(configs.seed)
    random.seed(configs.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    configs.exp_dir = configs.project_dir+'/results/' + configs.data_folder + '/' + configs.exp_dir_name
    exp_dir = configs.exp_dir

    try:
        os.makedirs(configs.exp_dir)
    except:
        pass
    try:
        os.makedirs(osp.join(configs.exp_dir, 'samples'))
    except:
        pass
    try:
        os.makedirs(osp.join(configs.exp_dir, 'checkpoints'))
    except:
        pass

    # loaders
    if 'CLEVR' in configs.data_folder:
        # we need the module's label->index dictionary from train loader
        train_loader = CLEVRTREE(phase='train', base_dir=osp.join(configs.base_dir, configs.data_folder),
                                 batch_size=configs.batch_size,
                                 random_seed=configs.seed, shuffle=True)
        test_loader = CLEVRTREE(phase='test', base_dir=osp.join(configs.base_dir, configs.data_folder),
                                batch_size=configs.batch_size,
                                random_seed=configs.seed, shuffle=False)
        gen_loader = CLEVRTREE(phase='test', base_dir=osp.join(configs.base_dir, configs.data_folder),
                               batch_size=configs.batch_size,
                               random_seed=configs.seed, shuffle=False)
    elif 'COLORMNIST' in configs.data_folder:
        train_loader = COLORMNISTTREE(phase='train', directory=configs.base_dir, folder=configs.data_folder,
                                      batch_size=configs.batch_size,
                                      random_seed=configs.seed, shuffle=True)
        test_loader = CLEVRTREE(phase='test', base_dir=osp.join(configs.base_dir, configs.data_folder),
                                batch_size=configs.batch_size,
                                random_seed=configs.seed, shuffle=False)
        gen_loader = COLORMNISTTREE(phase='test', directory=configs.base_dir, folder=configs.data_folder,
                                    batch_size=configs.batch_size,
                                    random_seed=configs.seed, shuffle=False)
    else:
        raise ValueError('invalid dataset folder name {}'.format(configs.data_folder))

    # hack, parameter
    im_size = gen_loader.im_size[2]

    # model
    if configs.net == 'PNP':
        Net = PNPNet
    elif configs.net == 'SIMPLE':
        Net = SimplePNPNet
    else:
        raise ValueError('configs.net ?= ', configs.net, 'not a valid value')
    model = Net(hiddim=configs.hiddim, latentdim=configs.latentdim,
                word_size=[configs.latentdim, configs.word_size, configs.word_size], pos_size=[8, 1, 1],
                nres=configs.nr_resnet, nlayers=4,
                nonlinear='elu', dictionary=train_loader.dictionary,
                op=[configs.combine_op, configs.describe_op],
                lmap_size=im_size // 2 ** configs.ds,
                downsample=configs.ds, lambdakl=-1, bg_bias=configs.bg_bias,
                normalize=configs.normalize,
                loss=configs.loss, debug_mode=False)

    if configs.checkpoint is not None and len(configs.checkpoint) > 0:
        model.load_state_dict(torch.load(configs.checkpoint))
        print('load model from {}'.format(configs.checkpoint))
    else:
        model.apply(weights_init)

    if configs.mode == 'train':
        train(model, train_loader, test_loader, gen_loader, configs=configs)
    elif configs.mode == 'test':
        print(exp_dir)
        print('Start generating...')
        generate(model, gen_loader=gen_loader, num_sample=configs.num_samples, target_dir=exp_dir)
    elif configs.mode == 'visualize':
        print(exp_dir)
        print('Start visualizing...')
        visualize(model, num_sample=50, target_dir=exp_dir)
    elif configs.mode == 'sample':
        print('Sampling')
        if configs.all_combinations:
            sample_all_combinations_tree(model, test_loader=gen_loader, tree_idx=configs.tree_idx, base_dir=exp_dir,
                                         num_sample=configs.num_samples)
        else:
            sample_tree(model, test_loader=gen_loader, tree_idx=configs.tree_idx, base_dir=exp_dir,
                        num_sample=configs.num_samples)
    else:
        raise ValueError('Wrong mode given:{}'.format(configs.mode))


def train(model, train_loader, test_loader, gen_loader, configs):
    model.train()
    # optimizer, it's better to set up lr for some modules separately so that the whole training become more stable
    params = [
        {'params': model.reader.parameters(), 'lr': 0.2 * configs.lr},
        {'params': model.h_mean.parameters(), 'lr': 0.1 * configs.lr},
        {'params': model.h_var.parameters(), 'lr': 0.1 * configs.lr},
        {'params': model.writer.parameters()},
        {'params': model.pos_dist.parameters()},
        {'params': model.combine.parameters()},
        {'params': model.describe.parameters()},
        {'params': model.box_vae.parameters(), 'lr': 10 * configs.lr},
        {'params': model.offset_vae.parameters(), 'lr': 10 * configs.lr},
        {'params': model.renderer.parameters()},
        {'params': model.bias_mean.parameters()},
        {'params': model.bias_var.parameters()}
        ]
    if configs.net == 'PNP':
        params.append({'params': model.vis_dist.parameters()})
    elif configs.net == 'SIMPLE':
        pass
    else:
        raise ValueError('configs.net ?= ', configs.net, 'not a valid value')
    optimizer = optim.Adamax(params, lr=configs.lr)

    model.cuda()

    trainer = PNPNetTrainer(model=model, train_loader=train_loader, val_loader=test_loader, gen_loader=gen_loader,
                            optimizer=optimizer,
                            configs=configs)

    minloss = 1000
    for epoch_num in range(0, configs.epochs + 1):
        timestamp_start = datetime.datetime.now(pytz.timezone('America/New_York'))
        trainer.train_epoch(epoch_num, timestamp_start)
        if epoch_num % configs.validate_interval == 0 and epoch_num > 0:
            minloss = trainer.validate(epoch_num, timestamp_start, minloss)
        if epoch_num % configs.sample_interval == 0 and epoch_num > 0:
            trainer.sample(epoch_num, sample_num=8, timestamp_start=timestamp_start)
        if epoch_num % configs.save_interval == 0 and epoch_num > 0:
            torch.save(model.state_dict(),
                       osp.join(configs.exp_dir, 'checkpoints', 'model_epoch_{0}.pth'.format(epoch_num)))


def generate(model, gen_loader, num_sample, target_dir):
    model.eval()
    model.cuda()

    epoch_end = False
    sample_dirs = []
    for i in range(num_sample):
        sample_dir = osp.join(target_dir, 'test-data-{}'.format(0))
        if not osp.isdir(sample_dir):
            os.mkdir(sample_dir)
        sample_dirs.append(sample_dir)

    image_idx = 0
    while epoch_end is False:
        data, trees, _, epoch_end = gen_loader.next_batch()

        with torch.no_grad():
            data = Variable(data).cuda()
  
            samples_image_dict = dict()
            batch_size = None
            for j in range(num_sample):
                sample = model.generate(data, trees)
                if not batch_size:
                    batch_size = sample.size(0)
                for i in range(0, sample.size(0)):
                    samples_image_dict.setdefault(i, list()).append(sample.cpu().data.numpy().transpose(0, 2, 3, 1)[i])
                model.clean_tree(trees)
  
            for i in range(batch_size):
                samples = np.clip(np.stack(samples_image_dict[i], axis=0), -1, 1)
                for j in range(num_sample):
                    sample = samples[j]
                    scipy.misc.imsave(osp.join(sample_dirs[j], 'image{:05d}_{:d}.png'.format(image_idx, j)), sample)
                    print(j)
                image_idx += 1

def sample_tree(model, test_loader, tree_idx, base_dir, num_sample):
    """
    Sample multiple image instances for a specified tree in the test dataset
    :param model: model
    :param test_loader: test loader
    :param tree_idx: the tree's index
    :param base_dir: base directory for saving results
    :param num_sample: number of samples to sample for the specified tree structure
    """
    model.eval()
    model.cuda()

    target_dir = osp.join(base_dir, 'tree-{}'.format(tree_idx))
    if not osp.isdir(target_dir):
        os.makedirs(target_dir)

    i = 0
    epoch_end = False
    while epoch_end is False and i <= tree_idx:
        data, tree, _, epoch_end = test_loader.next_batch()
        if i < tree_idx:
            continue
        with torch.no_grad():
            data = Variable(data).cuda()
            for i in range(num_sample):
                sample = model.generate(data, tree)
                sample = np.clip(sample.cpu().data.numpy().transpose(0, 2, 3, 1), -1, 1)
                sample = sample[0]
                scipy.misc.imsave(osp.join(target_dir, 'sample-{:05d}.png'.format(i)), sample)
                model.clean_tree(tree)
            tabular(target_dir, osp.join(target_dir, '{}.png'.format(tree_idx)))



def sample_all_combinations_tree(model, test_loader, tree_idx, base_dir, num_sample):
    """
    Sample multiple image instances for a specified tree in the test dataset
    :param model: model
    :param test_loader: test loader
    :param tree_idx: the tree's index
    :param base_dir: base directory for saving results
    :param num_sample: number of samples to sample for the specified tree structure
    """
    model.eval()
    model.cuda()

    for i in range(1, 5):
        target_dir = osp.join(base_dir, '{}'.format(i))
        if not osp.isdir(target_dir):
            os.makedirs(target_dir)
    attrs = [
        ['sphere', 'cube', 'cylinder'],
        ['red', 'brown', 'purple', 'cyan', 'yellow', 'green', 'gray', 'blue'],
        ['metal', 'rubber'],
        ['small', 'large']
    ]
    combinations = []
    get_all_combination_subsets(attrs, combinations)

    i = 0
    epoch_end = False
    while epoch_end is False and i <= tree_idx:
        data, trees, _, epoch_end = test_loader.next_batch()
        if i < tree_idx:
            continue
        tree_bk = copy.deepcopy(trees)
        with torch.no_grad():
            data = Variable(data).cuda()
            for comb in combinations:
                new_comb = [c for c in comb if c != 'zero']
                comb_str = '_'.join(new_comb)
                num_comb = len(new_comb)
                print(comb_str)
                target_dir = osp.join(base_dir, 'sample-combinations/tree-{}'.format(comb_str))
                if not osp.isdir(target_dir):
                    os.makedirs(target_dir)
                tree = copy.deepcopy(tree_bk)
                modify_tree(tree[0], comb)
                for i in range(num_sample):
                    sample = model.generate(data, tree)
                    sample = np.clip(sample.cpu().data.numpy().transpose(0, 2, 3, 1), -1, 1)
                    sample = sample[0]
                    scipy.misc.imsave(osp.join(target_dir, 'sample-{:05d}.png'.format(i)), sample)
                    model.clean_tree(tree)
                tabular(target_dir, osp.join(base_dir, '{}/{}.png'.format(num_comb, comb_str)))


def fix(base_dir):
    attrs = [
        ['sphere', 'cube', 'cylinder'],
        ['red', 'brown', 'purple', 'cyan', 'yellow', 'green', 'gray', 'blue'],
        ['metal', 'rubber'],
        ['small', 'large']
    ]
    combinations = []
    get_all_combination_subsets(attrs, combinations)
    for i in range(1, 5):
        target_dir = osp.join(base_dir, 'attrs-{}'.format(i))
        if not osp.isdir(target_dir):
            os.makedirs(target_dir)
    for comb in combinations:
        num_attr = len(comb)
        filename = '_'.join(comb) + '.png'
        dst = osp.join(base_dir, 'attrs-{}/{}'.format(num_attr, filename))
        src = osp.join(base_dir, filename)
        copyfile(src, dst)


def get_all_combination_subsets(lists, result, list_i=0, collected=[]):
    if list_i == len(lists):
        result.append(collected[:])
        return
    for item in lists[list_i]:
        collected.append(item)
        get_all_combination_subsets(lists, result, list_i+1, collected)
        collected.pop()
    if list_i != 0:
        collected.append('zero')
        get_all_combination_subsets(lists, result, list_i+1, collected)
        collected.pop()

def modify_tree(tree, words):
    for i, w in enumerate(words):
        tree.word = w
        if i < len(words)-1:
            tree = tree.children[0]
    tree.children = []
    tree.num_children = 0


def visualize(model, num_sample, target_dir):
    """
    Visualize intermediate results of PNPNet (e.g. visual concepts or partially composed image)
    :param model: model
    :param num_sample: number of samples to generate
    :param target_dir: target directory for saving results
    """
    model.eval()

    sample_dir = osp.join(target_dir, 'visualize-data')
    if not osp.isdir(sample_dir):
        os.mkdir(sample_dir)

    mode = 'transform'

    for i in range(0, len(model.dictionary)):
        if i not in [1, 2, 5]:
            continue
        print('index ', i, 'current word:', model.dictionary[i])

        data = Variable(torch.zeros(num_sample, len(model.dictionary))).cuda()
        data[:, i] = 1
        vis_dist = model.vis_dist(data)

        if mode == 'full':
            prior_mean, prior_var = model.renderer(vis_dist)
        elif mode == 'transform':
            prior_mean, prior_var = Variable(torch.zeros(vis_dist[0].size())).cuda(), Variable(
                torch.zeros(vis_dist[1].size())).cuda()
            sz = vis_dist[0].size(2)
            tsz = 5
            vis_dist[0] = model.transform(vis_dist[0], [tsz, tsz])
            vis_dist[1] = model.transform(vis_dist[1], [tsz, tsz])
            prior_mean[:, :, 5:5 + tsz, 5:5 + tsz] = vis_dist[0]
            prior_var[:, :, 5:5 + tsz, 5:5 + tsz] = vis_dist[1]
            prior_mean, prior_var = model.renderer([prior_mean, prior_var])
        else:
            raise ValueError('Invalid mode name {}'.format(mode))

        z_map = model.sampler(prior_mean, prior_var)

        rec = model.writer(z_map)

        for im in range(0, rec.size(0)):
            scipy.misc.imsave(osp.join(sample_dir, 'image-{}-{:05d}.png'.format(model.dictionary[i], im)),
                              rec[im].cpu().data.numpy().transpose(1, 2, 0))


def tabular(img_dir, output_path):
    files = [join(img_dir, f) for f in listdir(img_dir) if isfile(join(img_dir, f)) ]
    size = 64
    num = len(files) ** 0.5
    width = int(num) * size
    new_im = Image.new('RGB', (width,  width))

    index = 0
    for i in xrange(0, width, size):
        for j in xrange(0, width, size):
            im = Image.open(files[index])
            im.thumbnail((size, size))
            new_im.paste(im, (i, j))
            index += 1
    new_im.save(output_path)


if __name__ == '__main__':
    main()
