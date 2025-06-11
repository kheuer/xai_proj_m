import os

from torch.autograd import Variable
from collections import OrderedDict
from .base_model import BaseModel
from . import networks
import torch


class TestModel(BaseModel):
    def __init__(self, ckpt_path):
        self.ckpt_path = ckpt_path
        self.initialize(None)

    def name(self):
        return 'TestModel'

    def initialize(self, opt):

        #BaseModel.initialize(self, opt)
        #self.label_intensity = opt.label_intensity
        #self.label_intensity_styletransfer = opt.label_intensity_styletransfer
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        #self.loss_names = []
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        # self.visual_names = ['real_A', 'fake_B']
        #self.visual_names = ['fake_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['G', 'DeConv']
        self.save_dir = os.path.join(self.ckpt_path)
        self.gpu_ids = [0]
        self.netG = networks.define_stochastic_G(nlatent=16, #opt.nlatent,
                                                        input_nc=3,#opt.input_nc,
                                                        output_nc=3, #opt.output_nc,
                                                        ngf=32,#opt.ngf,
                                                     which_model_netG='resnet_9blocks',
                                                     norm='instance',
                                                    use_dropout=False,#opt.use_dropout,
                                                     gpu_ids=self.gpu_ids)
        self.netDeConv = networks.define_InitialDeconv(gpu_ids=self.gpu_ids)
        #self.load_networks(opt.which_epoch)
        self.load_networks('latest')
        #self.print_networks(opt.verbose)

    def set_input(self, input, label_intensity_style_transfer):
        # we need to use single_dataset mode
        input_A = input['A']
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0])
        self.input_A = input_A
        #self.image_paths = input['A_paths']
        self.add_item = self.netDeConv(Variable(torch.FloatTensor([label_intensity_style_transfer]).view(1,4,1,1)).cuda(self.gpu_ids[0]))#, async=True))

    def test(self):
        with torch.no_grad():
            self.real_A = Variable(self.input_A)
            self.fake_B = self.netG(self.real_A, self.add_item)

    def get_current_visuals(self):
        # return OrderedDict([('real_A', self.real_A), ('fake_B', self.fake_B)])
        return OrderedDict([('fake_B', self.fake_B)])

    # def load_networks(self, which_epoch):
    #     for name in self.model_names:
    #         if isinstance(name, str):
    #             save_filename = '%s_net_%s.pth' % (which_epoch, name)
    #             save_path = os.path.join(self.save_dir, save_filename)
    #             save_path = save_path.replace('latest_net_G.pth', 'latest_net_G_B.pth')
    #             net = getattr(self, 'net' + name)
    #             if len(self.gpu_ids) > 0 and torch.cuda.is_available():
    #                 net.load_state_dict(torch.load(save_path))
    #             else:
    #                 net.load_state_dict(torch.load(save_path))