import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from util import html


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    weights = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.5, 0.5]
    ]

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()

    for weight in weights:

        opt.label_intensity_styletransfer = weight
        model = create_model(opt)
        visualizer = Visualizer(opt)
        # create website
        web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
        webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
        current_dir = webpage.get_image_dir()
        weight_dir = '_'.join(str(w).replace('.', ',') for w in weight)
        webpage.img_dir = os.path.join(current_dir, weight_dir)
        os.makedirs(webpage.get_image_dir(), exist_ok=True)

        for i, data in enumerate(dataset):
            if i >= opt.how_many:
                break
            model.set_input(data, 0, 0)
            model.test()
            visuals = model.get_current_visuals()
            img_path = model.get_image_paths()
            print('%04d: process image... %s' % (i, img_path))
            visualizer.save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio)

        webpage.save()
