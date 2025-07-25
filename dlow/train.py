import time
import traceback

from util.util import tensor2im
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from prepare_run import prepare
from notify import send_progress_bar, send_discord_notification, ALERT_URL, INFO_URL
if __name__ == '__main__':
    try:
        opt = TrainOptions().parse()

        prepare(opt)


        data_loader = CreateDataLoader(opt)
        dataset = data_loader.load_data()
        dataset_size = len(data_loader)
        print('#training images = %d' % dataset_size)

        model = create_model(opt)
        visualizer = Visualizer(opt)
        total_steps = 0
        total_epochs = opt.niter + opt.niter_decay + 1
        send_discord_notification(INFO_URL, "start training")
        for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
            epoch_start_time = time.time()
            iter_data_time = time.time()
            epoch_iter = 0

            for i, data in enumerate(dataset):

                iter_start_time = time.time()
                if total_steps % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time
                visualizer.reset()
                # if total_steps % 5 == 0:
                #     sign = '0'
                # elif total_steps % 5 == 1:
                #     sign = '1'
                # elif total_steps % 5 == 2:
                #     sign = '2'
                # elif total_steps % 5 == 3:
                #     sign = '3'
                # elif total_steps % 5 == 4:
                #     sign = '4'
                # else:
                #     print("Error occur when getting the 0, 1, 0to1")
                total_steps += opt.batchSize
                epoch_iter += opt.batchSize
                model.set_input(data, '0')
                model.optimize_parameters()

                if total_steps % opt.display_freq == 0:
                    save_result = total_steps % opt.update_html_freq == 0
                    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)


                if total_steps % opt.print_freq == 0:
                    losses = model.get_current_losses()
                    t = (time.time() - iter_start_time) / opt.batchSize
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                    if opt.display_id > 0:
                        visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)

                if total_steps % opt.save_latest_freq == 0:
                    print('saving the latest model (epoch %d, total_steps %d)' %
                          (epoch, total_steps))
                    model.save_networks('latest')

                iter_data_time = time.time()
            if epoch % opt.save_epoch_freq == 0:
                print('saving the model at the end of epoch %d, iters %d' %
                      (epoch, total_steps))
                model.save_networks('latest')
                model.save_networks(epoch)

            print('End of epoch %d / %d \t Time Taken: %d sec' %
                  (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

            model.update_learning_rate()

            if opt.discord:
                send_progress_bar(epoch, total_epochs, 0, INFO_URL, time.time() - epoch_start_time, 0, '')
                visuals = model.get_current_visuals()
                for label, image in visuals.items():
                    image_numpy = tensor2im(image)
                    send_discord_notification(INFO_URL, "", image_numpy, 'epoch%.3d_%s.png' % (epoch, label))


    except Exception as e:
        if opt.discord:
            send_discord_notification(ALERT_URL, "🚨 Oh, oh! Something went wrong 🚨")
            send_discord_notification(ALERT_URL, traceback.format_exc())
            print(traceback.format_exc())
        else:
            print(traceback.format_exc())