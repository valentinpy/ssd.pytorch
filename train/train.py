import time
import datetime

from torch.autograd import Variable
from utils.visualization import vis_plot
from train.train_tools import *
from data.get_data_train import *

def train(args, viz = None):
    model_name = args["model"]
    dataset_name = args["name"]
    imageset_name = args["image_set"].split("/")[-1].split(".")[0]
    image_fusion = args["image_fusion"]

    cuda = torch.cuda.is_available() and args['cuda']
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # get hyperparameters
    batch_size, epochs, gamma, learning_rate = get_hyperparams(args)

    # load dataset
    print('Preparing the dataset...')
    data_loader, dataset = get_dataset_dataloader_train(args)

    # build net
    print('Building net...')
    model, criterion, optimizer = build_training_net(args)

    # set train state
    model.train()

    # create plots
    if args['visdom']:
        vis_title = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S - ")
        vis_title += str(model_name + ".PyTorch on '" + args["name"] + ' | lr: ' + str(learning_rate))
        vis_legend = ['Model loss', 'n/d', 'Total Loss'] if model_name == "YOLO" else ['loc loss', 'conf loss', 'Total Loss']
        iter_plot = viz.create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)

    epoch_size = len(dataset) // batch_size

    print("Training model: '{}' on dataset: '{}'".format(model_name, dataset_name))
    print('Dataset length: {}'.format(len(dataset)))
    print('Epoch size: {}, batch size: {}'.format(epoch_size, batch_size))
    print("Image fusion: {}".format(image_fusion))

    iteration_total = 0
    step_index = 0
    loc_loss = 0
    conf_loss = 0

    for epoch in range(epochs):
        for batch_i, (imgs, targets) in enumerate(data_loader):

            # Measure learning time for a batch: begin
            t0 = time.time()

            if model_name == "YOLO":
                # load and format train data (images + annotations)
                imgs = Variable(imgs.type(Tensor))
                targets = Variable(targets.type(Tensor), requires_grad=False)

                # forward pass, loss computation and backward propagation
                optimizer.zero_grad()
                loss = model(imgs, targets)
                loss.backward()
                optimizer.step()

                # Measure learning time for a batch: end
                batch_time = time.time() - t0

                # log progress
                if iteration_total % 50 == 0:
                    conf_loss = model.losses["conf"]
                    print_learning_progress_xxx_YOLO(batch_i, batch_size, batch_time, data_loader, epoch, epoch_size, epochs, iteration_total, model, loss)

            elif model_name in {"MOBILENET2_SSD", "VGG_SSD"}:
                # check if learning rate must be decreased according to iteration number
                lr_steps = args['ssd_lr_steps']
                if iteration_total in lr_steps:
                    step_index += 1
                    adjust_learning_rate(optimizer, gamma, step_index, learning_rate)

                # load and format train data (images + annotations)
                if cuda:
                    images = imgs.cuda()
                    targets = [ann.cuda() for ann in targets]
                else:
                    print("Cuda is required to learn!")
                    raise NotImplementedError

                # forward pass
                out = model(images)

                # loss computation +  backpropagation
                optimizer.zero_grad()
                loss_l, loss_c = criterion(out, targets)
                loss = loss_l + loss_c
                loss.backward()
                optimizer.step()
                loc_loss = loss_l.data.item()
                conf_loss = loss_c.data.item()

                # Measure learning time for a batch: end
                batch_time = time.time() - t0

                # log progress
                if iteration_total % 50 == 0:
                    print_learning_progress_xxx_SSD(batch_i, batch_size, batch_time, loss_c, data_loader, epoch, epoch_size, epochs, iteration_total, loss_l)

            else:
                raise NotImplementedError

            model.seen += imgs.size(0)

            # save progress
            if ((iteration_total) % (args['save_frequency']) == 0) and iteration_total > 0:
                saved_model_name = "%s/%s__%s__%s__fusion-%d__iter-%d.weights" % (args['save_folder'], model_name, dataset_name, imageset_name, image_fusion, epoch * len(data_loader) + batch_i)
                model.save_weights(saved_model_name)
                # torch.save(ssd_net.state_dict(), model_name)
                print("Weights saved for epoch {}/{}, batch {}/{}".format(epoch, epochs, batch_i, len(data_loader)))

            plot_loss1 = loc_loss if model_name in{"VGG_SSD", "MOBILENET2_SSD"} else loss.item()
            plot_loss2 = conf_loss if model_name in{"VGG_SSD", "MOBILENET2_SSD"} else 0

            if args['visdom']:
                viz.update_vis_plot(iteration_total, plot_loss1, plot_loss2, iter_plot, None, 'append')
            iteration_total += 1

    print("Finished training at {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))



def main(args):
    #prepare for CUDA
    cuda = torch.cuda.is_available() and args['cuda']
    if cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        print("WARNING: Cuda is not used!")
        torch.set_default_tensor_type('torch.FloatTensor')

    # prepare visualization
    if args['visdom']:
        viz = vis_plot()
    else:
        viz = None

    # train
    train(args, viz)

if __name__ == "__main__":
    args = arg_parser(role="train")
    check_args(args, role="train")
    main(args)
