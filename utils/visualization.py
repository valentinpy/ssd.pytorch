import visdom
import torch

class vis_plot:
    def __init__(self):
        self.viz = visdom.Visdom()


    def create_vis_plot(self, _xlabel, _ylabel, _title, _legend):
        return self.viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1, 3)).cpu(),
            opts=dict(
                xlabel=_xlabel,
                ylabel=_ylabel,
                title=_title,
                legend=_legend
            )
        )


    def update_vis_plot(self, iteration, loc, conf, window1, window2, update_type, epoch_size=1):
        self.viz.line(
            X=torch.ones((1, 3)).cpu() * iteration,
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
            win=window1,
            update=update_type
        )
        # initialize epoch plot on first iteration
        if (iteration == 0) and (window2 is not None):
            self.viz.line(
                X=torch.zeros((1, 3)).cpu(),
                Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
                win=window2,
                update=True
            )