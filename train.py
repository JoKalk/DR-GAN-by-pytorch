import time
import sys
from tensorboardX import SummaryWriter
sys.path.append('options')
from train_options import TrainOptions
sys.path.append('data')
from data_loader import CreateDataLoader
sys.path.append('model')
from model_Loader import CreateModel
sys.path.append('util')
from utils import error
from tqdm import tqdm

def run():
    opt = TrainOptions().parse()
    data_loader = CreateDataLoader(opt, False)
    #data_loader_mask = CreateDataLoader(opt, True)
    model = CreateModel(opt)

    writer = SummaryWriter('logs')

    err = error(model.save_dir)
    for epoch in range(opt.count_epoch + 1,  opt.epochs + 1):
        epoch_start_time = time.time()
        err.initialize()
        pbar = tqdm(total=len(data_loader))
        for i, data in enumerate(data_loader):#for i, (data, data_mask) in enumerate(zip(data_loader, data_loader_mask)):
            model.forward(data)
            pbar.update(1)
            model.optimize_G_parameters()
            if(i % opt.D_interval == 0):
                #model.forward(data_mask)
                model.optimize_D_parameters()
            #else:
            #    model.forward(data)
            #    model.optimize_G_parameters()
            err.add(model.Loss_G.data.item(), model.Loss_D.data.item())
        pbar.close()

        LOSSG, LOSSD = err.print_errors(epoch)
        writer.add_scalar('loss_g', LOSSG, epoch)
        writer.add_scalar('loss_d', LOSSD, epoch)
        print('End of epoch {0} \t Time Taken: {1} sec\n'.format(epoch, time.time()-epoch_start_time))
        model.save_result(epoch)
        if epoch % opt.save_epoch_freq == 0:
            print('Saving the model at the end of epoch {}\n'.format(epoch))
            model.save(epoch)

if __name__ == '__main__':
    run()