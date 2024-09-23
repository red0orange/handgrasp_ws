from options.test_options import TestOptions
from data import DataLoader
from models import create_model # bjkim
from utils.writer import Writer # bjkim


def run_test(epoch=-1, name='', writer=None, dataset_test=None):
    print('Running Test')
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    # opt.name = 'vae_lr_0002_bs_64_scale_1_npoints_128_radius_02_latent_size_2' #bjkim
    opt.name = 'vae_pretrained' #bjkim
    data_loader = DataLoader(opt) #bjkim 
    training_dataset, test_dataset, _ = data_loader.split_dataset(opt.dataset_split_ratio) #bjkim
    dataset_test = data_loader.create_dataloader(test_dataset, shuffle_batches=False) #bjkim
    dataset_test_size = len(test_dataset) #bjkim
    print('#test images = %d' % dataset_test_size) #bjkim
    
    model = create_model(opt)
    writer = Writer(opt) #bjkim
    # test
    point_clouds = []
    for data in dataset_test:
        model.set_input(data)
        ncorrect, nexamples = model.test()
        point_clouds.append(model.get_random_grasp_and_point_cloud())
        writer.update_counter(ncorrect, nexamples)
    writer.calculate_accuracy()
    writer.print_acc(epoch)
    writer.plot_acc(epoch)
    writer.plot_grasps(point_clouds, epoch)


if __name__ == '__main__':
    run_test()
