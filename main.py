import time
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

from utils.Transforms import FaceLandmarksDataset, Transforms, Network
from utils.file_managment import FileOp
from utils.plot_graph import Plot

visual = True


def create_pts(pr, name, bool_val):
    if bool_val:
        path = 'outputs/pr/'
    else:
        path = 'outputs/orig/'
    for ind in range(len(pr)):
        print(F'Create file {path}{name}file{ind}.pts ')
        with open(f'{path}{name}file{ind}.pts', 'w') as file:
            file.write('version: 1\nn_points: 68\n{\n')
            for i in pr[ind]:
                # print(str(i).replace('tensor([', '').replace('])', ''))
                file.write((str(i).replace('tensor([', '').replace('])', '').replace(' ', '').replace(',', ' ')) + '\n')
            # for i in pr_lst:
            #     file.write(i + '\n')
            file.write('}')
        file.close()
        ind += 1


def main():
    dataset = FaceLandmarksDataset(Transforms())
    len_valid_set = int(0.08 * len(dataset))
    len_train_set = len(dataset) - len_valid_set

    print("The length of Train set is {}".format(len_train_set))
    print("The length of Valid set is {}".format(len_valid_set))

    train_dataset, valid_dataset, = torch.utils.data.random_split(dataset, [len_train_set, len_valid_set])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1024, shuffle=True, num_workers=4)
    start_time = time.time()

    with torch.no_grad():
        best_network = Network()
        best_network.cuda()
        best_network.load_state_dict(torch.load('outputs/face_landmarks.pth'))
        best_network.eval()

        images, landmarks = next(iter(valid_loader))

        i = 0
        for x in iter(valid_loader):

            images = images.cuda()
            landmarks = (landmarks + 0.5) * 224
            predictions = (best_network(images).cpu() + 0.5) * 224
            predictions = predictions.view(-1, 68, 2)
            create_pts(predictions, i, True)
            create_pts(landmarks, i, False)
            i += 1

        if visual:
            plt.figure(figsize=(10, 40))
            for img_num in range(8):
                plt.subplot(8, 1, img_num + 1)
                plt.imshow(images[img_num].cpu().numpy().transpose(1, 2, 0).squeeze(), cmap='gray')
                plt.scatter(predictions[img_num, :, 0], predictions[img_num, :, 1], c='b', s=5)
                plt.scatter(landmarks[img_num, :, 0], landmarks[img_num, :, 1], c='g', s=5)
            plt.show()
            plt.savefig('CNN_res.png')

    print(i)
    print('Total number of test images: {}'.format(len(valid_dataset)))

    end_time = time.time()
    print("Elapsed Time : {}".format(end_time - start_time))
    print("Plot")


if __name__ == '__main__':
    main()
    orig_data_path = '/home/pavel/PycharmProjects/CNNLnd/outputs/orig'
    predictor_data_path = '/home/pavel/PycharmProjects/CNNLnd/outputs/pr'
    out_path = './outputs/330wccn'
    gr = Plot(gt_path=orig_data_path, predictions_path=predictor_data_path, output_path=out_path)
    gr.main()
