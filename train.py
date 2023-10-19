import torch
import time
import os
from torchvision import transforms
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd
from model import CNN
from PIL import Image

if __name__ == "__main__":

    # TODO: make 2 folders, train and test. put 2 subfolders in each, real and deepfake
    traintest_adr = 'data_folder'

    # making traintest df
    data = []
    classes = ['real', 'deepfake']

    for split in ['train', 'test']:
        split_path = os.path.join(traintest_adr, split)

        for class_name in classes:
            class_path = os.path.join(split_path, class_name)
            files = os.listdir(class_path)

            for file in files:
                file_adr = os.path.join(class_path, file)
                data.append({'file_adr': file_adr,
                             'class': 0 if class_name == 'real' else 1,
                             'train/test': 0 if split == 'train' else 1})

    data_df = pd.DataFrame(data)
    train_df = data_df.loc[data_df['train/test'] == 0, :]
    test_df = data_df.loc[data_df['train/test'] == 1, :]
    train_df = train_df.sample(frac=1).reset_index()
    test_df = test_df.sample(frac=1).reset_index()
    ########################################

    # parameters
    lr = 5e-4
    #####################
    weight_decay = 1e-5
    nr_epochs = 12
    lr_decay = 0.9
    train_batch_size = 128
    test_batch_size = 16
    model_param_adr = r'models/Xception_upb_fullface_epoch_25_param_FF++_186_2346.pkl'
    save_adr = ''

    transf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((299, 299)),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    frozen_params = 50

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device: ', device)

    model = CNN(pretrained=True, finetuning=True, frozen_params=frozen_params, architecture='Xception')
    model.to(device)

    cnn_params = list(model.parameters())
    optimizer = torch.optim.Adam(cnn_params, lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()

    for epoch in range(0, nr_epochs + 1):

        print('Epoch: ', epoch)
        train_loss = 0.0
        predictions_vect = []
        targets_vect = []
        prediction_df = pd.DataFrame(columns=['GT', 'prediction'])

        model.train()

        # shuffle dataset
        train_df = train_df.sample(frac=1).reset_index(drop=True)

        for i in range(len(train_df)//train_batch_size):

            t = time.time()
            data = []
            targets = train_df.loc[i*train_batch_size:i*train_batch_size+train_batch_size, 'class'].values
            # read images
            for img in train_df.loc[i*train_batch_size:i*train_batch_size+train_batch_size, 'file_adr']:
                try:
                    X = np.array(Image.open(img), dtype=np.float32)
                    X = transf(X)
                    data.append(X)
                except Exception as e:
                    print(f'Could not read {img}, error: \n{e}')

            labels = torch.Tensor(targets)
            data = torch.stack(data)
            data, targets = data.to(device), targets.to(device)

            outputs_gpu = model(data)

            outputs = outputs_gpu.to('cpu').flatten()
            targets = targets.to('cpu')

            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predictions_vect.append(outputs.detach())
            targets_vect.append(targets)

            train_loss += loss.item()
            batch_t = time.time() - t
            avg_loss = train_loss / (i+1)

            if len(torch.unique(torch.cat(targets_vect).flatten())) > 1:
                auc_train = roc_auc_score(torch.cat(targets_vect).flatten(), torch.cat(predictions_vect).flatten())
            else:
                auc_train = '-'

            print('Minibatch: ' + str(i) + '/' + str(len(train_df)//train_batch_size) + ' Loss avg: ' + str(avg_loss) +
                  ' AUC total: ' + str(auc_train) )
            train_loss = 0.0

        # Saving model
        torch.save(model.state_dict(),
                   os.path.join(save_adr,
                               'Xception_epoch_' + str(epoch) + '_param_' +
                                str(time.gmtime()[2]) + str(time.gmtime()[1]) + '_' + str(time.gmtime()[3]) + str(
                                    time.gmtime()[4]) + '.pkl'))


        # test
        print('Starting test')
        model.eval()
        test_loss = 0.0
        predictions_vect = []
        targets_vect = []

        for i in range(len(test_df) // test_batch_size):
            data = []
            targets = test_df.loc[i * test_batch_size:i * test_batch_size + test_batch_size, 'class'].values

            for img in test_df.loc[i * test_batch_size:i * test_batch_size + test_batch_size, 'file_adr']:
                try:
                    X = np.array(Image.open(img), dtype=np.float32)
                    X = transf(X)
                    data.append(X)
                except Exception as e:
                    print(f'Could not read {img}, error: \n{e}')

            labels = torch.Tensor(targets)
            data = torch.stack(data)
            data, targets = data.to(device), targets.to(device)

            with torch.no_grad():
                outputs_gpu = model(data)

            outputs = outputs_gpu.to('cpu').flatten()
            targets = targets.to('cpu')

            loss = criterion(outputs, targets)

            predictions_vect.append(outputs.detach())
            targets_vect.append(targets)

            test_loss += loss.item()

        # Calculate metrics for the entire test dataset
        if len(torch.unique(torch.cat(targets_vect).flatten())) > 1:
            auc_test = roc_auc_score(torch.cat(targets_vect).flatten(), torch.cat(predictions_vect).flatten())
        else:
            auc_test = '-'

        print('Testing completed. Average Loss: ' + str(test_loss / (i + 1)) + ' AUC total: ' + str(auc_test))