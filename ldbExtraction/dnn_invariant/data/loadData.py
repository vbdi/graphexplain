
import torch
import numpy as np
import os
from skimage import io
from skimage.transform import resize
from skimage.color import gray2rgb

'''
If the dataset is too large, you can choose to limit the size of each class
'''
max_per_class = 2000



'''
This is the folder that contains the data
It should contain a 'train' folder and a 'test' folder
Inside these two folders, you should have a folder for each class, like 'cats' and 'dogs'
And within the class folders, just put the relevant images inside
'''
#path = '/data/peter/ZhangLab/CellData/OCT/' # Retina OCT
path = '/home/mohit/Mohit/unsupervised_cluster/mnist/data/assira/' # Cats and Dogs



'''
This is the path where the loaded data is saved
'''
save_path = './data/'



'''
This is the list of class folders you want to read
'''
#classes = ['NORMAL', 'CNV', 'DME', 'DRUSEN'] # Retina OCT
#classes = ['NORMAL', 'DME'] # Retina OCT, just normal and DME
classes = ['cats', 'dogs'] # Cats and Dogs



'''
This is the filename for the loaded data
e.g. If data_name = 'X', then you have will 'X_train.ntvt' and 'X_test.ntvt' (or .tvt if unnormalized)
'''
#data_name = 'ZhangLab' # Retina OCT
#data_name = 'ZhangLab_02' # Retina OCT, just NORMAL and DME
data_name = 'Assira' # Cats and Dogs



'''
Height and width of the loaded data
'''
H = 224
W = 224



'''
Set to True if using ImageNet normalization
'''
ImageNetNormalize = True



class LoadData():
    def __init__(self, class_ids_=None, validation_ratio_=0.3, save_root_=save_path, isNormalized_=False):
        self._save_pathheader = save_root_ + data_name # the root path to save transformed loaded data
        self._isNormalized    = isNormalized_

        if(self._isNormalized == True):
            self._file_suffix = '.ntvt'
        else:
            self._file_suffix = '.tvt'

        # load all raw data
        X_train, y_train = self.load_data(path + 'train')
        X_test, y_test = self.load_data(path + 'test')

        # transform to torch.tensor
        X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
        y_train = torch.from_numpy(y_train).type(torch.LongTensor)
        X_test  = torch.from_numpy(X_test).type(torch.FloatTensor)
        y_test  = torch.from_numpy(y_test).type(torch.LongTensor)

        X_train = X_train.view(X_train.size(0), 3, H, W)
        X_test  = X_test.view(X_test.size(0), 3, H, W)

        print('org_train mean: ', X_train.mean())

        if (self._isNormalized == True):
            print(X_train.mean())
            print(X_train.std())
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])

            for i in range(3):
                X_train[:, i, :, :] = (X_train[:, i, :, :] - mean[i]) / std[i]
                X_test[:, i, :, :] = (X_test[:, i, :, :] - mean[i]) / std[i]

            print('normlaized_train mean: ', X_train.mean())

        # save data
        if class_ids_ == None:
            save_path_train = self._save_pathheader + '_Train' + self._file_suffix
            save_path_test = self._save_pathheader + '_Test' + self._file_suffix

            with open(save_path_train, 'wb') as fid:
                torch.save((X_train, y_train), fid)
                print('save to ' + save_path_train)
            with open(save_path_test, 'wb') as fid:
                torch.save((X_test, y_test), fid)
                print('save to' + save_path_test)
        else:
            idx_new_train    = self._getClassIdx(y_new_train, class_ids_)
            idx_new_valid    = self._getClassIdx(y_new_valid, class_ids_)
            idx_test         = self._getClassIdx(y_test, class_ids_)

            save_path_train = self._save_pathheader + '_' + ''.join(map(str, class_ids_)) + '_Train' + self._file_suffix
            save_path_valid = self._save_pathheader + '_' + ''.join(map(str, class_ids_)) + '_Valid' + self._file_suffix
            save_path_test  = self._save_pathheader + '_' + ''.join(map(str, class_ids_)) + '_Test' + self._file_suffix

            self._export(idx_new_train, X_new_train, y_new_train, save_path_train)
            self._export(idx_new_valid, X_new_valid, y_new_valid, save_path_valid)
            self._export(idx_test, X_test, y_test, save_path_test)


    def _export(self, idx_, X_, y_, save_path_):
        with open(save_path_, 'wb') as fid:
            num_ = 0
            for idx_cls in idx_:
                num_ += idx_cls.shape[0]

            y_ = torch.zeros(num_, dtype=torch.long)
            ptr = 0
            for i in range(idx_.__len__()):
                y_[ptr:ptr + idx_[i].__len__()] = i
                ptr += idx_[i].__len__()

            idx_ = np.concatenate(idx_, axis=0)
            X_ = X_[idx_]

            torch.save((X_, y_), fid)
            print('save to' + save_path_)


    def _getClassIdx(self, y, class_ids_):
        idx = []
        for cls in class_ids_:
            idx.append(np.where(y == cls)[0])

        return idx

    def load_data(self, path):

        for i in range(len(classes)):
            images_path = os.path.join(path, classes[i])
            fnames = os.listdir(images_path)

            fnames = fnames[:max_per_class]

            images_size = len(fnames)
            print('Class ', classes[i], ' has ', str(images_size), ' images.')

            if i == 0:
                images = np.zeros((images_size, 3, H, W))
                labels = np.repeat(i, images_size)
                pre_length = 0
            else:
                images = np.concatenate((images, np.zeros((images_size, 3, H, W))))
                pre_length = len(labels)
                labels = np.concatenate((labels, np.repeat(i, images_size)))

            for index, fname in enumerate(fnames):

                image_path = os.path.join(images_path, fname)
                img = io.imread(image_path)

                img = resize(img, (H, W))
                if img.ndim == 2:
                    img = gray2rgb(img)

                img = np.moveaxis(img, -1, 0)
                images[pre_length + index] = img

        print(images.shape)
        print(labels.shape)

        return images, labels


# ============================================
# Execute to make data
# ============================================

LoadData(isNormalized_=ImageNetNormalize)
