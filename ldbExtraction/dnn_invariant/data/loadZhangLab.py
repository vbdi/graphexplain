
import torch
import numpy as np

max_per_class = 1000

W = 224
H = 224

import torchvision.transforms as transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

class LoadZhangLab():
    def __init__(self, class_ids_=None, validation_ratio_=0.3, save_root_='/data/peter', isNormalized_=False):
        self._save_pathheader = save_root_ + '/ZhangLab/CellData/OCT/ZhangLab' # the root path to save transformed loaded data
        self._isNormalized    = isNormalized_

        if(self._isNormalized == True):
            self._file_suffix = '.ntvt'
        else:
            self._file_suffix = '.tvt'

        # load all raw data
        X_train, y_train = self.load_zhanglab('/data/peter/ZhangLab/CellData/OCT/train')
        X_test, y_test = self.load_zhanglab('/data/peter/ZhangLab/CellData/OCT/test')
        #X_train, y_train = self.load_assira('s3://vbdai-share/Peter/data/assira/train')
        #X_test, y_test = self.load_assira('s3://vbdai-share/Peter/data/assira/test')

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

            #X_new_train = 1.0 * (X_new_train - mean) / std
            #X_new_valid = 1.0 * (X_new_valid - mean) / std
            #X_test      = 1.0 * (X_test - mean) / std

            print('normlaized_train mean: ', X_train.mean())

        # save data
        if class_ids_ == None:
            save_path_train = self._save_pathheader + '_all_1000_Train' + self._file_suffix
            save_path_test  = self._save_pathheader  + '_all_1000_Test' + self._file_suffix

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

    # This load_mnist method is directly copied from utils/mnist_reader.py
    # from https://github.com/zalandoresearch/fashion-mnist
    def load_zhanglab(self, path):
        import os
        import numpy as np
        from skimage import io
        from skimage.transform import resize
        from skimage.color import gray2rgb
        from matplotlib.image import imsave

        classes = ['NORMAL', 'CNV', 'DME', 'DRUSEN']


        for i in range(len(classes)):
            images_path = os.path.join(path, classes[i])
            fnames = os.listdir(images_path)

            if len(fnames) > max_per_class:
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


        """Load MNIST data from `path`"""

        '''
        images_cats_path = os.path.join(path, 'NORMAL')
        images_dogs_path = os.path.join(path, 'DME')

        cats_fnames = os.listdir(images_cats_path)
        dogs_fnames = os.listdir(images_dogs_path)

        cats_fnames = cats_fnames[:5000]
        dogs_fnames = dogs_fnames[:5000]

        cats_size = len(cats_fnames)
        dogs_size = len(dogs_fnames)

        labels = np.concatenate((np.repeat(1, cats_size), np.repeat(0, dogs_size)))

        images = np.zeros((len(labels), 3, H, W))

        for index, fnames in enumerate(cats_fnames):

            cats_image = os.path.join(images_cats_path, fnames)
            img = io.imread(cats_image)
            img = resize(img, (H, W))
            if img.ndim == 2:
                img = gray2rgb(img)
            img = np.moveaxis(img, -1, 0)
            images[index] = img

        for index, fnames in enumerate(dogs_fnames):
            dogs_image = os.path.join(images_dogs_path, fnames)
            img = io.imread(dogs_image)
            if img.ndim == 2:
                img = gray2rgb(img)
            img = resize(img, (H, W))
            img = np.moveaxis(img, -1, 0)
            images[index + cats_size] = img
        '''

        print(images.shape)
        print(labels.shape)

        return images, labels


# ============================================
# Execute to make data
# ============================================


#LoadAssira(save_root_='s3://vbdai-share/Peter/data', isNormalized_=True)
#LoadAssira(save_root_='s3://vbdai-share/Peter/data', isNormalized_=False)

LoadZhangLab(isNormalized_=True)
#LoadZhangLab(isNormalized_=False)
