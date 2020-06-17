# encoding=utf-8
import numpy as np

class StandardScaler(object):
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def transform(self, data):
        data = np.array(data)
        if self.std == 0:
            print("WARNING: The std id zero, the transform will not divide the std.")
            trans = data - self.mean
        else:
            trans = (data - self.mean) / self.std
        return trans

    def inverse_transform(self, data):
        data = np.array(data)
        if self.std == 0:
            print("WARNING: The std id zero, the transform will not multiply the std.")
            trans = data + self.mean
        else:
            trans = (data * self.std) + self.mean
        return trans

    def save(self, path):
        np.savez_compressed(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        return cls(data['mean'], data['std'])

class OnehotScaler(object):
    def __init__(self, num_classes, start):
        self.num_classes = int(num_classes)
        self.start = start

    def transform(self, data):
        data = np.array(data)
        if len(data.shape) > 2:
            raise Exception(
                "The shape of data is %s, but the input shape of data is required as (-1,1) or (-1,) for one-hot transform." % (
                    data.shape))
        elif len(data.shape) == 2 and data.shape[1] is not 1:
            raise Exception(
                "The shape of data is %s, but the input shape of data is required as (-1,1) or (-1,) for one-hot transform." % (
                    data.shape))
        else:
            trans = data.reshape(-1)

        if self.num_classes == 1:
            trans = trans
        else:
            trans = data - self.start
            trans = np.eye(self.num_classes)[trans.astype(int)]
        return trans

    def inverse_transform(self, data):
        data = np.array(data)
        if len(data.shape) > 2 or data.shape[-1] is not self.num_classes:
            raise Exception(
                "The shape of data is %s, but the input shape of data is required as (-1,%s) or (-1,) for one-hot transform." % (
                data.shape, self.num_classes))
        trans = np.argmax(data, axis=1)
        trans = trans + self.start
        return trans

    def save(self, path):
        np.savez_compressed(path, num_classes=self.num_classes, start=self.start)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        return cls(data['num_classes'], data['start'])

class StaticFeatureScaler(object):
    def __init__(self):
        features_name = ['fes_'+str(i+1) for i in range(16)]
        features_type = {"fes_1": {"float": 1}, "fes_2": {"float": 1}, "fes_3": {"one_hot": 4, "start": 0},
                         "fes_4": {"one_hot": 5, "start": 1}, "fes_5": {"one_hot": 4, "start": 0},
                         "fes_6": {"binary": 1}, "fes_7": {"binary": 1}, "fes_8": {"binary": 1},
                         "fes_9": {"binary": 1}, "fes_10": {"binary": 1}, "fes_11": {"float": 1},
                         "fes_12": {"float": 1}, "fes_13": {"float": 1}, "fes_14": {"binary": 1},
                         "fes_15": {"one_hot": 3, "start": 0}, "fes_16": {"one_hot": 3, "start": 0}}
        idx2name = {}
        self.idx2type = {}
        for i in range(16):
            idx2name[i] = features_name[i]
            self.idx2type[i] = features_type[features_name[i]]
        self.dim = 0
        for i in self.idx2type:
            the_type = self.idx2type[i]
            for k in the_type:
                if k == "start":
                    continue
                self.dim += the_type[k]

    def transform(self, data):
        node_num, feas_num = data.shape
        features = []
        for i in range(feas_num):
            the_type = self.idx2type[i]
            if "float" in the_type:
                the_feature = np.reshape(data[:, i], (-1, 1))
                i_scaler = StandardScaler(the_feature.mean(), the_feature.std())
                the_feature = i_scaler.transform(the_feature)
            elif "one_hot" in the_type:
                the_feature = data[:, i]
                num_classes, start = the_type["one_hot"], the_type["start"]
                i_scaler = OnehotScaler(num_classes, start)
                the_feature = i_scaler.transform(the_feature)
            elif "binary" in the_type:
                the_feature = np.reshape(data[:, i], (-1, 1))
            else:
                raise Exception("Error type of the feature (%s). (float/binary/one_hot is acceptable)" % the_type)
            if isinstance(features, list):
                features = the_feature
            else:
                features = np.concatenate((features, the_feature), axis=1)
        print("the features shape is %s" % str(features.shape))
        if features.shape[1] is not self.dim:
            raise Exception("the shape of features should be %s" % self.dim)
        return features