import time

import numpy.random as npr
import xgboost as xgb
from keras import backend as K
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import *
from keras.models import Model
from keras.optimizers import SGD
from sklearn.cluster import Birch
from sklearn.decomposition import PCA


class AutoencoderModel(object):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.kernel_size = 2
        self.latent_dim = 128
        self.batch_size = 64
        self.fix_conv = False
        self.history = np.zeros((self.latent_dim, self.batch_size))
        self.encoder, self.decoder, self.autoencoder = self.__build_model(input_shape)
        self.online_counter = 0
        self.online_training_running = False

    def __build_model(self, input_shape):
        # Encoder/Decoder number of CNN layers and filters per layer
        layer_filters = [16, 32, 64]

        # Build the Autoencoder Model
        # First build the Encoder Model
        inputs = Input(shape=input_shape, name='encoder_input')
        x = inputs
        # Stack of Conv2D blocks
        # Notes:
        # 1) Use Batch Normalization before ReLU on deep networks
        # 2) Use MaxPooling2D as alternative to strides>1
        # - faster but not as good as strides>1
        for filters in layer_filters:
            x = Conv2D(filters=filters,
                       kernel_size=self.kernel_size,
                       strides=2,
                       activation='relu',
                       padding='same')(x)
            x = BatchNormalization()(x)

        # Shape info needed to build Decoder Model
        shape = K.int_shape(x)

        # Generate the latent vector
        x = Flatten()(x)
        latent = Dense(self.latent_dim, name='latent_vector')(x)

        # Instantiate Encoder Model
        encoder = Model(inputs, latent, name='encoder')
        encoder.summary()

        # Build the Decoder Model
        latent_inputs = Input(shape=(self.latent_dim,), name='decoder_input')
        x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
        x = Reshape((shape[1], shape[2], shape[3]))(x)

        # Stack of Transposed Conv2D blocks
        # Notes:
        # 1) Use Batch Normalization before ReLU on deep networks
        # 2) Use UpSampling2D as alternative to strides>1
        # - faster but not as good as strides>1
        for filters in layer_filters[::-1]:
            x = Conv2DTranspose(filters=filters,
                                kernel_size=self.kernel_size,
                                strides=2,
                                activation='relu',
                                padding='same')(x)
            x = BatchNormalization()(x)

        x = Conv2DTranspose(filters=input_shape[-1],
                            kernel_size=self.kernel_size,
                            padding='same')(x)

        outputs = Activation('sigmoid', name='decoder_output')(x)

        # Instantiate Decoder Model
        decoder = Model(latent_inputs, outputs, name='decoder')
        decoder.summary()

        # Autoencoder = Encoder + Decoder
        # Instantiate Autoencoder Model
        autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
        autoencoder.summary()

        autoencoder.compile(loss='mse', optimizer='adam')

        return encoder, decoder, autoencoder

    def load_model(self, weights_file):
        self.autoencoder.load_weights(weights_file)
        if self.fix_conv:
            self.__fix_autoenc_layers(self.encoder)
            self.__fix_autoenc_layers(self.decoder)
            self.autoencoder.compile(loss='mse',
                                     optimizer=SGD(lr=1e-4, momentum=0.9),
                                     metrics=['accuracy'])

    def __fix_autoenc_layers(self, model):
        for layer in model.layers:
            if "conv2d" in layer.name:
                layer.trainable = False

            if "batch_normalization" in layer.name:
                layer.trainable = False


    def save_model(self, weights_file):
        self.autoencoder.save_weights(weights_file)
        return weights_file

    def save_model_snapshot(self, weights_file_format):
        file_name = weights_file_format % int(time.time())
        return self.save_model(file_name)

    def fit(self, x_train, x_test, epochs=25, tensorboard=False):
        callbacks = []
        callbacks.append(EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto'))
        if tensorboard:
            callbacks.append(TensorBoard(log_dir='/tmp/autoencoder'))

        hist = self.autoencoder.fit(x_train, x_train,
                                    epochs=epochs,
                                    batch_size=self.batch_size,
                                    validation_data=(x_test, x_test),
                                    callbacks=callbacks,
                                    shuffle=True)
        return hist

    def online_fit(self, x):
        if self.online_training_running:
            return False

        if not self.online_counter:
            self.online_train = np.zeros(((self.batch_size, ) + self.input_shape))
            self.online_test = np.zeros(((int(self.batch_size * 0.1), ) + self.input_shape))

            all_idcs = np.arange(self.online_train.shape[0] + self.online_test.shape[0])
            self.online_test_plan = npr.choice(all_idcs, size=self.online_test.shape[0])

        if self.online_counter in self.online_test_plan:
            self.online_test = np.roll(self.online_test, 1, axis=0)
            self.online_test[0] = x
        else:
            self.online_train = np.roll(self.online_train, 1, axis=0)
            self.online_train[0] = x

        self.online_counter += 1

        if self.online_counter == self.batch_size:
            self.online_training_running = True
            hist = self.fit(self.online_train, self.online_test, epochs=10)
            self.online_training_running = False
            self.online_counter = 0
            return hist.history['loss'], hist.history['val_loss']
        else:
            return False


    def encode(self, x):
        y = self.encoder.predict(x, batch_size=1)
        self.__append_to_history(y)
        return y

    def __append_to_history(self, y):
        self.history = np.roll(self.history, 1, axis=1)
        self.history[:, 0] = y
        return y


class DecissionTreeObjectClassifierModel(object):
    def __init__(self):
        self.current_model = False
        self.learned_classes = list()
        self.max_classes = 10

        self.init_params = {'objective':'multi:softprob',
                            'num_class': self.max_classes}
        self.update_params = self.init_params.copy()
        self.update_params.update({
                            'process_type': 'update',
                            'updater'     : 'refresh',
                            'refresh_leaf': False})

    def load_model(self, model_file):
        bst = xgb.Booster({})
        bst.load_model('model.bin')

        self.current_model = bst.load_model(model_file)

    def online_fit(self, X, class_name):
        train_set = xgb.DMatrix(X, label=self.__class_name_to_label_vec(class_name))

        try:
            if self.current_model:
                self.current_model = xgb.train(self.update_params, train_set, xgb_model=self.current_model)
            else:
                self.current_model = xgb.train(self.init_params, train_set)
        except Exception as e:
            print(e)


    def __class_name_to_label_vec(self, class_name):
        if class_name not in self.learned_classes:
            self.learned_classes.append(class_name)
        idx = self.learned_classes.index(class_name)

        label = np.zeros((1, self.max_classes))
        label[0, idx] = 1.
        return label

    def predict_class(self, X):
        if not self.current_model:
            return False, False

        test_set = xgb.DMatrix(X)
        class_prop = self.current_model.predict(test_set)
        class_idx = np.argmax(class_prop)

        return self.learned_classes[class_idx], class_prop


class ClusteringObjectClassifierModel(object):
    def __init__(self):
        self.learned_classes = dict()
        self.max_classes = 10
        self.estimator = Birch(n_clusters=None, threshold=10.0)

    def online_fit(self, X, class_name):
        self.estimator.partial_fit(X)

        cluster_id = np.asscalar(self.estimator.labels_)
        if cluster_id not in self.learned_classes:
            print("Assigning cluster id %d to class %s" % (cluster_id, class_name))
            self.learned_classes[cluster_id] = class_name

        return self.__pca_on_cluster_centers(self.estimator.subcluster_centers_)

    def __pca_on_cluster_centers(self, cluster_centers):
        pca= PCA(n_components=2)
        coords = np.atleast_2d(pca.fit_transform(cluster_centers))
        if len(coords) < 2:
            return np.zeros(1), np.zeros(1)

        return coords[:, 0], coords[:, 1]

    def predict_class(self, X):
        if not hasattr(self.estimator, "root_"):
            return False, False

        cluster_id = np.asscalar(self.estimator.predict(X))
        if cluster_id not in self.learned_classes:
            return False, False

        return self.learned_classes[cluster_id], cluster_id


