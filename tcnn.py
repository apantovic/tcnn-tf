import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

class ModelTCNN():
    """Creates a TCNN model object.
        Params: y - pandas DataFrame/Series with all time series required for modeling purposes.
            **kwargs:
                xreg - pandas DataFrame with all external regressors (for prediction purposes, additional periods must be included)
                scale_data - bool, use standard scaler to scale data to 0-1 range; default True
                num_filters - int, number of filters to use in Conv layers; default 64
                kernel_size - int, size of Conv kernel; default 3
                num_stacks - int, number of stacks in TCNN arhitecture; default 2
                num_dilations - int, number of dilations to use in each layer; default 6
                padding - str, 'same' or 'causal'
                dropout_rate - float; default 0.0
                use_batch_norm - bool; default True
                activation - str; default 'relu'
                optimizer - str/keras optimizer object; default 'adam'
                num_epoch - int, number of epochs for training; default 100
                loss - str/keras optimizer object, used for loss evaluation in training; default 'mse'
        """

    def __init__(self, y, **kwargs):
        if isinstance(y, pd.Series):
            self.id = y.name
            self.out = 1
        else:
            self.id = y.columns
            self.out = len(y.columns)

        self.xreg = kwargs.get('xreg', pd.DataFrame())
        if self.xreg.shape[0]>0:
            self.y = y.join(self.xreg).replace([np.inf, -np.inf], 0).fillna(0)
        else:
            self.y = y.replace([np.inf, -np.inf], 0).fillna(0)

        self.y_norm = self.y.copy()
        self.scale_data = kwargs.get('scale_data', True)
        if self.scale_data:
            self.scaler = StandardScaler().fit(y)
            self.y_norm[self.id] = self.scaler.transform(y)

        self.num_filters = kwargs.get('num_filters',64)
        self.kernel_size = kwargs.get('kernel_size',3)
        self.num_stacks = kwargs.get('num_stacks',1)
        self.num_dilations = kwargs.get('num_dilations',6)
        self.padding = 'causal' if kwargs.get('padding','causal') not in ['causal','same'] else kwargs.get('padding','causal')
        self.dropout_rate = kwargs.get('dropout_rate',0.0)
        self.use_batch_norm = kwargs.get('use_batch_norm',True)
        self.activation = kwargs.get('activation','relu')
        self.optimizer = kwargs.get('optimizer','adam')
        self.num_epoch = kwargs.get('num_epoch',100)
        self.loss = kwargs.get('loss','mse')
        self.model = None

    def _define_model_object(self):
        """
        Build TCNN arhitecture
        """
        if self.model is not None:
            return self.model
        else: 
            shape_t, shape_f = len(self.y.index)//2, self.y_norm.shape[1]

            inputs = tf.keras.layers.Input(shape=(shape_t, shape_f))
            x = inputs
            x = tf.keras.layers.Conv1D(filters=self.num_filters, kernel_size=1, padding='same', name='initial_conv0')(x)
            for s in range(self.num_stacks):
                for i in range(self.num_dilations):
                    dilation_rate = 2 ** i
                    prev_x = x
                    for i in range(2):        ## 2 blocks
                        x = tf.keras.layers.Conv1D(filters=self.num_filters, kernel_size=self.kernel_size, dilation_rate=dilation_rate, padding=self.padding)(x)
                        if self.use_batch_norm:
                            x = tf.keras.layers.BatchNormalization(axis=-1)(x)
                        x = tf.keras.layers.Activation(self.activation)(x)
                        x = tf.keras.layers.SpatialDropout1D(self.dropout_rate)(x)

                    x = tf.keras.layers.Activation(self.activation)(prev_x + x)
            x = tf.keras.layers.Lambda(lambda l: l[:, 1, :])(x)
            out = tf.keras.layers.Dense(self.out)(x)
            model = tf.keras.models.Model(inputs=inputs, outputs=out)
            return model

    def fit(self):
        """
        Fit model to the provided data
        """
        model = self._define_model_object()

        generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(self.y_norm.values, self.y_norm[self.id].values, length=model.input.get_shape()[1], batch_size=1)  
        model.compile(optimizer=self.optimizer, loss=self.loss)

        model.fit(generator, steps_per_epoch=1, epochs=self.num_epoch, shuffle=False, verbose=0)
        self.model = model

        return self.model

    def save(self, path):
        """
        Save model object - provide full path, for example: '~/usr/models/mymodel.h5'
        """
        self.model.save(path)
    
    def load(self, path):
        """
        Load model object - provide full path, for example: '~/usr/models/mymodel.h5'
        """
        self.model = tf.keras.models.load_model(path)

    def predict(self, h):
        """
        Generate predictions for h steps ahead
        Params: h - number of steps to forecast
        If xreg data was used during the training, it must be included for next h periods in the future
        """
        periods=pd.date_range(start=max(self.y.index), periods=h+1, freq=self.y.index.freq)[1:]
        pred = pd.DataFrame(data=[], columns=self.y.columns, index=periods)
        if self.xreg.shape[0]>0:
            pred[self.xreg.columns] = self.xreg[self.xreg.index.isin(pred.index)].values
        tmp_pred = self.y_norm[-self.model.input.get_shape()[1]:]
        for i in range(h):
            inp = np.asarray(tmp_pred[-self.model.input.get_shape()[1]:].values.reshape((1, self.model.input.get_shape()[1], self.y_norm.shape[1]))).astype(np.float32)
            p = self.model.predict(inp, verbose=0)
            pred.loc[pred.index[i], self.id] = p
            tmp_pred = pd.concat([tmp_pred, pred.iloc[[i]]])

        if self.scale_data:
            res = self.scaler.inverse_transform(pred)
        else:
            res = pred.values
        res = pd.DataFrame(data=np.where(res<0, 0, res.astype(int)), columns=pred.columns, index=pred.index)
        return res
