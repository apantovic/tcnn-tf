# tcnn-tf
Temporal CNN implementation in TF2 for time series prediction

# Keras TCN

Inspiration for this came from [this]https://github.com/philipperemy/keras-tcn amazing repo, but one thing I've noticed is that it doesn't handle well modeling multiple time series, which is exactly what I needed for my project.
Original paper *Keras Temporal Convolutional Network*. [[paper](https://arxiv.org/abs/1803.01271)]

## API

The only dependency is tensorflow 2.x and ususal way to build model is to import model class and create the object instance by passing the time series dataframe.

```python
import pandas as pd
import tensorflow as tf
from tcnn import ModelTCNN

train = pd.read_csv('train.csv')
train.index = pd.date_range(start='2010-01-01',periods=146, freq='D')

model = ModelTCNN(y=train)            # use default values
model.fit()
model.predict(10)
...

###### you can access tf model object via model.model_
model.model_.summary()

model.save('save_path')
######
model.load('load_path')
```

### Arguments

```python
model = ModelTCNN(y=train, {'xreg': 0, 
                            'scale_data':True,
                            'num_filters':64,
                            'kernel_size':3,
                            'num_stacks':1,
                            'num_dilations':6,
                            'padding':'causal',
                            'dropout_rate':0.0,
                            'use_batch_norm':True,
                            'activation':'relu',
                            'optimizer':'adam',
                            'num_epoch':100,
                            'loss':'mse'
})
```

- `xreg`: pd.DataFrame. DataFrame with all external regressors (for prediction purposes, additional periods must be included)
- `scale_data`: bool, whether to use standard scaler to scale data to 0-1 range; default True
- `num_filters`: int, number of filters to use in Conv layers; default 64
- `kernel_size`: int, size of Conv kernel; default 3
- `num_stacks` : int, number of stacks in TCNN arhitecture; default 2
- `num_dilations`: int, number of dilations to use in each layer every subsequent 2 times greater than previous. eg: if num_dilations is 4 then dilations will be -> 2,4,8,16; default 6
- `padding`: str, padding type used in the Conv layer. 'causal' or 'same' (defaults to 'causal').
- `dropout_rate`: float, fraction of the input units to drop - default 0.0
- `use_batch_norm`: bool, whether to use batch normalization in the residual layers or not - default True
- `activation`: str. The activation used in the residual blocks o = activation(x + F(x)).
- `optimizer`: str/keras. Optimizer to use when training the model, either str or keras optimizer object - default 'adam'.
- `num_epoch`: int. Number of epochs to train the model on - default 100
- `loss`: str/keras. Loss function to use when training the model - deault 'mse'

### Input shape

Pandas DataFrame with time dimension in the rows and unique time-series in columns. Row index sould be date/date-time based.

### Output shape

Pandas DataFrame. By calling model.predict(num_steps), pandas dataframe will be generated with num_steps rows and same number of columns as the initial training dataset.

## Installation from the source

```bash
git clone git@github.com:apantovic/tcnn-tf.git && cd tcnn-tf
virtualenv -p python3 venv
source venv/bin/activate
pip install tensorflow==2.5.0
```

