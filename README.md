# TCN-finance
Conditional time series forecasting with convolutional neural networks.
Inspired from:
- Conditional time series forecasting with convolutional neural networks by Anastasia Borovykh, Sander Bohte and Cornelis W. Oosterlee (2018)
- Probabilistic Forecasting with Temporal Convolutional Neural Network by Yitian Chena, Yanfei Kangb, Yixiong Chenc, Zizhuo Wangd (2020)

## Why Temporal Convolutional Network?

- TCNs exhibit longer memory than recurrent architectures with the same capacity.
- Constantly performs better than LSTM/GRU architectures on a vast range of tasks (Seq. MNIST, Adding Problem, Copy Memory, Word-level PTB...).
- Parallelism, flexible receptive field size, stable gradients, low memory requirements for training, variable length inputs...

<p align="center">
  <img src="misc/Dilated_Conv.png">
  <b>Visualization of a stack of dilated causal convolutional layers (Wavenet, 2016)</b><br><br>
</p>

### Arguments
- `dataset`: pandas.Dataframe. The networks are trained to predict information from this dataset
- `dataset2`: pandas.Dataframe. This dataset is used as additional information to help predict values from the first dataset
- `device` : cpu or gpu
- `epochs`: Integer. Number of training epoch
- `kernel_size`: Integer. Size of the kernel used in convolutions
- `checkpoint`: Integer. Number of epoch after which networks are saved.
- `receptive_field`: Integer. Number of previous dates information used to forecast the future
- `n_layers_encod`: Integer. Number of layer of the encoder
- `n_outputs`: list of Integer. Number of channel of convolutions of the encoder
- `dilation`: Integer. Dilatation parameter of the convolution
- `stride`: Integer. Stride parameter of convolution
- `n_inputs`: Integer. Number of asset 
- `n_channel_decod`: Integer. Number of channel of the decoder
- `future_size`: Integer. Number of step forecasted in the future
- `learning_rate`: Float. Learning rate used for parameter optimization
- `sep_train_test`: Integer. Index of separation between training set and test set
- `outf`: String. Folder where networks are saved.
- `resume`: Boolean. True if networks have previously been trained and one wishes to resume training


