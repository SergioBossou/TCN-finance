import pandas as pd
import torch
import torch.nn as nn
import data_utils as dt
import net_utils as nt


class Train:

    def __init__(self,dataset, dataset2, checkpoint, receptive_field,
          future_size, cuda, epochs, n_layers_encod,
          n_inputs, n_outputs, dilation, stride, n_channel_decod, kernel_size,
          lr, outf, resume, sep_train_test):

        self.device = torch.device("cuda:0" if cuda else "cpu")
        self.dataset = dataset
        self.dataset2 = dataset2
        self.n_asset_y = dataset.shape[1]
        self.n_asset_X = dataset2.shape[1]
        self.dilation = dilation
        self.stride = stride
        self.n_channel_decod = n_channel_decod
        self.kernel_size = kernel_size
        self.lr = lr
        self.outf = outf
        self.resume = resume
        self.sep_train_test = sep_train_test
        self.checkpoint = checkpoint
        self.receptive_field = receptive_field
        self.future_size = future_size

        self.epochs = epochs
        self.n_layers_encod = n_layers_encod
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.init_nets()

        if self.resume:
            self.resumed()

        self.data_in_format()

    def init_nets(self,):
        """ Initialisation of neural networks """
        print("Creating new networks")
        self.encoder = nt.Encoder(n_layers=self.n_layers_encod, n_inputs=self.n_inputs, n_outputs=self.n_outputs,
                          dilation=self.dilation, receptive_field=self.receptive_field, stride=self.stride, kernel_size=self.kernel_size,
                          device=self.device).to(self.device)
        self.decoders = [nt.Decoder(n_channel=self.n_channel_decod, n_channel_X=self.n_asset_X, future_size=i, device=self.device).to(self.device)
                    for i in range(1, 1 + self.future_size)]

        print("Setting optimizers")
        # Optimizers are used by pytorch to update the weights after gradient is computed
        self.optimizerE = torch.optim.Adam(self.encoder.parameters(), lr=self.lr, betas=(0.5, 0.9))
        self.optimizerD = [torch.optim.Adam(self.decoders[i].parameters(), lr=self.lr, betas=(0.5, 0.9)) for i in range(self.future_size)]

    def resumed(self):
        """ Loads saved networks """
        print("Previous networks loaded")
        saved_data = torch.load(self.outf + f'/deeptcn.pt')
        self.encoder.load_state_dict(saved_data['encoder'])
        self.optimizerE.load_state_dict(saved_data['optimizerE_state_dict'])
        for i in range(self.future_size):
            self.decoders[i].load_state_dict(saved_data[f'decoder_{i}'])
            self.optimizerD[i].load_state_dict(saved_data[f'optimizerD_state_dict_{i}'])

    def data_in_format(self):
        """ Changes provided data format """
        # We separate training set and test set
        self.train_set_y = self.dataset.iloc[:self.sep_train_test, :]
        self.train_set_X = self.dataset2.iloc[:self.sep_train_test, :]
        self.test_set_y = self.dataset.iloc[self.sep_train_test:, :]
        self.test_set_X = self.dataset2.iloc[self.sep_train_test:, :]

        # We reshape the data so they can be used by the networks
        self.decod_y_train, self.encod_y_train = dt.data_in_shape(self.train_set_y, self.receptive_field, self.n_asset_y, self.future_size,
                                                               self.future_size, self.device)
        self.decod_X_train, self.encod_X_train = dt.data_in_shape(self.train_set_X, self.receptive_field, self.n_asset_X, self.future_size,
                                                               self.future_size, self.device)

        # We reshape the data so they can be used by the networks
        self.decod_y_test, self.encod_y_test = dt.data_in_shape(self.test_set_y, self.receptive_field, self.n_asset_y, self.future_size,
                                                             self.future_size, self.device)
        self.decod_X_test, self.encod_X_test = dt.data_in_shape(self.test_set_X, self.receptive_field, self.n_asset_X, self.future_size,
                                                             self.future_size, self.device)

    def train(self):
        """ Neural nets training """
        # Throughout training we save the MSE loss on train and test set
        self.Losses = []
        self.Losses_test = []

        for epoch in range(self.epochs):
            print(f'Epoch[{epoch}/{self.epochs}]')

            # We set the gradient of the networks to zero before computing gradient
            self.encoder.zero_grad()
            for i in range(self.future_size):
                self.decoders[i].zero_grad()

            # Training step
            encod_output = self.encoder(y=self.encod_y_train, X=self.encod_X_train)
            decod_outputs = [self.decoders[i](info=self.decod_X_train[:, :, :i + 1], memory=encod_output) for i in
                             range(self.future_size)]
            # decod_output = concatenate_time(decod_outputs, self.device)
            decod_output = torch.cat(decod_outputs, 2).to( self.device)
            loss = nn.MSELoss()

            error = loss(decod_output, self.decod_y_train)
            print(f'Loss_train: {error.item()}')
            error.backward()
            self.Losses.append(error.item())

            # Update the parameters
            for i in range(self.future_size):
                self.optimizerD[i].step()
            self.optimizerE.step()

            # Evaluation on test set
            encod_output_test = self.encoder(y=self.encod_y_test, X=self.encod_X_test)
            decod_outputs_test = [self.decoders[i](info=self.decod_X_test[:, :, :i + 1], memory=encod_output_test) for i in
                                  range(self.future_size)]
            # decod_output_test = concatenate_time(decod_outputs_test, self.device)
            decod_output_test = torch.cat(decod_outputs_test, 2).to(self.device)

            loss_test = nn.MSELoss()

            error_test = loss_test(decod_output_test, self.decod_y_test)
            print(f'Loss_test : {error_test.item()}')
            self.Losses_test.append(error_test.item())

            # At chexkpoint we save the networks and visualize the current results
            if epoch % self.checkpoint == 0:
                self.save_checkpoint()

        self.plot_figure()

    def save_checkpoint(self) :
        """ Save networks at checkpoint """
        torch.save({**{'encoder': self.encoder.state_dict(),
                       'optimizerE_state_dict': self.optimizerE.state_dict()},
                    **{f'decoder_{i}':self.decoders[i].state_dict() for i in range(self.future_size)},
                    **{f'optimizerD_state_dict_{i}': self.optimizerD[i].state_dict() for i in range(self.future_size)}},
                   self.outf + f'/deeptcn.pt')

    def plot_figure(self ):
        """ Plot current state """
        encod_output = self.encoder(X=self.encod_X_train, y=self.encod_y_train)
        decod_outputs = [self.decoders[i](info=self.decod_X_train[:, :, :i + 1], memory=encod_output) for i in
                               range(self.future_size)]

        decod_output = torch.cat(decod_outputs, 2).to(self.device)
        encod_output_test = self.encoder(X=self.encod_X_test, y=self.encod_y_test)
        decod_outputs_test = [self.decoders[i](info=self.decod_X_test[:, :, :i + 1], memory=encod_output_test) for i in range(self.future_size)]
        decod_output_test = torch.cat(decod_outputs_test, 2).to(self.device)


        asset_i1 = dt.concatin(
            [pd.DataFrame(decod_output.cpu().detach().numpy()[:, -self.future_size:].flatten()),
             pd.DataFrame(self.decod_y_train.cpu().detach().numpy()[:, 0, -self.future_size:].flatten())],
            colnames=['Network', 'True'])
        asset_i1.hist(bins = 100)

        asset_i2 = dt.concatin([pd.DataFrame(decod_output_test.cpu().detach().numpy()[:, -self.future_size:].flatten()),
             pd.DataFrame(self.decod_y_test.cpu().detach().numpy()[:,0, -self.future_size:].flatten())],
            colnames=['Network', 'True'])
        asset_i2.hist(bins = 100)

        asset_i1.plot()
        asset_i2.plot()

        pd.DataFrame(self.Losses, columns=['Loss function on train set']).plot()
        pd.DataFrame(self.Losses_test, columns=['Loss function on test set']).plot()




