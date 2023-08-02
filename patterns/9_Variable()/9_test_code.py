HP1 = {
    # 'NUM_HIDDEN_NODES'    : 10 ,
    # 'NUM_EXAMPLES'        : 5 ,
    'TRAIN_SPLIT': .8,
    'MINI_BATCH_SIZE': 500,  # 10*int((2/3)*(u_array_train.shape[1]+1)/4) ,
    'NUM_EPOCHS': 2000,
    'LEARNING_RATE': 5e-2,
    'LEARNING_RATE_DECAY': 500,
    'WEIGHT_DECAY': 5e-4,
    'NUM_MOMENTUM': 0.9,
    'NUM_PATIENCE': 50,
    'SEED': 2018
}


def to_np(x):
    return x.data.cpu().numpy()


def to_var(x):
    # if torch.cuda.is_available():
    # x = x.cuda(async=async)
    return Variable(x)


# creat FCNN network and define all other parameters

# build neural network
class NetworkFCNN(nn.Module):
    def __init__(self, D_in, D_out):
        super().__init__()

        # Inputs to hidden layer linear transformation
        self.lin1 = nn.Linear(D_in, 100)
        self.lin2 = nn.Linear(100, 100)
        self.lin3 = nn.Linear(100, 100)
        # Output layer,
        self.output = nn.Linear(100, D_out)

        # Define sigmoid activation and softmax output
        # self.tanh = F.tanh()

    def forward(self, x):  # this is where the data flows in the network, respecting
        # sequence of layers in forward method is very important.
        # Pass the input tensor through each of our operations

        x = self.lin1(x)
        x = F.tanh(x)

        x = self.lin2(x)
        x = F.tanh(x)

        x = self.lin3(x)
        x = F.tanh(x)

        x = self.output(x)
        y = F.tanh(x)

        return y


criterion1 = torch.nn.MSELoss(size_average=False)
optimizer1 = torch.optim.Adam(model.parameters(),
                              lr=HP1['LEARNING_RATE'])
# momentum=HP['NUM_MOMENTUM'],
# weight_decay=HP['WEIGHT_DECAY'],
# )
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer1, lr_lambda=[epoch_smooth_decay])  # select the LR schedule

modelFCNN = NetworkFCNN(R_k_plus_1_train.shape[1], 1)

# weight intizalization routine (Xavier if batch normalisation is there int he netowrk, or else uniform)
glorot_weight_zero_bias(model=modelFCNN)

train_set_ref = torch.utils.data.TensorDataset(torch.FloatTensor(R_k_train), torch.FloatTensor(R_k_plus_1_train))
train_loader_ref = torch.utils.data.DataLoader(train_set_ref, batch_size=HP1['MINI_BATCH_SIZE'],
                                               shuffle=False, pin_memory=True, num_workers=0)

modelFCNN.train()
model.eval()

ref_train_losses = []
ref_valid_losses = []
ref_valid_score = []
ref_epochs = []

# start = time.time()

# epoch_iter = tqdm(range(1, HP['NUM_EPOCHS'] + 1))
epoch_iter = range(1, HP1['NUM_EPOCHS'] + 1)

# for epoch in range(1, HP['NUM_EPOCHS'] + 1):

for epoch in epoch_iter:
    # epoch_iter.set_description('Epoch')

    ref_epochs.append(epoch)

    # training over all batch data
    batch_idx, tloss_avg_ref, vloss_avg_ref = 0, 0, 0
    for batch_idx, (data_rk, data_rk_plus1) in enumerate(train_loader_ref):
        # print (batch_idx)
        y_pred_ref = modelFCNN(to_var(data_rk_plus1))  # predict y based on x

        # print("y pred ref is")
        # print(y_pred_ref)

        # concatenate the tensor with the state vector along columns
        # sysnn_input = np.concatenate( (to_np(data_rk),to_np(y_pred_ref) ),axis=1)
        sysnn_input = torch.cat((data_rk, y_pred_ref), dim=1)

        # print ("print concantenatd input tensor is")
        # print(sysnn_input)

        # now feed in this input to the trained sys NN model
        output_sys = model(to_var(sysnn_input))

        # print ("output of sys NN is ")
        # print(output_sys)

        # print ("data_rk_plus1 is ")
        # print(data_rk_plus1)

        # compute the loss
        loss_FCNN = criterion(output_sys, to_var(data_rk_plus1))  # compute loss
        # print ("loss is")
        # print(loss_FCNN)

        optimizer1.zero_grad()  # clear gradients
        loss_FCNN.backward()  # compute gradients
        optimizer1.step()  # apply gradients

        tloss_avg_ref += loss_FCNN.item()

    tloss_avg_ref /= batch_idx + 1
    ref_train_losses.append(tloss_avg_ref)

    print(" Epoch : %s , Train loss: %s " % (epoch, tloss_avg_ref))