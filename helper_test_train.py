import torch


def eval_model(model: torch.nn.Module,
               test_dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               device: str):
    '''a function for evaluating any model'''

    model.train(mode=False)  # same as model.eval()
    model.to(device)

    batch_size = test_dataloader.batch_size
    # going through the batches in test_dataloader
    correct_pred = 0
    total_loss = 0
    for X_test, y_test in test_dataloader:
        # X_test = batch of BATCH_SIZE images
        # y_test = corresponding labels
        X_test, y_test = X_test.to(device), y_test.to(device)

        # accuracy
        y_pred = model(X_test)
        correct_pred += torch.eq(y_pred.argmax(dim=1), y_test).sum().item()

        # loss
        loss = loss_fn(y_pred, y_test)
        total_loss += loss

    accuracy = (correct_pred/((len(test_dataloader))*(batch_size)))*100
    total_loss = total_loss / (len(test_dataloader))

    # returning a dictionary
    return {"model_name": model.__class__.__name__,
            "model_acc": accuracy,
            "model_loss": total_loss}


def train_model(model: torch.nn.Module,
                loss_fn: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                train_dataloader: torch.utils.data.DataLoader,
                epochs: int,
                device: str):
    '''this function trains a model with the specified number of epochs'''

    import time
    start_time = time.time()

    model.to(device)
    for epoch in range(epochs):
        print("training in epoch: ", epoch)

        batch_count = 0
        for X, y in train_dataloader:
            X, y = X.to(device), y.to(device)
            batch_count += 1
            # X = a batch of 32 images
            # y = the corresponding labels for the images
            model.train()  # setting our model to train mode. whether we are testing or training is needed to be known by some layers but not all
            # 1 forward pass
            y_pred = model(X)
            print(y_pred.shape)
            print(y.shape)
            # 2 loss for the batch
            loss = loss_fn(y_pred, y)
            # 3 optimizer zero grad
            optimizer.zero_grad()
            # 4 backward()
            loss.backward()  # the backward function doesn't work without a forward pass prior to its call. hence we had to calulate loss. i guess the forward pass creates the necessary computational graph for gradient calculation
            # 5 update weights
            optimizer.step()

            if (batch_count % 400 == 0):
                print("finished training with batch ", batch_count)

        print("finished training of epoch ", epoch)
        if (epoch == (epochs-1)):
            print("finished training")

    end_time = time.time()
    print('total time consumed: ', start_time - end_time)


def train_model_w_loss_tracking():
    '''
    this function trains a model. and also:
    keeps track of total train loss and test loss in each epoch.
    '''
    pass
