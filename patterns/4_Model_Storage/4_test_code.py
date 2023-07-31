for epoch in range(1, epochs + 1):
    for i, data in enumerate(trainloader, 0):

        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels.flatten())
        loss.backward()
        optimizer.step()

        # get statistics every 2000 mini-batchss
        running_loss += loss.item()
        if i % 2000 == 1999:

            # log the running training loss
            training_loss.append(running_loss / 2000)

            # log the running validation loss
            with torch.no_grad():
                running_val_loss = 0.0
                for i_val, data_val in enumerate(valloader, 0):
                    inputs_val, labels_val = data_val
                    outputs_val = net(inputs_val)
                    loss_val = criterion(outputs_val, labels_val.flatten()).item()
                    running_val_loss += loss_val
            validation_loss.append(running_val_loss / len(valloader))

            print('[%d, %5d] train_loss: %.3f | val_loss: %.3f' %
                  (epoch, i + 1, running_loss / 2000, running_val_loss / len(valloader)))

            # save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_loss': running_loss / 2000,
                'validation_loss': running_val_loss / len(valloader)
            }, PATH + '/epoch{}_model.pt'.format(epoch))

            running_loss = 0.0