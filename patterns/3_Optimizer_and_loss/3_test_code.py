for e in range(ep):
    for i in range(len(trainX)):
        input_ = torch.tensor(trainX[i]).reshape(1, 1, 1)
        targ_ = torch.tensor(trainY[i])

        out_ = model(input_)

        loss = loss_fn(out_, targ_)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()