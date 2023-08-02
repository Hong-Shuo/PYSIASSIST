# Compute loss params
x = torch.hstack((input_mags,params))
out1 = network(x,fc_weights)
out2 = network(out,fc_weights)
loss_params = MyLoss(out2)

# Set the values in the other params
with torch.no_grad():
  params2.copy_(out1)
  params3.copy_(out2)

# Compute the other losses
out = network(params2,fc_weights)
loss_params2 = MyLoss(out)

loss_params3 = MyLoss(params3)

# Since each loss has its own params and are independents, we can get
# all the gradients in one go:
loss = loss_params1 + loss_params2 + loss_params3
loss.backward()