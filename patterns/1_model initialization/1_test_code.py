beta = 0.5 #The interpolation parameter
params1 = model1.named_parameters()
params2 = model2.named_parameters()

dict_params2 = dict(params2)

for name1, param1 in params1:
    if name1 in dict_params2:
        dict_params2[name1].data.copy_(beta*param1.data + (1-beta)*dict_params2[name1].data)

model.load_state_dict(dict_params2)