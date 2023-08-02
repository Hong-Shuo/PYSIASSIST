class BatchNorm2d_new(nn.BatchNorm2d):
    def forward(self, input):
        self._check_input_dim(input)

        statistics_tbn = {'mean': input.mean(dim=[0, 2, 3]).detach().clone(),
                          'var': input.var(dim=[0, 2, 3]).detach().clone()}
        statistics_tbn['inv_std'] = torch.reciprocal(torch.sqrt((statistics_tbn['var'] + self.eps)))

        output = input - _unsqueeze_(statistics_tbn['mean'])
        output = output * _unsqueeze_(statistics_tbn['inv_std'])
        output = output * _unsqueeze_(self.weight)
        output = output + _unsqueeze_(self.bias)
        return output


def _unsqueeze_(tensor):
    return tensor.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)


# This function change origin BN to New BN in our network, reclusively.
def convert_bn(module):
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module_output = BatchNorm2d_new(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine
        )
        module_output.weight.data = module.weight.data.clone()
        module_output.bias.data = module.bias.data.clone()
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var

    for name, child in module.named_children():
        module_output.add_module(
            name, convert_bn(child),
        )
    del module
    return module_output


model = BatchNorm2d_new()