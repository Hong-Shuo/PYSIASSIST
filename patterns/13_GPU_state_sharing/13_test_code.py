def inference(self, num_iter=10):
    # Essentially softmax
    prediction = exp_and_normalize(-self.unary)
    for i in range(num_iter):
        tmp1 = -self.unary

        # Main computational effort
        tmp2 = self.kernel.compute(prediction) 

        tmp1 = tmp1 - tmp2
        prediction = exp_and_normalize(tmp1)
inference()