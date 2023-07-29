from torch.utils.data import DataLoader, Dataset, TensorDataset

class ScaledDataset(TensorDataset):
    '''
    A custom subclass of torch's TensorDataset that makes the dataset appear larger
    than it is. Intended for use with REINFORCE, where large batch sizes mean
    epochs are too fast.

    input:
        xs: list of torch.tensors as input to the model
        ys: list of torch.tensors as output of the model
        scaling_factor: number of times to repeat the data
    '''

    def __init__(self, xs, ys, scaling_factor=1):
        super().__init__(xs, ys)

        self.scaling_factor = scaling_factor

    def __len__(self):
        return len(self.tensors[0]) * self.scaling_factor

    def __getitem__(self, k):
        k = k % len(self.tensors[0])
        return tuple(tensor[k] for tensor in self.tensors)  # , torch.zeros(1)