import h5py
import torch.utils.data


def load_data(path):
    """Reads data from H5 file, and returns training & validation sets.
    """
    with h5py.File(path, "r") as f:
        x_train = f["Training"]["Inputs"].__array__()
        y_train = f["Training"]["Labels"].__array__()
        x_val = f["Validation"]["Inputs"].__array__()
        y_val = f["Validation"]["Labels"].__array__()

    x_train = torch.from_numpy(x_train).type(torch.float32)
    y_train = torch.from_numpy(y_train).type(torch.float32)
    x_val = torch.from_numpy(x_val).type(torch.float32)
    y_val = torch.from_numpy(y_val).type(torch.float32)

    return (x_train, y_train), (x_val, y_val)


def get_data_loader(inputs, labels, batch_size=25):
    """Fetches a PyTorch DataLoader, which gives a convenient sampling method.
    """
    dataset = torch.utils.data.TensorDataset(inputs, labels)
    return torch.utils.data.DataLoader(
        dataset, num_workers=4, shuffle=True, batch_size=batch_size
    )
