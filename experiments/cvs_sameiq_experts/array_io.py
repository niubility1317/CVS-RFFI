"""Avoid the incompatible Torch2.1/NumPy2 C-array bridge on N607."""


def same_values(tensor, array):
    return tuple(tensor.shape) == tuple(array.shape) and tensor.tolist() == array.tolist()


def tensor_iq(array, device, torch):
    return torch.frombuffer(bytearray(array.tobytes(order='C')), dtype=torch.float32).reshape(array.shape).to(device)
