"""Fixed source-only 30-view schedule, with matched total training exposure."""


def diverse_epochs(epoch):
    if not isinstance(epoch, int) or not 1 <= epoch <= 20:
        raise ValueError('Training epoch must be in 1..20')
    first = 1 + 3 * ((epoch - 1) % 10)
    return (first, first + 1, first + 2)
