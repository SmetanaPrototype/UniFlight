def get_y(x, x_array, y_array):
    if len(x_array) == 0 or len(y_array) == 0:
        return 0
    return np.interp(x, x_array, y_array)

def calculate_multi(*args):
    return [math.prod(items) for items in zip(*args)]


def calculate_sum(*args):
    if not args:
        return []

    vectors = [list(v) for v in args]
    base_idx = max(range(len(vectors)), key=lambda i: len(vectors[i]))
    result = vectors[base_idx].copy()

    for i, vec in enumerate(vectors):
        if i == base_idx:
            continue
        for j, value in enumerate(vec):
            result[j] += value

    return result

def normalize_list(dataset, target):
    """Приводит сумму dataset.masses к target пропорционально.
    Если текущая сумма 0 — ничего не делает."""
    current = sum(dataset)
    if current <= 0.0:
        return
    scale = target / current
    for i in range(len(dataset)):
        dataset[i] *= scale