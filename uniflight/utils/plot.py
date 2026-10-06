def interpolate_color(start_color, end_color, i, total):
    return [
        start_color[j] + (end_color[j] - start_color[j]) * i / (total - 1)
        for j in range(len(start_color))
    ]