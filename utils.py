def calculate_speed(prev_y, curr_y, fps, pixels_per_meter=8):
    pixel_distance = abs(curr_y - prev_y)
    meters = pixel_distance / pixels_per_meter
    time = 1 / fps
    speed_kmh = (meters / time) * 3.6
    return round(speed_kmh, 2)
