import numpy as np

def compute_sobel_gradients_two_loops(image):
    # Get image dimensions
    height, width = image.shape

    # Initialize output gradients
    gradient_x = np.zeros_like(image, dtype=np.float64)
    gradient_y = np.zeros_like(image, dtype=np.float64)

    # Pad the image with zeros to handle borders
    padded_image = np.pad(image, ((1, 1), (1, 1)), mode='constant', constant_values=0)
# __________end of block__________

    # Define the Sobel kernels for X and Y gradients
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float64)

    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype=np.float64)

    # Apply Sobel filter for X and Y gradients using convolution
    for i in range(1, height + 1):
        for j in range(1, width + 1):
            window = padded_image[i-1:i+2, j-1:j+2]
            gradient_x[i-1, j-1] = np.sum(window * sobel_x)
            gradient_y[i-1, j-1] = np.sum(window * sobel_y)

    return gradient_x, gradient_y

def compute_gradient_magnitude(sobel_x, sobel_y):
    '''
    Compute the magnitude of the gradient given the x and y gradients.

    Inputs:
        sobel_x: numpy array of the x gradient.
        sobel_y: numpy array of the y gradient.

    Returns:
        magnitude: numpy array of the same shape as the input [0] with the magnitude of the gradient.
    '''
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    return magnitude

def compute_gradient_direction(sobel_x, sobel_y):
    '''
    Compute the direction of the gradient given the x and y gradients. Angle must be in degrees in the range (-180; 180].
    Use arctan2 function to compute the angle.

    Inputs:
        sobel_x: numpy array of the x gradient.
        sobel_y: numpy array of the y gradient.

    Returns:
        gradient_direction: numpy array of the same shape as the input [0] with the direction of the gradient.
    '''
    # Calculate angle in radians using arctangent of y/x
    radians = np.arctan2(sobel_y, sobel_x)

    # Convert radians to degrees
    degrees = np.degrees(radians)

    # Adjust angles to be in (-180, 180]

    # Угол -181° формально эквивалентен +179°
    # -181° → -181° + 360° = 179°
    degrees = np.where(degrees <= -180, degrees + 360, degrees)

    # 181° → 181° - 360° = -179°
    degrees = np.where(degrees > 180, degrees - 360, degrees)

    # Если угол = -180°, заменяет его на 180°.
    degrees = np.where(degrees == -180, 180, degrees)

    return degrees

cell_size = 7
def compute_hog(image, pixels_per_cell=(cell_size, cell_size), bins=9):
    # 1. Convert the image to grayscale if it's not already (assuming the image is in RGB or BGR)
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)  # Simple averaging to convert to grayscale

    # 2. Compute gradients with Sobel filter
    gradient_x, gradient_y = compute_sobel_gradients_two_loops(image)

    # 3. Compute gradient magnitude and direction
    magnitude = compute_gradient_magnitude(gradient_x, gradient_y)
    direction = compute_gradient_direction(gradient_x, gradient_y)

    # 4. Create histograms of gradient directions for each cell
    cell_height, cell_width = pixels_per_cell
    n_cells_x = image.shape[1] // cell_width
    n_cells_y = image.shape[0] // cell_height

    histograms = np.zeros((n_cells_y, n_cells_x, bins))

    # Ширина каждого бина в градусах
    bin_width = 360 / bins  # 40 градусов на бин

    for i in range(n_cells_y):
        for j in range(n_cells_x):
            # Координаты текущей ячейки
            y_start = i * cell_height
            y_end = y_start + cell_height
            x_start = j * cell_width
            x_end = x_start + cell_width

            # Вырезаем область для текущей ячейки
            cell_magnitude = magnitude[y_start:y_end, x_start:x_end]
            cell_direction = direction[y_start:y_end, x_start:x_end]

            # Создаем гистограмму для текущей ячейки
            cell_histogram = np.zeros(bins)
            for y in range(cell_magnitude.shape[0]):
                for x in range(cell_magnitude.shape[1]):
                    angle = cell_direction[y, x]
                    mag = cell_magnitude[y, x]

                    # Преобразуем угол в диапазон 0-360
                    adjusted_angle = angle + 180  # Теперь диапазон [0, 360)
                    bin_index = int(adjusted_angle // bin_width)

                    # Обработка граничных случаев (например, угол = 180)
                    bin_index = min(bin_index, bins - 1)
                    cell_histogram[bin_index] += mag

            # Нормировка гистограммы (сумма = 1)
            total = cell_histogram.sum()
            if total > 0:
                cell_histogram /= total

            histograms[i, j] = cell_histogram

    return histograms