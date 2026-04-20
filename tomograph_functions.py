import numpy as np
from helpers import show_img,read_img,normalize_img


def bresenham_algorithm(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """
    Finds coordinates for line
    
    :param x0: Discrite x coordinate for the start
    :type x0: int
    :param y0: Discrite y coordinate for the start
    :type y0: int
    :param x1: Discrite x coordinate for the end
    :type x1: int
    :param y1: Discrite y coordinate for the end
    :type y1: int
    :return: List of coordinates the line goes through
    :rtype: list[tuple[int, int]]
    """

    line_coords: list[tuple[int, int]] = []
    dx:int=abs(x1-x0)
    sx:int=1 if x0<x1 else -1
    dy:int=-abs(y1-y0)
    sy:int=1 if y0<y1 else -1
    error:int=dx+dy

    while True:
        line_coords.append((x0,y0))
        e2:int=2*error
        if e2>=dy:
            if x0==x1:
                break
            error+=dy
            x0+=sx
        if e2<=dx:
            if y0==y1:
                break
            error+=dx
            y0+=sy
    return line_coords


def radon_transform(image: np.ndarray,
                   numb_detectors: int = 180,
                   step: float = 1,
                   angular_spread: float = 90) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Radon transform to creating sinogram
    
    :param image: input image
    :type image: np.ndarray
    :param numb_detectors: Number of detectors
    :type numb_detectors: int
    :param step: Angle for moving the emiter and the detectors in a single iteration
    :type step: float
    :param angular_spread: Angular spread of the detectors
    :type angular_spread: float
    :return: The functions returns intermediate (normalized) sinograms and the final sinogram
    :rtype: tuple[list[ndarray], ndarray]
    """

    step = np.radians(step)
    angular_spread = np.radians(angular_spread)

    height, width = image.shape
    center = np.array([width // 2, height // 2])

    radius = max(center)*3

    angles = np.arange(0.0, 2 * np.pi, step)

    sinogram = np.zeros((len(angles), numb_detectors), dtype=np.float64)
    intermediate_sinograms = [sinogram.copy()]

    detector_angle_gap = angular_spread / (numb_detectors - 1)

    for view, angle in enumerate(angles):
        emiter = np.array([np.cos(angle), np.sin(angle)]) * radius + center
        emiter_x, emiter_y = int(emiter[0]), int(emiter[1])

        for i in range(numb_detectors):
            detector_angle = angle + np.pi - angular_spread / 2 + i * detector_angle_gap
            detector_i = np.array([np.cos(detector_angle),np.sin(detector_angle)]) * radius + center
            detector_i_x,detector_i_y = int(detector_i[0]), int(detector_i[1])

            coords = bresenham_algorithm(emiter_x, emiter_y, detector_i_x, detector_i_y)

            vals = []
            for x, y in coords:
                if 0 <= x < width and 0 <= y < height:
                    vals.append(image[y, x])

            sinogram[view, i] = np.mean(vals) if vals else 0.0

        intermediate_sinograms.append(sinogram.copy())

    max_val = sinogram.max()
    min_val = sinogram.min()
    if max_val > min_val:
        intermediate_sinograms = [normalize_img(x) for x in intermediate_sinograms]

    return intermediate_sinograms, sinogram



def filtr_sinogram(sinogram:np.ndarray)->np.ndarray:
    """
    Apply filtering function to the given sinogram
    
    :param sinogram: Sinogram to filter
    :type sinogram: np.ndarray
    :return: Filtered version of the sinogram
    :rtype: ndarray
    """
    kernel_list=list(range(-10, 11))
    for idx,k in enumerate(kernel_list):
        if k == 0:
            kernel_list[idx] = 1
        elif k % 2 == 0:
            kernel_list[idx] = 0
        else:
            kernel_list[idx] = (-4 / (np.pi**2)) / (k**2)

    kernel=np.array(kernel_list)

    for i in range(sinogram.shape[0]):
        sinogram[i,:]=np.convolve(sinogram[i,:],kernel,"same")
    return sinogram


def back_projection(sinogram: np.ndarray,
                    org_height: int,
                    org_width: int,
                    numb_detectors: int = 180,
                    step: float = 1,
                    angular_spread: float = 90)-> tuple[list[np.ndarray], np.ndarray]:
    """
    Doing back projection to recreate a image
    
    :param sinogram: sinogram to recreate the image from
    :type sinogram: np.ndarray
    :param org_height: Original image's height
    :type org_height: int
    :param org_width: Original image's width
    :type org_width: int
    :param numb_detectors: Number of detectors
    :type numb_detectors: int
    :param step: Angle for moving the emiter and the detectors in a single iteration
    :type step: float
    :param angular_spread: Angular spread of the detectors
    :type angular_spread: float
    :return: Intermediate reconstructed images and the fina image
    :rtype: tuple[list[ndarray], ndarray]
    """

    step = np.radians(step)
    angular_spread = np.radians(angular_spread)

    reconstructed = np.zeros((org_height, org_width), dtype=np.float64)
    intermediate_recon = [reconstructed.copy()]

    center = np.array([org_width // 2, org_height // 2])
    radius = max(center)*3

    angles = np.arange(0.0, 2 * np.pi, step)
    detector_angle_gap = angular_spread / (numb_detectors - 1)

    for view, angle in enumerate(angles):
        emiter = np.array([np.cos(angle), np.sin(angle)]) * radius + center
        emiter_x, emiter_y = int(emiter[0]), int(emiter[1])

        for i in range(numb_detectors):
            detector_angle = angle + np.pi - angular_spread / 2 + i * detector_angle_gap
            detector_i = np.array([np.cos(detector_angle),np.sin(detector_angle)]) * radius + center
            detector_i_x, detector_i_y = int(detector_i[0]), int(detector_i[1])

            coords = bresenham_algorithm(emiter_x, emiter_y, detector_i_x, detector_i_y)

            for x, y in coords:
                if 0 <= x < org_width and 0 <= y < org_height:
                    reconstructed[y, x] += sinogram[view, i]

        intermediate_recon.append(reconstructed.copy())

    return intermediate_recon, reconstructed


if __name__=="__main__":

    img = read_img("tomograf-obrazy\SADDLE_PE-large.JPG")
    show_img(img, "Readed img")

    # Creating sinogram
    inter_sins, sin = radon_transform(img, 180, 1)
    show_img(sin, "Sinogram unfiltered")

    #Filtering
    filtered = filtr_sinogram(sin)
    show_img(filtered, "Filtered sinogram")

    _, recon_unfiltered = back_projection(sin, img.shape[0], img.shape[1])
    _, recon_filtered = back_projection(filtered, img.shape[0], img.shape[1])

    #comaring reconstructed
    show_img(recon_unfiltered, "Unfiltered recon")
    show_img(recon_filtered, "Filtered recon")

    # Comparing final images
    show_img(normalize_img(recon_unfiltered), "Unfiltered normalize recon")
    show_img(normalize_img(recon_filtered), "Final image")
