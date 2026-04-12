import numpy as np
import cv2
import skimage.io as ski



#blad liczyc z biblioteki np sciki learn
#bresenham własny
#TODO funckje do paddingu i unpaddingu (zapamietywac oryginalne wymiary obrazu)



def plot_line_low(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """
    Finds coordinates for line if OX is deritive axis
    
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
    dx: int = x1 - x0 
    dy: int = y1 - y0 
    y_increment: int = 1

    if dy < 0:
        y_increment *= -1
        dy *= -1

    difference: int = 2 * dy - dx
    y: int = y0
    for x in range(x0, x1 + 1):
        line_coords.append((x, y))
        if difference > 0:
            y += y_increment
            difference += 2 * (dy - dx)
        else:
            difference += 2 * dy
    return line_coords

def plot_line_high(x0:int, y0:int, x1:int, y1:int) -> list[tuple[int, int]]:
    """
    Finds coordinates for line if OY is deritive axis
    
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
    dx: int = x1 - x0
    dy: int = y1 - y0
    x_increment: int = 1

    if dx < 0:
        dx *=-1
        x_increment *=-1
        
    difference: int = 2 * dx - dy
    x: int = x0

    for y in range(y0, y1 + 1):
        line_coords.append((x, y))
        if difference > 0:
            x += x_increment
            difference += 2 * (dx - dy)
        else:
            difference += 2 * dx
    return line_coords

def bresenham_algorith(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
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
    if abs(y1 - y0) < abs(x1 - x0):
        if x0 > x1:
            return plot_line_low(x0=x1, y0=y1, x1=x0, y1=y0)
        return plot_line_low(x0, y0, x1, y1)
    if y0 > y1:
        return plot_line_high(x0=x1, y0=y1, x1=x0, y1=y0)
    return plot_line_high(x0, y0, x1, y1)


def prepare_padded_image(image: np.ndarray)->tuple[np.ndarray,int,tuple[int,int]]:
    
    """
    Add padding to the given image
    
    :param image: Image to be padded
    :type image: np.ndarray
    :return: Tuple which contains the padded image, the radius and the input image's shape
    :rtype: tuple[ndarray, int,tuple[int,int]]
    """
    height, width = image.shape
    max_side = max(height, width)

    padded_image = np.zeros((max_side, max_side))

    # Centrowanie oryginału
    offset_y = (max_side - height) // 2
    offset_x = (max_side - width) // 2
    padded_image[offset_y:offset_y+height, offset_x:offset_x+width] = image

    # Promień r = (przekątna kwadratu / 2) zaokrąglona w górę
    diagonal = np.sqrt(2 * (max_side**2))
    r = int(np.ceil(diagonal / 2))

    return padded_image, r,(height,width)

def radon_transform(image: np.ndarray, n_detectors: int, phi_angle: float, n_steps: int) -> tuple[list[np.ndarray], np.ndarray,int,tuple[int,int]]:
    
    """
    Radon trasnformation to create a sinogram
    
    :param image: X-ray image to make a sinogram from
    :type image: np.ndarray
    :param n_detectors: Number of detecotrs
    :type n_detectors: int
    :param phi_angle: detector range
    :type phi_angle: float
    :param n_steps: Number of steps to do
    :type n_steps: int
    :return: Function returns intermediate sinograms ,the final singorma (All normalized),\nradius(length from the image center to the emiter) ,original image's height and width (as a tupe)
    :rtype: tuple[list[ndarray], ndarray, int, tuple[int, int]]
    """
    # Przygotowanie obrazu i promienia r
    padded_img, r,(org_height,org_width) = prepare_padded_image(image)

    heighy, width = padded_img.shape
    center_x, center_y = width // 2, heighy // 2

    sinogram = np.zeros((n_steps, n_detectors))
    intermediate_sinograms = []

    # Kąty obrotu alfa
    alphas = np.linspace(0, 2 * np.pi, n_steps, endpoint=False)

    for step_idx, alpha in enumerate(alphas):

        emiter_x = int(round(r * np.cos(alpha) + center_x))
        emiter_y = int(round(r * np.sin(alpha) + center_y))
        
        for i in range(n_detectors):

            detector_angle = alpha + np.pi - (phi_angle / 2) + i * (phi_angle / (n_detectors - 1))

            detcor_i_x = int(round(r * np.cos(detector_angle) + center_x))
            detcor_i_y = int(round(r * np.sin(detector_angle) + center_y))

            # Algorytm Bresenhama
            coordinates = bresenham_algorith(emiter_x, emiter_y, detcor_i_x, detcor_i_y)

            line_sum = 0.0
            count = 0
            #brane sa tylko pkt bedace czescia obrazu
            for px, py in coordinates:
                if 0 <= px < width and 0 <= py < heighy:
                    line_sum += padded_img[py, px]
                    count += 1

            if count > 0:
                # POPRAWKA LUSTRZANEGO ODBICIA:
                # Zapisujemy wynik do sinogramu odwracając indeks detektora (n_detectors - 1 - i)
                sinogram[step_idx, n_detectors - 1 - i] = line_sum / count
        
        intermediate_sinograms.append(sinogram.copy())
    max_val=sinogram.max()
    min_val=sinogram.min()
    intermediate_sinograms=[(x-min_val)/(max_val-min_val) for x in intermediate_sinograms]

    return intermediate_sinograms, (sinogram-min_val)/(max_val-min_val),r,(org_height,org_width)


def back_projection(final_sinogram: np.ndarray, n_detectors: int, phi_angle: float, n_steps: int,radius:int,original_img_height:int,original_img_width:int,filtering:bool=True)->tuple[list[np.ndarray], np.ndarray]:
    
    """
    Back projection to reconstruct the image
    
    :param final_sinogram: Sinogram the imaghe is made from
    :type final_sinogram: np.ndarray
    :param n_detectors: Number of detectors
    :type n_detectors: int
    :param phi_angle: Detectors range
    :type phi_angle: float
    :param n_steps: Number of steps
    :type n_steps: int
    :param radius: The radius of the image (length from the image center to the emiter)
    :type radius: int
    :param original_img_height: Height of the original image
    :type original_img_height: int
    :param original_img_width: Width of the original image
    :type original_img_width: int
    :param filtering: Switch to decide if filtering should be applied or not (default:True)
    :type filtering: bool
    :return: Function returns intermediate images and the final image
    :rtype: tuple[list[ndarray], ndarray]
    """
    
    height, width = original_img_height,original_img_width
    center_x, center_y = width // 2, height // 2

    reconstructed = np.zeros((original_img_height, original_img_width))
    intermediate_reconstructed_img = []
    # Kąty obrotu alfa
    alphas = np.linspace(0, 2 * np.pi, n_steps, endpoint=False)
    if filtering:
        # Tworzenie kernala
        k_values = np.arange(-10, 11)
        kernel = np.zeros(len(k_values))

        for i, k in enumerate(k_values):
            if k == 0:
                kernel[i] = 1
            elif k % 2 == 0:
                kernel[i] = 0
            else:
                kernel[i] = (-4 / (np.pi**2)) / (k**2)

        filtered_sinogram = np.zeros_like(final_sinogram)

        for i in range(final_sinogram.shape[0]):
            filtered_sinogram[i, :] = np.convolve(final_sinogram[i, :], kernel, mode="same")
    else:
        filtered_sinogram = final_sinogram.copy()


    for step_idx, alpha in enumerate(alphas):

        emiter_x = int(round(radius * np.cos(alpha) + center_x))
        emiter_y = int(round(radius * np.sin(alpha) + center_y))

        for i in range(n_detectors):

            detector_angle = alpha + np.pi - (phi_angle / 2) + i * (phi_angle / (n_detectors - 1))

            detcor_i_x = int(round(radius * np.cos(detector_angle) + center_x))
            detcor_i_y = int(round(radius * np.sin(detector_angle) + center_y))

            # Algorytm Bresenhama
            coordinates = bresenham_algorith(emiter_x, emiter_y, detcor_i_x, detcor_i_y)
            #Filtering of coords
            coordinates=[(px,py) for (px,py) in coordinates if ((0<=px<width)and(0<=py<height))]
            count = len(coordinates)
            view_val_per_point = filtered_sinogram [step_idx, n_detectors - 1 - i]#/count if count>0 else 0

            for px, py in coordinates:
                reconstructed[py,px] += view_val_per_point
        
        intermediate_reconstructed_img.append(reconstructed.copy())
    max_val=reconstructed.max()
    min_val=reconstructed.min()
    intermediate_reconstructed_img=[(x-min_val)/(max_val-min_val) for x in intermediate_reconstructed_img]
    return intermediate_reconstructed_img, (reconstructed-min_val)/(max_val-min_val)

if __name__=="__main__":
    detector_numb:int=5
    viev_numb:int=10

    image=cv2.imread("tomograf-obrazy\Kropka.jpg",0)

    cv2.imshow("grey",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 2. Definicja parametrów zgodnych z Twoimi wzorami:
    # n_detectors (n) - liczba detektorów w wachlarzu 
    n_detectors = 180 

    # phi_angle (phi) - rozpiętość wachlarza w radianach [cite: 2, 5]
    # pi / 2 to 90 stopni - dość szeroki wachlarz
    phi_angle = np.pi / 2 

    # n_steps - ile "zdjęć" (obrotów alfa) wykonujemy dookoła obiektu
    n_steps = 90


    # 3. Wywołanie funkcji
    intermediate_list, finsal_sinogram ,radius,(org_height,org_width)= radon_transform(
        image=image, 
        n_detectors=n_detectors, 
        phi_angle=phi_angle, 
        n_steps=n_steps, 
    )
    
    cv2.imshow("grey",finsal_sinogram)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    intermediate_list,final_reconstructed=back_projection(finsal_sinogram,n_detectors,phi_angle,n_steps,radius,org_height,org_width)
    cv2.imshow("Filter",final_reconstructed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    intermediate_list,final_reconstructed=back_projection(finsal_sinogram,n_detectors,phi_angle,n_steps,radius,org_height,org_width,False)
    cv2.imshow("NO Filter",final_reconstructed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
