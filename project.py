import numpy
import cv2
import skimage.io as ski



#blad liczyc z biblioteki np sciki learn
#bresenham własny
#TODO funckje do paddingu i unpaddingu (zapamietywac oryginalne wymiary obrazu)
import numpy as np

# --- Twój algorytm Bresenhama ---

def plot_line_low(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
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
    if abs(y1 - y0) < abs(x1 - x0):
        if x0 > x1:
            return plot_line_low(x0=x1, y0=y1, x1=x0, y1=y0)
        return plot_line_low(x0, y0, x1, y1)
    if y0 > y1:
        return plot_line_high(x0=x1, y0=y1, x1=x0, y1=y0)
    return plot_line_high(x0, y0, x1, y1)


def prepare_padded_image(image: np.ndarray):
    """
    1. Tworzy kwadratowy obraz o boku równym dłuższemu bokowi oryginału.
    2. Oblicza promień r jako połowę przekątnej kwadratu, zaokrągloną w górę.
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

    return padded_image, r

def radon_transform(image: np.ndarray, n_detectors: int, phi_angle: float, n_steps: int) -> tuple[list[np.ndarray], np.ndarray]:
    # Przygotowanie obrazu i promienia r
    padded_img, r = prepare_padded_image(image)

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
        
        intermediate_sinograms.append(sinogram.copy()/np.max(sinogram))

    return intermediate_sinograms, sinogram/np.max(sinogram)
if __name__=="__main__":
    detector_numb:int=5
    viev_numb:int=10
    sample_views=numpy.zeros(shape=(viev_numb,detector_numb,),dtype=numpy.float64)
    sample_views[3,3]=2
    sample_views[2,2]=5
    sample_views=sample_views/sample_views.max()
    image=cv2.imread("tomograf-obrazy\Kolo.jpg",0)

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
    intermediate_list, final_sinogram = radon_transform(
        image=image, 
        n_detectors=n_detectors, 
        phi_angle=phi_angle, 
        n_steps=n_steps, 
    )
    
    cv2.imshow("grey",final_sinogram)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
