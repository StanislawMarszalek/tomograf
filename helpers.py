import numpy as np
import matplotlib.pyplot as plt
import cv2
def show_img(image:np.ndarray,title:str|None=None)->None:
    """
    Display a given image in the gray scale
    
    :param image: Image to display
    :type image: np.ndarray
    :param title: Title for the image , if None no title is displayed
    :type title: str | None
    """
    if isinstance(title,str):
        plt.title(title)
    plt.imshow(image,cmap="gray")
    plt.show()
    return


def read_img(pathfile:str)->cv2.UMat|None:
    """
    Read and return an image
    
    :param pathfile: Description
    :type pathfile: str
    :return: Description
    :rtype: UMat | None
    """
    try:
        img=cv2.imread(pathfile,0)
    except (ValueError,FileNotFoundError,FileExistsError):
        print(f"Could not ope the file: {pathfile}")
        return None
    img = img.astype(np.float64)
    return img/img.max()


def normalize_img(image:np.ndarray)->np.ndarray:
    """
    Normalize the given image
    
    :param image: Description
    :type image: np.ndarray
    :return: Normalized image
    :rtype: ndarray
    """
    image = np.maximum(image, 0)
    image = (image / np.quantile(image, 0.999)).astype(np.float64) if np.quantile(image, 0.999)!=0 else image
    image = np.minimum(image, 1)
    return image
