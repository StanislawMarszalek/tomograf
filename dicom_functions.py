import numpy as np
import pydicom
from helpers import show_img


def read_dicom_file(pathfile:str)->tuple[np.ndarray,dict[str,str]]:
    """
    Read a dicom file
    
    :param pathfile: Path to the dicom file
    :type pathfile: str
    :return: An image from the dicom file and data about about the patient 
    and the examination (for example: name,ID,Age)
    :rtype: tuple[ndarray, dict[str, str]]
    """
    data=pydicom.dcmread(pathfile)

    image=data.pixel_array
    if image.ndim>2:
        image=image[0]
    image = np.asarray(image, dtype=float)

    #Basic data about the patient and the examination
    meta_data:dict[str,str]={
        "PatientName": str(getattr(data, "PatientName", "")),
        "PatientID": str(getattr(data, "PatientID", "")),
        "PatientSex":str(getattr(data, "PatientSex", "")),
        "PatientAge":str(getattr(data, "PatientAge", "")),
        "StudyDate": str(getattr(data,"StudyDate", "")),
        "ImageComments": str(getattr(data, "ImageComments", "")),
    }
    return image,meta_data


if __name__=="__main__":
    img,data=read_dicom_file("tomograf-dicom\Kolo.dcm")
    show_img(img,"Read from dicom file")
    for name,value in data.items():
        print(f"{name}: {value}")
