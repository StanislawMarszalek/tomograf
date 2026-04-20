import datetime as dt
import numpy as np
from pydicom import dcmread
from pydicom.dataset import Dataset, FileDataset
from pydicom.filewriter import validate_file_meta
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid
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
    data=dcmread(pathfile)

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


def save_dicom_file(file_path: str, image: np.ndarray,
                    patient_name: str, patient_id: str,
                    patient_sex:str,patient_age:str,
                    study_date: str, comments: str)->None:
    #Image must be nromalized before apllying clip
    """
    Save given values and given image as a dicon file 
    (values must be given in a appropriate format )
    
    :param file_path: Path where the dicom file will be saved
    :type file_path: str
    :param image: Image to save in the dicom file (image must be normalized)
    :type image: np.ndarray
    :param patient_name: Name of the patient 
    :type patient_name: str
    :param patient_id: ID of the patients
    :type patient_id: str
    :param patient_sex: Gender of the patient
    :type patient_sex: str
    :param patient_age: Age of the patient
    :type patient_age: str
    :param study_date: Date of the examination
    :type study_date: str
    :param comments: Comments about the examination
    :type comments: str
    """
    #Image must be normalized before apllying clip
    img16 = np.clip(image * 65535.0, 0, 65535).astype(np.uint16)

    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()
    validate_file_meta(file_meta, enforce_standard=False)

    data_set = FileDataset(file_path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    data_set.is_little_endian = True
    data_set.is_implicit_VR = False

    data_set.SOPClassUID = SecondaryCaptureImageStorage
    data_set.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    data_set.StudyInstanceUID = generate_uid()
    data_set.SeriesInstanceUID = generate_uid()
    data_set.FrameOfReferenceUID = generate_uid()
    data_set.Modality = "OT"
    data_set.PatientName = patient_name or "Anonymous"
    data_set.PatientID = patient_id or "000000"
    data_set.StudyDate = study_date if len(study_date) == 8 and study_date.isdigit() else ""
    data_set.ContentDate = data_set.StudyDate
    data_set.ContentTime = dt.datetime.now().strftime("%H%M%S")
    data_set.ImageComments = comments or ""
    data_set.PatientSex=patient_sex or ""
    data_set.PatientAge=patient_age or ""
    data_set.StudyDescription = "Tomograph reconstruction"
    data_set.SeriesDescription = "Tomograph reconstruction"

    data_set.Rows, data_set.Columns = img16.shape
    data_set.SamplesPerPixel = 1
    data_set.PhotometricInterpretation = "MONOCHROME2"
    data_set.BitsAllocated = 16
    data_set.BitsStored = 16
    data_set.HighBit = 15
    data_set.PixelRepresentation = 0
    data_set.PixelSpacing = ["1", "1"]
    data_set.RescaleIntercept = "0"
    data_set.RescaleSlope = "1"
    data_set.PixelData = img16.tobytes()
    data_set.save_as(file_path, write_like_original=False)

if __name__=="__main__":
    img,data_test=read_dicom_file("./SADDLE_PE-large.dcm_updated_2026_4_19_21_5.dcm")
    show_img(img,"Read from dicom file")
    for name,value in data_test.items():
        print(f"{name}: {value}")
    img=(img-np.min(img))/(np.max(img)-np.min(img))
    save_dicom_file("./tst.dcm",img,"Kowalski^Jan","","O","018M","","test 123")
    img,data_test=read_dicom_file("./tst.dcm")
    for name,value in data_test.items():
        print(f"{name}: {value}")
