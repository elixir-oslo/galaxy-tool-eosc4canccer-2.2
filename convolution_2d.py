import argparse
import os
from typing import List, Tuple

import cv2
import numpy as np
import pydicom


def get_dicom_files(path_input_folder: str) -> List[str]:
    """
    Find all DICOM files in the given folder.

    Parameters
    ----------
    path_input_folder : str
        Path to the directory containing DICOM files.

    Returns
    -------
    List[str]
        List of DICOM file paths.
    """
    if not os.path.exists(path_input_folder):
        raise FileNotFoundError(f"Folder '{path_input_folder}' does not exist.")

    dicom_files = [
        os.path.normpath(os.path.join(root, file))
        for root, _, files in os.walk(path_input_folder)
        for file in files
        if file.lower().endswith(".dcm") and "_denoised" not in file.lower()
    ]

    if not dicom_files:
        raise FileNotFoundError(
            f"No DICOM files (.dcm) found in '{path_input_folder}'."
        )
    return dicom_files


def check_order_dicom(list_input_dicom: List[str]) -> List[str]:
    """
    Check that all DICOM images in the input folder have valid and unique Instance Number,
    slices are 2D and have the same dimensions, data type and valid images.
    Reorder the DICOM file paths based on the Instance Number.

    Parameters
    ----------
    list_input_dicom : List[str]
        List of DICOM file paths.

    Returns
    -------
    List[str]
        List of DICOM file paths sorted by Instance Number.
    """
    dicom_with_instances = []
    instance_numbers_set = set()
    reference_shape = None
    reference_dtype = None

    for path in list_input_dicom:

        ds = pydicom.dcmread(path)

        instance_number = getattr(ds, "InstanceNumber", None)
        if instance_number is None:
            raise ValueError(f"Missing InstanceNumber in DICOM file {path}")

        if instance_number in instance_numbers_set:
            raise ValueError(
                f"Duplicate Instance Number detected: {instance_number} in DICOM file {path}"
            )
        instance_numbers_set.add(instance_number)

        if not isinstance(ds.pixel_array, np.ndarray):
            raise TypeError(
                f"Invalid image format. Expected a NumPy array. DICOM file {path}."
            )

        if len(ds.pixel_array.shape) != 2:
            raise ValueError(f"DICOM file {path} is not a 2D slice.")

        if reference_shape is None:
            reference_shape = ds.pixel_array.shape
        elif ds.pixel_array.shape != reference_shape:
            raise ValueError(
                f"Inconsistent slice dimensions detected in DICOM file {path}."
            )

        if reference_dtype is None:
            reference_dtype = ds.pixel_array.dtype
        elif ds.pixel_array.dtype != reference_dtype:
            raise ValueError(
                f"Inconsistent slice data type detected in DICOM file {path}."
            )
        dicom_with_instances.append((instance_number, path))

    # Sort files by Instance Number
    dicom_with_instances.sort(key=lambda x: x[0])
    return [path for _, path in dicom_with_instances]


def convolution_2d(
    list_input_dicom_sorted: List[str],
    vol_dims: Tuple[int, int, int],
    vol_dtype: np.dtype,
    kernel: np.ndarray,
    path_output_folder: str,
    dict_original_filenames: dict,
) -> None:
    """
    2D Convolution (Image Filtering) and save processed DICOM files to an output folder.

    Parameters
    ----------
    list_input_dicom_sorted : List[str]
        List of DICOM file paths sorted by Instance Number.
    vol_dims : Tuple[int, int, int]
        Dimensions of the volume.
    vol_dtype : np.dtype
        Data type of the volume.
    kernel : np.ndarray
        Convolution kernel.
    path_output_folder: str
         Folder where to save the DICOM files
    """
    dicom_meta = pydicom.dcmread(list_input_dicom_sorted[0])
    elem_2 = dicom_meta.SeriesInstanceUID
    index_last_2 = elem_2.rfind(".")
    new_series_instance_uid = elem_2[0 : index_last_2 + 1] + str(
        int(elem_2[index_last_2 + 1 : len(elem_2)]) + 1
    )

    for i_dicom, path_dicom in enumerate(list_input_dicom_sorted):

        dico = pydicom.dcmread(path_dicom)

        # Apply 2D convolution filter and ensure data type consistency
        image = dico.pixel_array
        den_max = cv2.filter2D(image, -1, kernel).astype(vol_dtype)

        # Update DICOM tags
        elem_01 = dico[0x0008, 0x103E].value
        new_elem_01 = "".join(elem_01)
        if new_elem_01.rfind("_DENOISED") == -1:
            dico.SeriesDescription = str(new_elem_01) + str("_DENOISED")

        elem_1 = dico.SOPInstanceUID
        index_last = elem_1.rfind(".")
        dico.SOPInstanceUID = elem_1[0 : index_last + 1] + str(
            int(elem_1[index_last + 1 : len(elem_1)]) + len(list_input_dicom_sorted)
        )

        dico.file_meta.MediaStorageSOPInstanceUID = elem_1[0 : index_last + 1] + str(
            int(elem_1[index_last + 1 : len(elem_1)]) + len(list_input_dicom_sorted)
        )

        dico.FrameOfReferenceUID = elem_1[0 : index_last + 1] + str(
            int(elem_1[index_last + 1 : len(elem_1)]) + len(list_input_dicom_sorted)
        )

        dico.SeriesInstanceUID = new_series_instance_uid

        # Update pixel data with the processed image
        dico.PixelData = den_max.tobytes()

        # Construct the new file name for the denoised DICOM file
        name_denoised = os.path.basename(dict_original_filenames[path_dicom]).replace(".dcm", "_denoised.dcm")
        new_path = os.path.join(path_output_folder, name_denoised)
        dico.save_as(new_path)


            


def main():
    """
    Main function to process DICOM files, generate a 3D image, and 2D Convolution.
    """
    # path_input_folder = "./input"
    # path_output_folder = "./output"
    
    parser = argparse.ArgumentParser(description="Process DICOM files and calculate SNR.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input folder containing DICOM files.")
    parser.add_argument("--metadata", type=str, required=True, help="Path to metadata directory containing original filenames.")
    args = parser.parse_args()

    path_input_files = args.input
    metadata_info = args.metadata
    # metadata_info = None
    path_output_folder = "convolution_outputs"
    
    # Check if the output folder exists, if not create it
    if not os.path.exists(path_output_folder):
        os.makedirs(path_output_folder)
     
    if "," in path_input_files:
        path_input_files = path_input_files.split(",")
        
    if "," in metadata_info:
        metadata_info = metadata_info.split(",")
        
    dict_original_filenames = {}
    for i in range(len(path_input_files)):
        dict_original_filenames[path_input_files[i]] = metadata_info[i]
        

    try:        
        list_input_dicom_sorted = path_input_files

        # 2D Convolution (Image Filtering)
        ref_ds = pydicom.dcmread(list_input_dicom_sorted[0])
        num_rows, num_columns = ref_ds.pixel_array.shape
        num_slices = len(list_input_dicom_sorted)
        vol_dims = (num_slices, num_rows, num_columns)
        vol_dtype = ref_ds.pixel_array.dtype

        kernel = np.ones((5, 5), vol_dtype) / 25

        convolution_2d(
            list_input_dicom_sorted, vol_dims, vol_dtype, kernel, path_output_folder, dict_original_filenames
        )
        print("2D Convolution completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
