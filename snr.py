import argparse
import os
from typing import List, Tuple

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
        if file.lower().endswith(".dcm")
    ]
    
    all_files = [
        os.path.normpath(os.path.join(root, file))
        for root, _, files in os.walk(path_input_folder)
        for file in files
    ]


    if not dicom_files:
        raise FileNotFoundError(
            f"No DICOM files (.dcm) found in '{path_input_folder}'. . Ony '{" ".join(all_files)}'"
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


def calculate_snr(
    list_input_dicom_sorted: List[str],
    vol_dims: Tuple[int, int, int],
    vol_dtype: np.dtype,
    kernel_size: int,
) -> float:
    """
    Calculate the volume signal-to-noise ratio.

    Parameters
    ----------
    list_input_dicom_sorted : List[str]
        List of DICOM file paths sorted by Instance Number.
    vol_dims : Tuple[int, int, int]
        Dimensions of the volume.
    vol_dtype : np.dtype
        Data type of the volume.
    kernel_size : int
        Dimensions of the kernel.

    Returns
    -------
    float
        SNR value.
    """
    # ROI parameters
    roi_background = np.empty((vol_dims[0], kernel_size, kernel_size), dtype=vol_dtype)
    roi_object = np.empty((vol_dims[0], kernel_size, kernel_size), dtype=vol_dtype)
    object_row_start = (vol_dims[1] - kernel_size) // 2
    object_col_start = (vol_dims[2] - kernel_size) // 2

    # Calculate the ROI of the background and object
    for path_ind, path_elem in enumerate(list_input_dicom_sorted):

        ds = pydicom.dcmread(path_elem)
        image = ds.pixel_array

        roi_background[path_ind, :, :] = image[:kernel_size, :kernel_size]

        roi_object[path_ind, :, :] = image[
            object_row_start : object_row_start + kernel_size,
            object_col_start : object_col_start + kernel_size,
        ]

    std_background = roi_background.std()
    return (
        float("inf")
        if std_background == 0
        else round(roi_object.mean() / std_background, 2)
    )


def save_snr_txt(
    snr: float,
    output_file: str,
    series_number: str,
    format_file: str,
) -> None:
    """
    Save the SNR value in the output folder as a text file.

    Parameters
    ----------
    snr : float
        SNR value.
    path_output_folder : str
        Folder where to save the text file
    series_number : str
        Series number
    format_file : str
        Format of the file

    """
    # if not os.path.exists(path_output_folder):
    #     raise FileNotFoundError(f"Folder '{path_output_folder}' does not exist.")

    # output_path = os.path.join(
    #     path_output_folder, f"snr_scan_{series_number}.{format_file}"
    # )

    string_to_write = f"SNR for scan {series_number}: {snr}."
    if snr == float("inf"):
        string_to_write += " The signal is present but the noise is zero. "

    with open(output_file, "w") as file:
        file.write(string_to_write)


def main():
    """
    Main function to process DICOM files and calculate SNR.
    """
    # path_input_folder = "./input"
    # path_output_folder = "./output"
    # kernel_size = 80
    
    parser = argparse.ArgumentParser(description="Process DICOM files and calculate SNR.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input folder containing DICOM files.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output folder to save the SNR result.")
    parser.add_argument("--kernel_size", type=int, default=80, help="Kernel size for SNR calculation.")
    args = parser.parse_args()

    path_input_files = args.input
    path_output_folder = args.output
    kernel_size = args.kernel_size
    
    if "," in path_input_files:
        path_input_files = path_input_files.split(",")
    
    # raise ValueError(
    #     f"Invalid input folder '{path_input_files}'. Please provide a valid path. '{type((path_input_files))}'"
    # )

    try:    
        # # Get a list of dicom files contained in XNAT input folder
        # list_input_dicom = get_dicom_files(path_input_folder)
        # print(
        #     f"Found {len(list_input_dicom)} DICOM files in '{path_input_folder}':\n"
        #     + "\n".join(f"'{file}'" for file in list_input_dicom)
        # )

        # Verify that the DICOM files in the XNAT input folder are valid
        # and reorder them based on the Instance Number.
        # list_input_dicom_sorted = check_order_dicom(list_input_dicom)
        # print(
        #     "DICOM files sorted by InstanceNumber:\n"
        #     + "\n".join(f"'{file}'" for file in list_input_dicom_sorted)
        # )
        list_input_dicom_sorted = path_input_files

        # Calculate SNR
        ref_ds = pydicom.dcmread(list_input_dicom_sorted[0])
        num_rows, num_columns = ref_ds.pixel_array.shape
        num_slices = len(list_input_dicom_sorted)
        vol_dims = (num_slices, num_rows, num_columns)
        vol_dtype = ref_ds.pixel_array.dtype

        snr = calculate_snr(list_input_dicom_sorted, vol_dims, vol_dtype, kernel_size)
        print(f"SNR calculated successfully. SNR = {snr}")

        # Save SNR in XNAT output folder
        series_number = str(getattr(ref_ds, "SeriesNumber", "unknown"))
        save_snr_txt(snr, path_output_folder, series_number, "txt")
        print(f"SNR for scan {series_number} saved successfully.")

    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
