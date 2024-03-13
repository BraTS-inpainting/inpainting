### COPIED FROM https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/utils/tensor.py
import sys

import nibabel as nib
import numpy as np
import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanSquaredError,
    MeanSquaredLogError,
)

# these dependencies were used for computing the 2023 challenge metrics
# [tool.poetry.dependencies]
# torchmetrics = ">=1.1.2"
# python = ">=3.10"
# nibabel = ">=3.0"
# numpy = ">=1.25"
# torch = ">=2.0.1"


# def calculate_rmse(gt, pred):
#     gt_tensor = torch.tensor(gt)
#     pred_tensor = torch.tensor(pred)

#     rmse_value = mean_squared_error(pred_tensor, gt_tensor, squared=False)
#     return rmse_value.item()

### COPIED FROM https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/metrics/synthesis.py


def read_nifti(
    input_nifti_path: str,
    maintain_dtype: bool = True,
) -> np.ndarray:
    """
    Read a NIfTI file and return its data as a NumPy array.

    Args:
        input_nifti_path (str): Path to the input NIfTI file.
        maintain_dtype (bool, optional): If True, maintain the data type of the NIfTI data.
                                         If False, allow data type conversion to float. Default is True.

    Returns:
        numpy.ndarray: NIfTI data as a NumPy array.
    """
    the_nifti = nib.load(input_nifti_path)
    nifti_data = the_nifti.get_fdata()

    if maintain_dtype:
        # Get the data type from the NIfTI header
        data_type = the_nifti.header.get_data_dtype()
        # Convert data type if necessary
        nifti_data = nifti_data.astype(data_type, copy=False)

    return nifti_data


def read_nifti_to_tensor(input_nifti_path: str):
    the_amazing_tensor = (
        torch.Tensor(read_nifti(input_nifti_path)).unsqueeze(0).contiguous()
    )
    return the_amazing_tensor


def _structural_similarity_index(
    target: torch.Tensor,
    prediction: torch.Tensor,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Computes the structural similarity index between the target and prediction.

    Args:
        target (torch.Tensor): The target tensor.
        prediction (torch.Tensor): The prediction tensor.
        mask (torch.Tensor, optional): The mask tensor. Defaults to None.

    Returns:
        torch.Tensor: The structural similarity index.
    """
    ssim = StructuralSimilarityIndexMeasure(return_full_image=True)
    _, ssim_idx_full_image = ssim(preds=prediction, target=target)
    mask = torch.ones_like(ssim_idx_full_image) if mask is None else mask
    try:
        ssim_idx = ssim_idx_full_image[mask]
    except Exception as e:
        print(f"Error: {e}")
        if len(ssim_idx_full_image.shape) == 0:
            ssim_idx = torch.ones_like(mask) * ssim_idx_full_image
    return ssim_idx.mean()


def _mean_squared_error(
    target: torch.Tensor,
    prediction: torch.Tensor,
    squared: bool = True,
) -> torch.Tensor:
    """
    Computes the mean squared error between the target and prediction.

    Args:
        target (torch.Tensor): The target tensor.
        prediction (torch.Tensor): The prediction tensor.
        TODO update documentation
    """
    mse = MeanSquaredError(
        squared=squared,
    )

    return mse(preds=prediction, target=target)


def _peak_signal_noise_ratio(
    target: torch.Tensor,
    prediction: torch.Tensor,
    data_range: tuple = None,
    epsilon: float = None,
) -> torch.Tensor:
    """
    Computes the peak signal to noise ratio between the target and prediction.

    Args:
        target (torch.Tensor): The target tensor.
        prediction (torch.Tensor): The prediction tensor.
        data_range (tuple, optional): If not None, this data range (min, max) is used as enumerator instead of computing it from the given data. Defaults to None.
        epsilon (float, optional): If not None, this epsilon is added to the denominator of the fraction to avoid infinity as output. Defaults to None.
    """

    if epsilon == None:
        psnr = (
            PeakSignalNoiseRatio()
            if data_range == None
            else PeakSignalNoiseRatio(data_range=data_range[1] - data_range[0])
        )
        return psnr(preds=prediction, target=target)
    else:  # implementation of PSNR that does not give 'inf'/'nan' when 'mse==0'
        mse = _mean_squared_error(target=target, prediction=prediction)
        if data_range == None:  # compute data_range like torchmetrics if not given
            min_v = (
                0 if torch.min(target) > 0 else torch.min(target)
            )  # look at this line
            max_v = torch.max(target)
        else:
            min_v, max_v = data_range
        return 10.0 * torch.log10(((max_v - min_v) ** 2) / (mse + epsilon))


def _mean_squared_log_error(
    target: torch.Tensor,
    prediction: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the mean squared log error between the target and prediction.

    Args:
        target (torch.Tensor): The target tensor.
        prediction (torch.Tensor): The prediction tensor.
    """
    mle = MeanSquaredLogError()
    return mle(preds=prediction, target=target)


def _mean_absolute_error(
    target: torch.Tensor,
    prediction: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the mean absolute error between the target and prediction.

    Args:
        target (torch.Tensor): The target tensor.
        prediction (torch.Tensor): The prediction tensor.
    """
    mae = MeanAbsoluteError()
    return mae(preds=prediction, target=target)


### SIMPLIFIED FROM: https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/cli/generate_metrics.py
# the problem_type == "synthesis" branch


def _normalize_with_percentiles(
    input_tensor,
    normalization_tensor=None,
    p_min=0.5,
    p_max=99.5,
    strictly_positive=True,
):
    """Normalizes a tensor based on percentiles. Clips values below and above the percentile.
    Percentiles for normalization can come from another tensor.

    Args:
        input_tensor (torch.Tensor): Tensor to be normalized based on the data from the normalization_tensor.
            If normalization_tensor is None, the percentiles from this tensor will be used.
        normalization_tensor (torch.Tensor, optional): The tensor used for obtaining the percentiles.
        p_min (float, optional): Lower end percentile. Defaults to 0.5.
        p_max (float, optional): Upper end percentile. Defaults to 99.5.
        strictlyPositive (bool, optional): Ensures that really all values are above 0 before normalization. Defaults to True.

    Returns:
        torch.Tensor: The input_tensor normalized based on the percentiles of the reference tensor.
    """
    normalization_tensor = (
        input_tensor if normalization_tensor is None else normalization_tensor
    )
    v_min, v_max = np.percentile(
        normalization_tensor, [p_min, p_max]
    )  # get p_min percentile and p_max percentile

    # set lower bound to be 0 if strictlyPositive is enabled
    v_min = max(v_min, 0.0) if strictly_positive else v_min
    output_tensor = np.clip(
        input_tensor, v_min, v_max
    )  # clip values to percentiles from reference_tensor
    output_tensor = (output_tensor - v_min) / (
        v_max - v_min
    )  # normalizes values to [0;1]
    return output_tensor


def generate_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    normalization_tensor: torch.Tensor,  # for normalization see documentation
    mask: torch.Tensor,  # mask to compute the metrics in, in the 2023 = healthy_mask
):
    """
    TODO: Docstring :D

     Expected tensor shape: [1, 240, 240, 155]

        normalization_tensor (torch.Tensor, optional): The tensor used for obtaining the percentiles for normalization.
            You should provide the same image the inference models sees. So t1n_voided = t1n*mask.
            Note: we discussed whether normalization would be more correct if we omit the artificially introduced
            background voxels generated by masking with t1n*mask. To not have these artificial background voxels
            do t1n[mask] instead!

    """

    # Get Infill region (we really are only interested in the infill region)

    output_infill = (prediction * mask).float()
    gt_image_infill = (target * mask).float()

    # Normalize to [0;1] based on GT (otherwise MSE will depend on the image intensity range)
    # use all the tissue that is not masked for normalization
    gt_image_infill = _normalize_with_percentiles(
        gt_image_infill,
        normalization_tensor=normalization_tensor,
        p_min=0.5,
        p_max=99.5,
        strictly_positive=True,
    )
    output_infill = _normalize_with_percentiles(
        output_infill,
        normalization_tensor=normalization_tensor,
        p_min=0.5,
        p_max=99.5,
        strictly_positive=True,
    )

    output = {}

    output["ssim"] = _structural_similarity_index(
        target=gt_image_infill,
        prediction=output_infill,
        mask=mask,
    ).item()

    # only voxels that are to be inferred (-> flat array)
    # these are required for mse, psnr, etc.
    gt_image_infill_masked = gt_image_infill[mask]
    output_infill_masked = output_infill[mask]

    output["mse"] = _mean_squared_error(
        target=gt_image_infill_masked,
        prediction=output_infill_masked,
        squared=True,
    ).item()

    output["rmse"] = _mean_squared_error(
        target=gt_image_infill_masked,
        prediction=output_infill_masked,
        squared=False,
    ).item()

    output["msle"] = _mean_squared_log_error(
        target=gt_image_infill_masked,
        prediction=output_infill_masked,
    ).item()

    output["mae"] = _mean_absolute_error(
        target=gt_image_infill_masked,
        prediction=output_infill_masked,
    ).item()

    # torchmetrics PSNR using "max"
    output["psnr"] = _peak_signal_noise_ratio(
        target=gt_image_infill_masked,
        prediction=output_infill_masked,
    ).item()

    # same as above but with epsilon for robustness
    output["psnr_eps"] = _peak_signal_noise_ratio(
        target=gt_image_infill_masked,
        prediction=output_infill_masked,
        epsilon=sys.float_info.epsilon,
    ).item()

    # torchmetrics PSNR but with fixed data range of 0 to 1
    output["psnr_01"] = _peak_signal_noise_ratio(
        target=gt_image_infill_masked,
        prediction=output_infill_masked,
        data_range=(0, 1),
    ).item()

    # same as above but with epsilon for robustness
    output["psnr_01_eps"] = _peak_signal_noise_ratio(
        target=gt_image_infill_masked,
        prediction=output_infill_masked,
        data_range=(0, 1),
        epsilon=sys.float_info.epsilon,
    ).item()

    return output


def compute_image_quality_metrics(
    prediction: str,
    healthy_mask: str,
    reference_t1: str,
    voided_t1: str,
) -> dict:
    print("computing metrics!")
    print("prediction:", prediction)
    print("healthy_mask:", healthy_mask)
    print("reference_t1:", reference_t1)
    print("voided_t1:", voided_t1)

    prediction_data = read_nifti_to_tensor(prediction)
    healthy_mask_data = read_nifti_to_tensor(healthy_mask).bool()
    reference_t1_data = read_nifti_to_tensor(reference_t1)
    voided_t1_data = read_nifti_to_tensor(voided_t1)

    metrics = generate_metrics(
        prediction=prediction_data,
        target=reference_t1_data,
        normalization_tensor=voided_t1_data,
        mask=healthy_mask_data,
    )

    return metrics
