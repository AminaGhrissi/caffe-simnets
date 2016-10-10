#ifndef _CAFFE_UTIL_IM2COL_HPP_
#define _CAFFE_UTIL_IM2COL_HPP_

#include <cmath>

namespace caffe {

template <typename Dtype>
void im2col_nd_cpu(const Dtype* data_im, const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, Dtype* data_col);

template <typename Dtype>
void im2col_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    Dtype* data_col);

template <typename Dtype>
void col2im_nd_cpu(const Dtype* data_col, const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, Dtype* data_im);

template <typename Dtype>
void col2im_cpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    Dtype* data_im);

template <typename Dtype>
void im2col_nd_gpu(const Dtype* data_im, const int num_spatial_axes,
    const int col_size, const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, Dtype* data_col);

template <typename Dtype>
void im2col_gpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    Dtype* data_col);

template <typename Dtype>
void col2im_nd_gpu(const Dtype* data_col, const int num_spatial_axes,
    const int im_size, const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, Dtype* data_im);

template <typename Dtype>
void col2im_gpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    Dtype* data_im);

/**
 * @brief A 3D variation of im2col. Arranges 3D patches in column matrix.
 * @details A 3D variation of im2col. Arrange 3D patches in column matrix.
 *          More specifically, it actually creates a 4D array where the 3D indexes
 *          specify where the patch originated from the original image, and the
 *          forth dimension specifies the patch data.
 *
 * @param data_im The incoming image array with multiple channels.
 * @param channels Number of channels of the input image.
 * @param height The height of the input image.
 * @param width The width of the input image.
 * @param patch_c The size of the patch in the channels dimension.
 * @param patch_h The height of the patch.
 * @param patch_w The width of the patch.
 * @param pad_c The padding along the channels dimension.
 * @param pad_h The padding along the vertical dimension.
 * @param pad_w The padding along the horizontal dimension.
 * @param stride_c The stride along the channels dimension.
 * @param stride_h The stride along the vertical dimension.
 * @param stride_w The stride along the horizontal dimension.
 * @param data_col The output column matrix.
 */
template <typename Dtype>
void im2col_3d_cpu(const Dtype* data_im,
                   const int channels, const int height, const int width,
                   const int patch_c, const int patch_h, const int patch_w,
                   const int pad_c, const int pad_h, const int pad_w,
                   const int stride_c, const int stride_h, const int stride_w,
                   Dtype* data_col,
                   const bool round_down = true, const Dtype out_of_bounds_value = 0);
template <typename Dtype>
void im2col_3d_gpu(const Dtype* data_im,
                   const int channels, const int height, const int width,
                   const int patch_c, const int patch_h, const int patch_w,
                   const int pad_c, const int pad_h, const int pad_w,
                   const int stride_c, const int stride_h, const int stride_w,
                   Dtype* data_col,
                   const bool round_down = true, const Dtype out_of_bounds_value = 0);
/**
 * @brief A 3D variation of col2im. Sums of the column matrix back to original image.
 * @details A 3D variation of col2im. Sums of the column matrix back to original image.
 *          More specifically, it actually reads from a 4D array where the 3D indexes
 *          specify where the patch originated from the original image, and the
 *          forth dimension specifies the patch data.
 *
 * @param data_col The input column matrix.
 * @param channels Number of channels of the input image.
 * @param height The height of the input image.
 * @param width The width of the input image.
 * @param patch_c The size of the patch in the channels dimension.
 * @param patch_h The height of the patch.
 * @param patch_w The width of the patch.
 * @param pad_c The padding along the channels dimension.
 * @param pad_h The padding along the vertical dimension.
 * @param pad_w The padding along the horizontal dimension.
 * @param stride_c The stride along the channels dimension.
 * @param stride_h The stride along the vertical dimension.
 * @param stride_w The stride along the horizontal dimension.
 * @param data_im The output matrix.
 */
template <typename Dtype>
void col2im_3d_cpu(const Dtype* data_col,
                   const int channels, const int height, const int width,
                   const int patch_c, const int patch_h, const int patch_w,
                   const int pad_c, const int pad_h, const int pad_w,
                   const int stride_c, const int stride_h, const int stride_w,
                   Dtype* data_im, const bool round_down = true);

template <typename Dtype>
void col2im_3d_gpu(const Dtype* data_col,
                   const int channels, const int height, const int width, 
                   const int patch_c, const int patch_h, const int patch_w,
                   const int pad_c, const int pad_h, const int pad_w,
                   const int stride_c, const int stride_h, const int stride_w,
                   Dtype* data_im, const bool round_down = true);

/**
 * A helper function to calculate the output dimension's size give the original
 * size of the image, padding, patch size and stride.
 * @param  image_size The size of the dimension in the original image
 * @param  pad_size   The amount of padding to apply to the original image
 * @param  patch_size The size of the dimension in the patch taken from the image
 * @param  stride     The patch's stride over the original image
 * @param  round_down Whether to round down or up when calculating the size
 * @return            The output size of the patch image
 * @remarks round_down can be used to control pooling/conv style im2col method.
 */
inline int dimension_out_size(const int image_size, const int pad_size, const int patch_size,
                              const int stride, const bool round_down) {
  if (round_down) {
    return (image_size + 2 * pad_size - patch_size) / stride + 1;
  } else {
    return static_cast<int>(std::ceil(static_cast<float>(image_size + 2 * pad_size - patch_size) / stride)) + 1;
  }
}
}  // namespace caffe

#endif  // CAFFE_UTIL_IM2COL_HPP_
