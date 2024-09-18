import { matrixSlice } from '../utils/cnn.js';

export function array1d(length, f) {
  return Array.from({length: length}, f ? ((v, i) => f(i)) : undefined);
}

function array2d(height, width, f) {
  return Array.from({length: height}, (v, i) => Array.from({length: width}, f ? ((w, j) => f(i, j)) : undefined));
}

export function generateOutputMappings(stride, output, kernelLength, padded_input_size, dilation) {
  const outputMapping = array2d(output.length, output.length, (i, j) => array2d(kernelLength, kernelLength));
  for (let h_out = 0; h_out < output.length; h_out++) {
    for (let w_out = 0; w_out < output.length; w_out++) {
      for (let h_kern = 0; h_kern < kernelLength; h_kern++) {
        for (let w_kern = 0; w_kern < kernelLength; w_kern++) {
          const h_im = h_out * stride + h_kern * dilation;
          const w_im = w_out * stride + w_kern * dilation;
          outputMapping[h_out][w_out][h_kern][w_kern] = h_im * padded_input_size + w_im;
        }
      }
    }
  }
  return outputMapping;
}

export function compute_input_multiplies_with_weight(hoverH, hoverW, 
                                              padded_input_size, weight_dims, outputMappings, kernelLength) {
  
  const [h_weight, w_weight] = weight_dims;
  const input_multiplies_with_weight = array1d(padded_input_size * padded_input_size);
  for (let h_weight = 0; h_weight < kernelLength; h_weight++) {
    for (let w_weight = 0; w_weight < kernelLength; w_weight++) {
      const flat_input = outputMappings[hoverH][hoverW][h_weight][w_weight];
      if (typeof flat_input === "undefined") continue;
      input_multiplies_with_weight[flat_input] = [h_weight, w_weight];
    }
  }
  return input_multiplies_with_weight;
}

export function getMatrixSliceFromInputHighlights(matrix, highlights, kernelLength) {
  var indices = highlights.reduce((total, value, index) => {
  if (value != undefined) total.push(index);
    return total;
  }, []);
  return matrixSlice(matrix, Math.floor(indices[0] / matrix.length), Math.floor(indices[0] / matrix.length) + kernelLength, indices[0] % matrix.length, indices[0] % matrix.length + kernelLength);
}

export function getMatrixSliceFromOutputHighlights(matrix, highlights) {
  var indices = highlights.reduce((total, value, index) => {
  if (value != false) total.push(index);
    return total;
  }, []);
  return matrixSlice(matrix, Math.floor(indices[0] / matrix.length), Math.floor(indices[0] / matrix.length) + 1, indices[0] % matrix.length, indices[0] % matrix.length + 1);
}

/**
 * 
 * @param {*} imageLength 
 * @returns 
 */
// Edit these values to change size of low-level conv visualization.
export function getVisualizationSizeConstraint(imageLength) {
  let sizeOfGrid = 150;
  let maxSizeOfGridCell = 20;
  return sizeOfGrid / imageLength > maxSizeOfGridCell ? maxSizeOfGridCell : sizeOfGrid / imageLength;
}

/**
 * 获取图像数据的范围。
 * 
 * @param {*} image 输入图像
 * @returns 图像数据的范围，包括最小值、最大值和范围
 */
export function getDataRange(image) {
  let maxRow = image.map(function(row){ return Math.max.apply(Math, row); }); // 获取每行的最大值
  let max = Math.max.apply(null, maxRow); // 获取图像的最大值
  let minRow = image.map(function(row){ return Math.min.apply(Math, row); }); // 获取每行的最小值
  let min = Math.min.apply(null, minRow); // 获取图像的最小值
  let range = {
    range: 2 * Math.max(Math.abs(min), Math.abs(max)), // 计算范围
    min: min, // 最小值
    max: max // 最大值
  };
  return range; // 返回范围对象
}

/**
 * 根据输入图像大小约束网格。
 * 
 * @param {*} image 输入图像
 * @param {*} constraint 约束值
 * @returns 生成的网格数据
 */
export function gridData(image, constraint = getVisualizationSizeConstraint(image.length)) {
  // 根据输入图像大小约束网格。
  var data = new Array(); // 初始化数据数组
  var xpos = 1; // 初始化x坐标位置
  var ypos = 1; // 初始化y坐标位置
  var width = constraint; // 设置网格宽度为约束值
  var height = constraint; // 设置网格高度为约束值

  // 遍历图像的每一行
  for (var row = 0; row < image.length; row++) {
    data.push(new Array()); // 为每一行添加一个新数组

    // 遍历图像的每一列
    for (var column = 0; column < image[0].length; column++) {
      // 将当前像素的信息添加到数据数组中
      data[row].push({
        text: Math.round(image[row][column] * 100) / 100, // 将像素值四舍五入到小数点后两位
        row: row, // 当前行索引
        col: column, // 当前列索引
        x: xpos, // 当前x坐标位置
        y: ypos, // 当前y坐标位置
        width: width, // 网格宽度
        height: height // 网格高度
      });
      xpos += width; // 更新x坐标位置
    }
    xpos = 1; // 重置x坐标位置
    ypos += height; // 更新y坐标位置
  }
  return data; // 返回生成的网格数据
}