// Enum of node types
const nodeType = {
  INPUT: 'input',
  CONV: 'conv',
  POOL: 'pool',
  RELU: 'relu',
  FC: 'fc',
  FLATTEN: 'flatten'
}

class Node {
  /**
   * 每个神经元节点的类结构。
   * 
   * @param {string} layerName 节点所在层的名称。
   * @param {int} index 此节点在其层中的索引。
   * @param {string} type 节点类型 {input, conv, pool, relu, fc}。 
   * @param {number} bias 与此节点相关的偏差。
   * @param {[[number]]} output 此节点的输出。
   */
  constructor(layerName, index, type, bias, output) {
    this.layerName = layerName; // 节点所在层的名称
    this.index = index; // 此节点在其层中的索引
    this.type = type; // 节点类型
    this.bias = bias; // 与此节点相关的偏差
    this.output = output; // 此节点的输出

    // 权重存储在链接中
    this.inputLinks = []; // 输入链接
    this.outputLinks = []; // 输出链接
  }
}

/**  
 * 构造函数，用于初始化神经网络中的连接对象。
 * 
 * @param {Object} source - 连接的起始节点。
 * @param {Object} dest - 连接的目标节点。
 * @param {number} weight - 连接的权重。
 * 
 * @class Connection 表示神经网络中两个节点之间的连接。
 * @classdesc Connection 类定义了神经网络中节点之间的连接，包括起始节点、目标节点和连接权重。
 */
class Link {
  constructor(source, dest, weight) {
    this.source = source;
    this.dest = dest;
    this.weight = weight;
  }
}

/**
 * 
 * @param {*} nnJSON 神经网络的JSON表示
 * @param {*} inputImageArray 输入图像数组
 * @returns 返回构建的神经网络
 */
const constructNNFromJSON = (nnJSON, inputImageArray) => {
  console.log(nnJSON);
  console.log(inputImageArray);
  let nn = [];  // 初始化神经网络

  // 添加第一层（输入层）
  let inputLayer = [];
  let inputShape = nnJSON[0].input_shape;

  // 第一层的三个节点的输出是inputImageArray的通道
  for (let i = 0; i < inputShape[2]; i++) {
    let node = new Node('input', i, nodeType.INPUT, 0, inputImageArray[i]);
    inputLayer.push(node);
  }

  nn.push(inputLayer);
  let curLayerIndex = 1;

  // 遍历nnJSON中的每一层
  nnJSON.forEach(layer => {
    let curLayerNodes = [];
    let curLayerType;

    // 确定当前层的类型
    if (layer.name.includes('conv')) {
      curLayerType = nodeType.CONV;
    } else if (layer.name.includes('pool')) {
      curLayerType = nodeType.POOL;
    } else if (layer.name.includes('relu')) {
      curLayerType = nodeType.RELU;
    } else if (layer.name.includes('output')) {
      curLayerType = nodeType.FC;
    } else if (layer.name.includes('flatten')) {
      curLayerType = nodeType.FLATTEN;
    } else {
      console.log('Find unknown type');
    }

    let shape = layer.output_shape.slice(0, 2);
    let bias = 0;
    let output;
    // 如果当前层是FLATTEN或FC，输出为0，否则初始化为二维数组
    if (curLayerType === nodeType.FLATTEN || curLayerType === nodeType.FC) {
      output = 0;
    } else {
      output = init2DArray(shape[0], shape[1], 0);
    }

    // 将神经元添加到这一层
    for (let i = 0; i < layer.num_neurons; i++) {
      if (curLayerType === nodeType.CONV || curLayerType === nodeType.FC) {
        bias = layer.weights[i].bias;
      }
      let node = new Node(layer.name, i, curLayerType, bias, output)

      // 将此节点连接到所有前面的节点（创建链接）
      if (curLayerType === nodeType.CONV || curLayerType === nodeType.FC) {
        // CONV和FC层在链接中有权重。链接是一对多的
        for (let j = 0; j < nn[curLayerIndex - 1].length; j++) {
          let preNode = nn[curLayerIndex - 1][j];
          let curLink = new Link(preNode, node, layer.weights[i].weights[j]);
          preNode.outputLinks.push(curLink);
          node.inputLinks.push(curLink);
        }
      } else if (curLayerType === nodeType.RELU || curLayerType === nodeType.POOL) {
        // RELU和POOL层没有权重。链接是一对一的
        let preNode = nn[curLayerIndex - 1][i];
        let link = new Link(preNode, node, null);
        preNode.outputLinks.push(link);
        node.inputLinks.push(link);
      } else if (curLayerType === nodeType.FLATTEN) {
        // Flatten层没有权重。链接是多对一的。
        // 使用虚拟权重存储前一个节点中的对应条目（行，列）
        // tf2.keras中的flatten()的顺序是：通道 -> 行 -> 列
        let preNodeWidth = nn[curLayerIndex - 1][0].output.length,
          preNodeNum = nn[curLayerIndex - 1].length,
          preNodeIndex = i % preNodeNum,
          preNodeRow = Math.floor(Math.floor(i / preNodeNum) / preNodeWidth),
          preNodeCol = Math.floor(i / preNodeNum) % preNodeWidth,
          link = new Link(nn[curLayerIndex - 1][preNodeIndex],
            node, [preNodeRow, preNodeCol]);

        nn[curLayerIndex - 1][preNodeIndex].outputLinks.push(link);
        node.inputLinks.push(link);
      }
      curLayerNodes.push(node);
    }

    // 将当前层添加到神经网络
    nn.push(curLayerNodes);
    curLayerIndex++;
  });

  return nn;  // 返回构建的神经网络
}

export const constructNN = (inputImageFile) => {
  // 加载保存的模型文件
  return new Promise((resolve, reject) => {
    // 从指定URL获取JSON数据
    fetch('PUBLIC_URL/assets/data/nn_10.json')
      .then(response => {
        // 将响应内容解析为JSON
        response.json().then(nnJSON => {
          // 获取输入图像的数组
          getInputImageArray(inputImageFile)
            .then(inputImageArray => {
              // 从JSON和输入图像数组构建神经网络
              let nn = constructNNFromJSON(nnJSON, inputImageArray);
              // 将构建的神经网络作为Promise的解析结果
              resolve(nn);
            })
        });
      })
      // 如果在上述过程中出现错误，将错误作为Promise的拒绝原因
      .catch(error => {
        reject(error);
      });
  });
}

// Helper functions

/**
 * 根据给定的大小和默认值创建一个二维数组（矩阵）。
 * 
 * @param {int} height 矩阵的高度（行数）
 * @param {int} width 矩阵的宽度（列数）
 * @param {int} fill 用于填充此矩阵的默认值
 */
export const init2DArray = (height, width, fill) => {
  let array = [];
  // 遍历行
  for (let r = 0; r < height; r++) {
    let row = new Array(width).fill(fill);
    array.push(row);
  }
  return array;
}

/**
 * 两个矩阵的点积。
 * @param {[[number]]} mat1 矩阵1
 * @param {[[number]]} mat2 矩阵2
 */
const matrixDot = (mat1, mat2) => {
  // 断言两个矩阵的维度匹配
  console.assert(mat1.length === mat2.length, '维度不匹配');
  console.assert(mat1[0].length === mat2[0].length, '维度不匹配');

  let result = 0;
  // 遍历矩阵的每个元素
  for (let i = 0; i < mat1.length; i++) {
    for (let j = 0; j < mat1[0].length; j++) {
      // 计算点积
      result += mat1[i][j] * mat2[i][j];
    }
  }

  return result;
}

/**
 * 矩阵元素逐个相加。
 * @param {[[number]]} mat1 矩阵1
 * @param {[[number]]} mat2 矩阵2
 */
export const matrixAdd = (mat1, mat2) => {
  // 断言两个矩阵的维度匹配
  console.assert(mat1.length === mat2.length, '维度不匹配');
  console.assert(mat1[0].length === mat2[0].length, '维度不匹配');

  // 初始化一个与输入矩阵同维度的结果矩阵
  let result = init2DArray(mat1.length, mat1.length, 0);

  // 遍历矩阵的每个元素
  for (let i = 0; i < mat1.length; i++) {
    for (let j = 0; j < mat1.length; j++) {
      // 计算元素逐个相加的结果
      result[i][j] = mat1[i][j] + mat2[i][j];
    }
  }

  return result;
}

/**
 * 对矩阵进行二维切片。
 * @param {[[number]]} mat 矩阵
 * @param {int} xs 第一维度（行）的起始索引
 * @param {int} xe 第一维度（行）的结束索引
 * @param {int} ys 第二维度（列）的起始索引
 * @param {int} ye 第二维度（列）的结束索引
 */
export const matrixSlice = (mat, xs, xe, ys, ye) => {
  // 对矩阵进行切片操作
  return mat.slice(xs, xe).map(s => s.slice(ys, ye));
}

/**
 * 计算矩阵的最大值。
 * @param {[[number]]} mat 矩阵
 */
const matrixMax = (mat) => {
  // 初始化当前最大值为负无穷大
  let curMax = -Infinity;
  // 遍历矩阵的每个元素
  for (let i = 0; i < mat.length; i++) {
    for (let j = 0; j < mat[0].length; j++) {
      // 如果当前元素大于当前最大值，则更新最大值
      if (mat[i][j] > curMax) {
        curMax = mat[i][j];
      }
    }
  }
  // 返回最大值
  return curMax;
}

/**
 * 将画布图像数据转换为维度为[height, width, 3]的3D数组。
 * 每个像素的范围是0-255。
 * @param {[int8]} imageData 画布图像数据
 */
const imageDataTo3DArray = (imageData) => {
  // 获取图像维度（假设为正方形图像）
  let width = Math.sqrt(imageData.length / 4);

  // 为每个通道创建数组占位符
  let imageArray = [init2DArray(width, width, 0), init2DArray(width, width, 0),
  init2DArray(width, width, 0)];

  // 遍历数据以填充上述通道数组
  for (let i = 0; i < imageData.length; i++) {
    let pixelIndex = Math.floor(i / 4),
      channelIndex = i % 4,
      row = Math.floor(pixelIndex / width),
      column = pixelIndex % width;

    // 如果通道索引小于3，则填充对应的像素值
    if (channelIndex < 3) {
      imageArray[channelIndex][row][column] = imageData[i];
    }
  }

  return imageArray;
}

/**
 * 获取给定图像文件的3D像素值数组。
 * @param {string} imgFile 图像文件的文件路径
 * @returns 返回一个Promise，其值为对应的3D数组
 */
const getInputImageArray = (imgFile) => {
  // 创建一个新的canvas元素
  let canvas = document.createElement('canvas');
  // 设置canvas样式为不显示
  canvas.style.cssText = 'display:none;';
  // 将canvas元素添加到body中
  document.getElementsByTagName('body')[0].appendChild(canvas);
  // 获取canvas的2D渲染上下文
  let context = canvas.getContext('2d');

  // 返回一个新的Promise
  return new Promise((resolve, reject) => {
    // 创建一个新的Image对象
    let inputImage = new Image();
    // 设置Image对象的源为图像文件路径
    inputImage.src = imgFile;
    // 当图像加载完成时
    inputImage.onload = () => {
      // 在canvas上绘制图像
      context.drawImage(inputImage, 0, 0,);
      // 获取图像数据并将其转换为3D数组
      let imageData = context.getImageData(0, 0, inputImage.width,
        inputImage.height).data;

      // 移除新创建的canvas元素
      canvas.parentNode.removeChild(canvas);

      // 在控制台打印3D数组
      console.log(imageDataTo3DArray(imageData));
      // 将3D数组作为Promise的解析值
      resolve(imageDataTo3DArray(imageData));
    }
    // 如果图像加载出错，拒绝Promise
    inputImage.onerror = reject;
  })
}

/**
 * Compute convolutions of one kernel on one matrix (one slice of a tensor).
 * @param {[[number]]} input Input, square matrix
 * @param {[[number]]} kernel Kernel weights, square matrix
 * @param {int} stride Stride size
 * @param {int} padding Padding size
 */
export const singleConv = (input, kernel, stride = 1, padding = 0) => {
  // TODO: implement padding

  // Only support square input and kernel
  console.assert(input.length === input[0].length,
    'Conv input is not square');
  console.assert(kernel.length === kernel[0].length,
    'Conv kernel is not square');

  let stepSize = (input.length - kernel.length) / stride + 1;

  let result = init2DArray(stepSize, stepSize, 0);

  // Window sliding
  for (let r = 0; r < stepSize; r++) {
    for (let c = 0; c < stepSize; c++) {
      let curWindow = matrixSlice(input, r * stride, r * stride + kernel.length,
        c * stride, c * stride + kernel.length);
      let dot = matrixDot(curWindow, kernel);
      result[r][c] = dot;
    }
  }
  return result;
}

/**
 * Convolution operation. This function update the outputs property of all nodes
 * in the given layer. Previous layer is accessed by the reference in nodes'
 * links.
 * @param {[Node]} curLayer Conv layer.
 */
const convolute = (curLayer) => {
  console.assert(curLayer[0].type === 'conv', 'Wrong layer type');

  // Itereate through all nodes in curLayer to update their outputs
  curLayer.forEach(node => {
    /*
     * Accumulate the single conv result matrices from previous channels.
     * Previous channels (node) are accessed by the reference in Link objects.
     */
    let newOutput = init2DArray(node.output.length, node.output.length, 0);

    for (let i = 0; i < node.inputLinks.length; i++) {
      let curLink = node.inputLinks[i];
      let curConvResult = singleConv(curLink.source.output, curLink.weight);
      newOutput = matrixAdd(newOutput, curConvResult);
    }

    // Add bias to all element in the output
    let biasMatrix = init2DArray(newOutput.length, newOutput.length, node.bias);
    newOutput = matrixAdd(newOutput, biasMatrix);

    node.output = newOutput;
  })
}

/**
 * Activate matrix mat using ReLU (max(0, x)).
 * @param {[[number]]} mat Matrix
 */
const singleRelu = (mat) => {
  // Only support square matrix
  console.assert(mat.length === mat[0].length, 'Activating non-square matrix!');

  let width = mat.length;
  let result = init2DArray(width, width, 0);

  for (let i = 0; i < width; i++) {
    for (let j = 0; j < width; j++) {
      result[i][j] = Math.max(0, mat[i][j]);
    }
  }
  return result;
}

/**
 * Update outputs of all nodes in the current ReLU layer. Values of previous
 * layer nodes are accessed by the links stored in the current layer.
 * @param {[Node]} curLayer ReLU layer
 */
const relu = (curLayer) => {
  console.assert(curLayer[0].type === 'relu', 'Wrong layer type');

  // Itereate through all nodes in curLayer to update their outputs
  for (let i = 0; i < curLayer.length; i++) {
    let curNode = curLayer[i];
    let preNode = curNode.inputLinks[0].source;
    curNode.output = singleRelu(preNode.output);
  }
}

/**
 * Max pool one matrix.
 * @param {[[number]]} mat Matrix
 * @param {int} kernelWidth Pooling kernel length (only supports 2)
 * @param {int} stride Pooling sliding stride (only supports 2)
 * @param {string} padding Pading method when encountering odd number mat,
 * currently this function only supports 'VALID'
 */
export const singleMaxPooling = (mat, kernelWidth = 2, stride = 2, padding = 'VALID') => {
  console.assert(kernelWidth === 2, 'Only supports kernen = [2,2]');
  console.assert(stride === 2, 'Only supports stride = 2');
  console.assert(padding === 'VALID', 'Only support valid padding');

  // Handle odd length mat
  // 'VALID': ignore edge rows and columns
  // 'SAME': add zero padding to make the mat have even length
  if (mat.length % 2 === 1 && padding === 'VALID') {
    mat = matrixSlice(mat, 0, mat.length - 1, 0, mat.length - 1);
  }

  let stepSize = (mat.length - kernelWidth) / stride + 1;
  let result = init2DArray(stepSize, stepSize, 0);

  for (let r = 0; r < stepSize; r++) {
    for (let c = 0; c < stepSize; c++) {
      let curWindow = matrixSlice(mat, r * stride, r * stride + kernelWidth,
        c * stride, c * stride + kernelWidth);
      result[r][c] = matrixMax(curWindow);
    }
  }
  return result;
}

/**
 * Max pooling one layer.
 * @param {[Node]} curLayer MaxPool layer
 */
const maxPooling = (curLayer) => {
  console.assert(curLayer[0].type === 'pool', 'Wrong layer type');

  // Itereate through all nodes in curLayer to update their outputs
  for (let i = 0; i < curLayer.length; i++) {
    let curNode = curLayer[i];
    let preNode = curNode.inputLinks[0].source;
    curNode.output = singleMaxPooling(preNode.output);
  }
}

/**
 * Flatten a previous 2D layer (conv2d or maxpool2d). The flatten order matches
 * tf2.keras' implementation: channel -> row -> column.
 * @param {[Node]} curLayer Flatten layer
 */
const flatten = (curLayer) => {
  console.assert(curLayer[0].type === 'flatten', 'Wrong layer type');

  // Itereate through all nodes in curLayer to update their outputs
  for (let i = 0; i < curLayer.length; i++) {
    let curNode = curLayer[i];
    let preNode = curNode.inputLinks[0].source;
    let coordinate = curNode.inputLinks[0].weight;
    // Take advantage of the dummy weights
    curNode.output = preNode.output[coordinate[0]][coordinate[1]];
  }
}

const fullyConnect = (curLayer) => {
  console.assert(curLayer[0].type === 'fc', 'Wrong layer type');
  // TODO
}

export const tempMain = async () => {
  let nn = await constructNN('PUBLIC_URL/assets/img/koala.jpeg');
  convolute(nn[1]);
  relu(nn[2])
  convolute(nn[3]);
  relu(nn[4]);
  maxPooling(nn[5]);
  convolute(nn[6]);
  relu(nn[7])
  convolute(nn[8]);
  relu(nn[9]);
  maxPooling(nn[10]);
  convolute(nn[11]);
  relu(nn[12])
  convolute(nn[13]);
  relu(nn[14]);
  maxPooling(nn[15]);
  flatten(nn[16]);
  console.log(nn[16].map(d => d.output));
}