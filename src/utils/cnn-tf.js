/* global tf */

// Network input image size
const networkInputSize = 64;

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
   * @param {int} index 节点在其层中的索引。
   * @param {string} type 节点类型 {input, conv, pool, relu, fc}。
   * @param {number} bias 与此节点关联的偏置。
   * @param {number[]} output 此节点的输出。
   */
  constructor(layerName, index, type, bias, output) {
    this.layerName = layerName;
    this.index = index;
    this.type = type;
    this.bias = bias;
    this.output = output;

    // 权重存储在链接中
    this.inputLinks = [];
    this.outputLinks = [];
  }
}

class Link {
  /**
   * 每两个节点之间链接的类结构。
   * 
   * @param {Node} source 源节点。
   * @param {Node} dest 目标节点。
   * @param {number} weight 与此链接关联的权重。它可以是一个数字，
   *  一维数组或二维数组。
   */
  constructor(source, dest, weight) {
    this.source = source;
    this.dest = dest;
    this.weight = weight;
  }
}

/**
 * 使用每一层提取的输出构建一个CNN。
 * 
 * @param {number[][]} allOutputs 每层输出的数组。
 *  allOutputs[i][j] 是第i层节点j的输出。
 * @param {Model} model 加载的tf.js模型。
 * @param {Tensor} inputImageTensor 加载的输入图像张量。
 */
const constructCNNFromOutputs = (allOutputs, model, inputImageTensor) => {
  let cnn = [];

  // 添加第一层（输入层）
  let inputLayer = [];
  let inputShape = model.layers[0].batchInputShape.slice(1);
  let inputImageArray = inputImageTensor.transpose([2, 0, 1]).arraySync();

  // 第一层的三个节点的输出是inputImageArray的通道
  for (let i = 0; i < inputShape[2]; i++) {
    let node = new Node('input', i, nodeType.INPUT, 0, inputImageArray[i]);
    inputLayer.push(node);
  }
                                                                                                                   
  cnn.push(inputLayer);
  let curLayerIndex = 1;

  for (let l = 0; l < model.layers.length; l++) {
    let layer = model.layers[l];
    // 获取当前输出
    let outputs = allOutputs[l].squeeze();
    outputs = outputs.arraySync();

    let curLayerNodes = [];
    let curLayerType;

    // 根据层名称识别层类型
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
      console.log('发现未知类型');
    }

    // 根据层类型构建此层
    switch (curLayerType) {
      case nodeType.CONV: {
        let biases = layer.bias.val.arraySync();
        // 新顺序为 [output_depth, input_depth, height, width]
        let weights = layer.kernel.val.transpose([3, 2, 0, 1]).arraySync();

        // 将节点添加到此层
        for (let i = 0; i < outputs.length; i++) {
          let node = new Node(layer.name, i, curLayerType, biases[i],
            outputs[i]);

          // 将此节点连接到所有先前的节点（创建链接）
          // CONV层在链接中有权重。链接是多对一的。
          for (let j = 0; j < cnn[curLayerIndex - 1].length; j++) {
            let preNode = cnn[curLayerIndex - 1][j];
            let curLink = new Link(preNode, node, weights[i][j]);
            preNode.outputLinks.push(curLink);
            node.inputLinks.push(curLink);
          }
          curLayerNodes.push(node);
        }
        break;
      }
      case nodeType.FC: {
        let biases = layer.bias.val.arraySync();
        // 新顺序为 [output_depth, input_depth]
        let weights = layer.kernel.val.transpose([1, 0]).arraySync();

        // 将节点添加到此层
        for (let i = 0; i < outputs.length; i++) {
          let node = new Node(layer.name, i, curLayerType, biases[i],
            outputs[i]);

          // 将此节点连接到所有先前的节点（创建链接）
          // FC层在链接中有权重。链接是多对一的。

          // 由于我们正在可视化logit值，我们需要跟踪softmax之前的原始值
          let curLogit = 0;
          for (let j = 0; j < cnn[curLayerIndex - 1].length; j++) {
            let preNode = cnn[curLayerIndex - 1][j];
            let curLink = new Link(preNode, node, weights[i][j]);
            preNode.outputLinks.push(curLink);
            node.inputLinks.push(curLink);
            curLogit += preNode.output * weights[i][j];
          }
          curLogit += biases[i];
          node.logit = curLogit;
          curLayerNodes.push(node);
        }

        // 根据节点TF索引对展平层进行排序
        cnn[curLayerIndex - 1].sort((a, b) => a.realIndex - b.realIndex);
        break;
      }
      case nodeType.RELU:
      case nodeType.POOL: {
        // RELU和POOL没有偏置和权重
        let bias = 0;
        let weight = null;

        // 将节点添加到此层
        for (let i = 0; i < outputs.length; i++) {
          let node = new Node(layer.name, i, curLayerType, bias, outputs[i]);

          // RELU和POOL层没有权重。链接是一对一的
          let preNode = cnn[curLayerIndex - 1][i];
          let link = new Link(preNode, node, weight);
          preNode.outputLinks.push(link);
          node.inputLinks.push(link);

          curLayerNodes.push(node);
        }
        break;
      }
      case nodeType.FLATTEN: {
        // 展平层没有偏置和权重。
        let bias = 0;

        for (let i = 0; i < outputs.length; i++) {
          // 展平层没有权重。链接是多对一的。
          // 使用虚拟权重来存储前一个节点中的相应条目作为（行，列）
          // tf2.keras中的flatten()顺序为：通道 -> 行 -> 列
          let preNodeWidth = cnn[curLayerIndex - 1][0].output.length,
            preNodeNum = cnn[curLayerIndex - 1].length,
            preNodeIndex = i % preNodeNum,
            preNodeRow = Math.floor(Math.floor(i / preNodeNum) / preNodeWidth),
            preNodeCol = Math.floor(i / preNodeNum) % preNodeWidth,
            // 使用通道、行、列来计算实际索引，顺序为行 -> 列 -> 通道
            curNodeRealIndex = preNodeIndex * (preNodeWidth * preNodeWidth) +
              preNodeRow * preNodeWidth + preNodeCol;
          
          let node = new Node(layer.name, i, curLayerType,
              bias, outputs[i]);
          
          // TF使用（i）索引进行计算，但实际顺序应为（curNodeRealIndex）。
          // 我们将在输出层计算logit值后使用实际顺序对节点进行排序。
          node.realIndex = curNodeRealIndex;

          let link = new Link(cnn[curLayerIndex - 1][preNodeIndex],
              node, [preNodeRow, preNodeCol]);

          cnn[curLayerIndex - 1][preNodeIndex].outputLinks.push(link);
          node.inputLinks.push(link);

          curLayerNodes.push(node);
        }

        // 根据节点TF索引对展平层进行排序
        curLayerNodes.sort((a, b) => a.index - b.index);
        break;
      }
      default:
        console.error('遇到未知层类型');
        break;
    }

    // 将当前层添加到NN
    cnn.push(curLayerNodes);
    curLayerIndex++;
  }

  return cnn;
}

/**
 * 构建一个具有给定模型和输入的CNN。
 * 
 * @param {string} inputImageFile 输入图像的文件名。
 * @param {Model} model 加载的tf.js模型。
 */
export const constructCNN = async (inputImageFile, model) => {
  // 加载图像文件
  let inputImageTensor = await getInputImageArray(inputImageFile, true);

  // 需要以批次形式提供给模型
  let inputImageTensorBatch = tf.stack([inputImageTensor]);

  // 为了获取中间层的输出，我们将遍历模型中的所有层，并依次应用转换。
  let preTensor = inputImageTensorBatch;
  let outputs = [];

  // 遍历所有层，并构建一个以该层为输出的模型
  for (let l = 0; l < model.layers.length; l++) {
    let curTensor = model.layers[l].apply(preTensor);

    // 记录输出张量
    // 因为批次中只有一个元素，所以我们使用squeeze()
    // 我们还希望在这里使用CHW顺序
    let output = curTensor.squeeze();
    if (output.shape.length === 3) {
      output = output.transpose([2, 0, 1]);
    }
    outputs.push(output);

    // 更新preTensor以进行下一次嵌套迭代
    preTensor = curTensor;
  }

  let cnn = constructCNNFromOutputs(outputs, model, inputImageTensor);
  return cnn;
}


/**
 * 裁剪3D数组中大小为64x64x3的最大中心方块。
 * 
 * @param {[int8]} arr 需要裁剪和填充的数组（如果不存在64x64的裁剪）
 * @returns 64x64x3的数组
 */
const cropCentralSquare = (arr) => {
  let width = arr.length;
  let height = arr[0].length;
  let croppedArray;

  // 如果图像小于64x64，则裁剪图像的最大方块并填充裁剪后的图像。
  if (width < networkInputSize || height < networkInputSize) {
    // TODO(robert): 完成填充逻辑。现在推送给Omar，当他准备好时可以继续工作。
    let cropDimensions = Math.min(width, height);
    let startXIdx = Math.floor(width / 2) - (cropDimensions / 2);
    let startYIdx = Math.floor(height / 2) - (cropDimensions / 2);
    let unpaddedSubarray = arr.slice(startXIdx, startXIdx + cropDimensions).map(i => i.slice(startYIdx, startYIdx + cropDimensions));
  } else {
    let startXIdx = Math.floor(width / 2) - Math.floor(networkInputSize / 2);
    let startYIdx = Math.floor(height / 2) - Math.floor(networkInputSize / 2);
    croppedArray = arr.slice(startXIdx, startXIdx + networkInputSize).map(i => i.slice(startYIdx, startYIdx + networkInputSize));
  }
  return croppedArray;
}

/**
 * 将画布图像数据转换为维度为[height, width, 3]的3D张量。
 * 请记住，tensorflow使用NHWC顺序（批次，高度，宽度，通道）。
 * 每个像素的值在0-255范围内。
 * 
 * @param {[int8]} imageData 画布图像数据
 * @param {int} width 画布图像宽度
 * @param {int} height 画布图像高度
 */
const imageDataTo3DTensor = (imageData, width, height, normalize=true) => {
  // 创建3D数组的占位符
  let imageArray = tf.fill([width, height, 3], 0).arraySync();

  // 遍历数据以填充上述通道数组
  for (let i = 0; i < imageData.length; i++) {
    let pixelIndex = Math.floor(i / 4),
      channelIndex = i % 4,
      row = width === height ? Math.floor(pixelIndex / width)
                              : pixelIndex % width,
      column = width === height ? pixelIndex % width
                              : Math.floor(pixelIndex / width);
    
    if (channelIndex < 3) {
      let curEntry  = imageData[i];
      // 将原始像素值从[0, 255]归一化到[0, 1]
      if (normalize) {
        curEntry /= 255;
      }
      imageArray[row][column][channelIndex] = curEntry;
    }
  }

  // 如果图像不是64x64，则适当地裁剪或填充图像。
  if (width != networkInputSize && height != networkInputSize) {
    imageArray = cropCentralSquare(imageArray)
  }

  let tensor = tf.tensor3d(imageArray);
  return tensor;
}

/**
 * 获取给定图像文件的3D像素值数组。
 * 
 * @param {string} imgFile 图像文件的文件路径
 * @returns 返回包含相应3D数组的Promise
 */
const getInputImageArray = (imgFile, normalize=true) => {
  let canvas = document.createElement('canvas');
  canvas.style.cssText = 'display:none;';
  document.getElementsByTagName('body')[0].appendChild(canvas);
  let context = canvas.getContext('2d');

  return new Promise((resolve, reject) => {
    let inputImage = new Image();
    inputImage.crossOrigin = "Anonymous";
    inputImage.src = imgFile;
    let canvasImage;
    inputImage.onload = () => {
      canvas.width = inputImage.width;
      canvas.height = inputImage.height;
      // 如果输入图像太大，则调整网络的输入图像大小，只裁剪中心64x64部分，
      // 以便仍然提供代表性的输入图像到网络中。
      if (inputImage.width > networkInputSize || inputImage.height > networkInputSize) {
        // 第一步 - 使用较小的尺寸调整图像大小以缩小图像。
        let resizeCanvas = document.createElement('canvas'),
            resizeContext = resizeCanvas.getContext('2d');
        let smallerDimension = Math.min(inputImage.width, inputImage.height);
        const resizeFactor = (networkInputSize + 1) / smallerDimension;
        resizeCanvas.width = inputImage.width * resizeFactor;
        resizeCanvas.height = inputImage.height * resizeFactor;
        resizeContext.drawImage(inputImage, 0, 0, resizeCanvas.width,
          resizeCanvas.height);

        // 第二步 - 水平翻转非方形图像并将其旋转90度，因为非方形图像不是直立存储的。
        if (inputImage.width != inputImage.height) {
          context.translate(resizeCanvas.width, 0);
          context.scale(-1, 1);
          context.translate(resizeCanvas.width / 2, resizeCanvas.height / 2);
          context.rotate(90 * Math.PI / 180);
        }

        // 第三步 - 在原始画布上绘制调整大小的图像。
        if (inputImage.width != inputImage.height) {
          context.drawImage(resizeCanvas, -resizeCanvas.width / 2, -resizeCanvas.height / 2);
        } else {
          context.drawImage(resizeCanvas, 0, 0);
        }
        canvasImage = context.getImageData(0, 0, resizeCanvas.width,
          resizeCanvas.height);

      } else {
        context.drawImage(inputImage, 0, 0);
        canvasImage = context.getImageData(0, 0, inputImage.width,
          inputImage.height);
      }
      // 获取图像数据并将其转换为3D数组
      let imageData = canvasImage.data;
      let imageWidth = canvasImage.width;
      let imageHeight = canvasImage.height;

      // 移除这个新创建的画布元素
      canvas.parentNode.removeChild(canvas);

      resolve(imageDataTo3DTensor(imageData, imageWidth, imageHeight, normalize));
    }
    inputImage.onerror = reject;
  })
}

/**
 * 加载模型的包装器。
 * 
 * @param {string} modelFile 转换后的模型json文件名（通过tensorflowjs.py）。
 */
export const loadTrainedModel = (modelFile) => {
  return tf.loadLayersModel(modelFile);
}
