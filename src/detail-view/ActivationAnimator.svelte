<script>
  import { createEventDispatcher } from "svelte";
  import {
    array1d,
    getMatrixSliceFromOutputHighlights,
    getVisualizationSizeConstraint,
    getMatrixSliceFromInputHighlights,
    gridData,
  } from "./DetailviewUtils.js";
  import Dataview from "./Dataview.svelte";

  export let image;
  export let output;
  export let isPaused;
  export let dataRange;

  const dispatch = createEventDispatcher(); // 创建事件调度器
  const padding = 0; // 填充大小为0
  let padded_input_size = image.length + padding * 2; // 计算填充后的输入大小
  $: padded_input_size = image.length + padding * 2; // 响应式更新填充后的输入大小

  let gridInputMatrixSlice = gridData([[0]]); // 初始化输入矩阵切片
  let gridOutputMatrixSlice = gridData([[0]]); // 初始化输出矩阵切片
  let inputHighlights = array1d(image.length * image.length, (i) => true); // 初始化输入高亮数组
  let outputHighlights = array1d(output.length * output.length, (i) => true); // 初始化输出高亮数组
  let interval; // 定义interval变量
  $: {
    let inputHighlights = array1d(image.length * image.length, (i) => true); // 响应式更新输入高亮数组
    let outputHighlights = array1d(output.length * output.length, (i) => true); // 响应式更新输出高亮数组
    let interval; // 响应式更新interval变量
  }
  let counter;

  // 在mouseover和start-relu之间有很多重复。TODO: 修复这个问题。
  function startRelu() {
    counter = 0;
    if (interval) clearInterval(interval); // 如果interval存在，则清除它
    interval = setInterval(() => {
      if (isPaused) return; // 如果暂停，则返回
      const flat_animated = counter % (output.length * output.length); // 计算当前动画的索引
      outputHighlights = array1d(output.length * output.length, (i) => false); // 初始化输出高亮数组
      inputHighlights = array1d(image.length * image.length, (i) => undefined); // 初始化输入高亮数组
      const animatedH = Math.floor(flat_animated / output.length); // 计算当前动画的行索引
      const animatedW = flat_animated % output.length; // 计算当前动画的列索引
      outputHighlights[animatedH * output.length + animatedW] = true; // 设置当前输出高亮
      inputHighlights[animatedH * output.length + animatedW] = true; // 设置当前输入高亮
      const inputMatrixSlice = getMatrixSliceFromInputHighlights(
        image,
        inputHighlights,
        1,
      ); // 从输入高亮中获取矩阵切片
      gridInputMatrixSlice = gridData(inputMatrixSlice); // 更新输入矩阵切片
      const outputMatrixSlice = getMatrixSliceFromOutputHighlights(
        output,
        outputHighlights,
      ); // 从输出高亮中获取矩阵切片
      gridOutputMatrixSlice = gridData(outputMatrixSlice); // 更新输出矩阵切片
      counter++; // 计数器递增
    }, 250); // 每250毫秒执行一次
  }

  function handleMouseover(event) {
    // 初始化输出高亮数组，将所有元素设置为false
    outputHighlights = array1d(output.length * output.length, (i) => false);

    // 获取鼠标悬停的行和列索引
    const animatedH = event.detail.hoverH;
    const animatedW = event.detail.hoverW;

    // 设置当前悬停位置的输出高亮
    outputHighlights[animatedH * output.length + animatedW] = true;

    // 初始化输入高亮数组，将所有元素设置为undefined
    inputHighlights = array1d(image.length * image.length, (i) => undefined);

    // 设置当前悬停位置的输入高亮
    inputHighlights[animatedH * output.length + animatedW] = true;

    // 从输入高亮中获取矩阵切片
    const inputMatrixSlice = getMatrixSliceFromInputHighlights(
      image,
      inputHighlights,
      1,
    );

    // 更新输入矩阵切片
    gridInputMatrixSlice = gridData(inputMatrixSlice);

    // 从输出高亮中获取矩阵切片
    const outputMatrixSlice = getMatrixSliceFromOutputHighlights(
      output,
      outputHighlights,
    );

    // 更新输出矩阵切片
    gridOutputMatrixSlice = gridData(outputMatrixSlice);

    // 暂停动画
    isPaused = true;

    // 发送暂停消息
    dispatch("message", {
      text: isPaused,
    });
  }

  startRelu();
  let gridImage = gridData(image);
  let gridOutput = gridData(output);
  $: {
    startRelu();
    gridImage = gridData(image);
    gridOutput = gridData(output);
  }
</script>

<div class="column has-text-centered">
  <div class="header-text">
    Input ({image.length}, {image[0].length})
  </div>
  <Dataview
    on:message={handleMouseover}
    data={gridImage}
    highlights={inputHighlights}
    outputLength={output.length}
    isKernelMath={false}
    constraint={getVisualizationSizeConstraint(image.length)}
    {dataRange}
    stride={1}
  />
</div>
<div class="column has-text-centered">
  <span>
    max(
    <Dataview
      data={gridData([[0]])}
      highlights={outputHighlights}
      isKernelMath={true}
      constraint={20}
      {dataRange}
    />
    ,
    <Dataview
      data={gridInputMatrixSlice}
      highlights={outputHighlights}
      isKernelMath={true}
      constraint={20}
      {dataRange}
    />
    ) =
    <Dataview
      data={gridOutputMatrixSlice}
      highlights={outputHighlights}
      isKernelMath={true}
      constraint={20}
      {dataRange}
    />
  </span>
</div>
<div class="column has-text-centered">
  <div class="header-text">
    Output ({output.length}, {output[0].length})
  </div>
  <Dataview
    on:message={handleMouseover}
    data={gridOutput}
    highlights={outputHighlights}
    isKernelMath={false}
    outputLength={output.length}
    constraint={getVisualizationSizeConstraint(output.length)}
    {dataRange}
    stride={1}
  />
</div>

<style>
  .column {
    padding: 5px;
  }
</style>
