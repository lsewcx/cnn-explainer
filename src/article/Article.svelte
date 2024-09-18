<script>
  import HyperparameterView from "../detail-view/Hyperparameterview.svelte";

  let softmaxEquation = `$$\\text{Softmax}(x_{i}) = \\frac{\\exp(x_i)}{\\sum_j \\exp(x_j)}$$`;
  let reluEquation = `$$\\text{ReLU}(x) = \\max(0,x)$$`;

  let currentPlayer;
</script>

<body>
  <div id="description">
    <h2>什么是卷积神经网络？</h2>
    <p>
      在机器学习中，分类器将类别标签分配给数据点。例如，<em>图像分类器</em
      >会为图像中的对象生成类别标签（例如，鸟、飞机）。<em>卷积神经网络</em
      >，简称CNN，是一种分类器，非常擅长解决这个问题！
    </p>
    <p>
      CNN是一种神经网络：一种用于识别数据模式的算法。一般来说，神经网络由一组组织成层的神经元组成，每个神经元都有自己的可学习权重和偏置。让我们将CNN分解为其基本构建块。
    </p>
    <ol>
      <li>
        <strong>张量</strong
        >可以被认为是一个n维矩阵。在上面的CNN中，张量将是3维的，输出层除外。
      </li>
      <li>
        <strong>神经元</strong
        >可以被认为是一个函数，它接受多个输入并产生一个输出。神经元的输出在上面表示为<span
          style="color:#FF7577;">红色</span
        >
        &rarr; <span style="color:#60A7D7;">蓝色</span>
        <strong>激活图</strong>。
      </li>
      <li>
        <strong>层</strong>只是具有相同操作的一组神经元，包括相同的超参数。
      </li>
      <li>
        <strong>核权重和偏置</strong
        >，虽然每个神经元都是独特的，但在训练阶段进行调整，使分类器能够适应提供的问题和数据集。它们在可视化中用<span
          style="color:#BC8435;">黄色</span
        >
        &rarr;
        <span style="color:#39988F;">绿色</span
        >的分散色标编码。具体值可以通过点击神经元或在<em>卷积弹性解释视图</em
        >中悬停在核/偏置上查看。
      </li>
      <li>
        CNN传达了一个<strong>可微分的评分函数</strong
        >，在输出层的可视化中表示为<strong>类别分数</strong>。
      </li>
    </ol>
    <p>
      如果你之前学习过神经网络，这些术语可能对你来说很熟悉。那么，是什么让CNN与众不同呢？CNN利用了一种特殊类型的层，恰当地命名为卷积层，使它们能够很好地从图像和类似图像的数据中学习。关于图像数据，CNN可以用于许多不同的计算机视觉任务，例如<a
        href="http://ijcsit.com/docs/Volume%207/vol7issue5/ijcsit20160705014.pdf"
        title="CNN应用">图像处理、分类、分割和目标检测</a
      >。
    </p>
    <p>
      在CNN
      Explainer中，你可以看到一个简单的CNN如何用于图像分类。由于网络的简单性，其性能并不完美，但这没关系！CNN
      Explainer中使用的网络架构<a
        href="http://cs231n.stanford.edu/"
        title="由斯坦福CS231n提出的Tiny VGG网络">Tiny VGG</a
      >包含了许多当今最先进的CNN中使用的相同层和操作，但规模较小。这样，入门会更容易理解。
    </p>

    <h2>网络的每一层都做什么？</h2>
    <p>
      让我们逐层了解网络。阅读时，可以随意点击和悬停在上面的可视化部分进行互动。
    </p>
    <h4 id="article-input">输入层</h4>
    <p>
      输入层（最左边的层）表示输入到CNN的图像。因为我们使用RGB图像作为输入，所以输入层有三个通道，分别对应红色、绿色和蓝色通道，这些通道在这一层中显示。点击上面的<img
        class="is-rounded"
        width="12%"
        height="12%"
        src="PUBLIC_URL/assets/figures/network_details.png"
        alt="网络详情图标"
      />图标时，使用颜色标尺显示详细信息（在此层和其他层上）。
    </p>
    <h4 id="article-convolution">卷积层</h4>
    <p>
      卷积层是CNN的基础，因为它们包含学习到的核（权重），这些核提取出区分不同图像的特征——这正是我们想要分类的！当你与卷积层互动时，你会注意到前一层和卷积层之间的链接。每个链接代表一个独特的核，用于卷积操作以生成当前卷积神经元的输出或激活图。
    </p>
    <p>
      卷积神经元与前一层对应神经元的输出和独特的核进行逐元素点积。这将产生与独特核一样多的中间结果。卷积神经元是所有中间结果与学习到的偏置相加的结果。
    </p>
    <p>
      例如，让我们看看上面Tiny
      VGG架构中的第一个卷积层。注意，这一层有10个神经元，而前一层只有3个神经元。在Tiny
      VGG架构中，卷积层是全连接的，这意味着每个神经元都连接到前一层的每个神经元。关注第一个卷积层中最上面的卷积神经元的输出，我们看到当我们悬停在激活图上时，有3个独特的核。
    </p>
    <div class="figure">
      <img
        src="PUBLIC_URL/assets/figures/convlayer_overview_demo.gif"
        alt="点击第一个卷积层激活图的最上面节点"
        width="60%"
        height="60%"
        align="middle"
      />
      <div class="figure-caption">
        图1.
        当你悬停在第一个卷积层最上面节点的激活图上时，你可以看到应用了3个核以生成这个激活图。点击这个激活图后，你可以看到每个独特核的卷积操作。
      </div>
    </div>

    <p>
      这些卷积核的大小是由网络架构设计者指定的超参数。为了生成卷积神经元的输出（激活图），我们必须对上一层的输出和网络学习到的独特卷积核进行逐元素点积。在TinyVGG中，点积操作使用步幅为1，这意味着每次点积卷积核都会移动1个像素，但这是网络架构设计者可以调整以更好地适应其数据集的超参数。我们必须对所有3个卷积核执行此操作，这将产生3个中间结果。
    </p>
    <div class="figure">
      <img
        src="PUBLIC_URL/assets/figures/convlayer_detailedview_demo.gif"
        alt="点击最上面的第一个卷积层激活图"
      />
      <div class="figure-caption">
        图2. 应用于生成讨论的激活图的最上面的中间结果的卷积核。
      </div>
    </div>
    <p>
      然后，进行逐元素求和，包含所有3个中间结果以及网络学习到的偏置。之后，生成的二维张量将是界面上方可查看的第一个卷积层中最上面的神经元的激活图。必须对每个神经元的激活图应用相同的操作。
    </p>
    <p>
      通过一些简单的数学计算，我们可以推断出在第一个卷积层中应用了3 x 10 =
      30个独特的卷积核，每个卷积核的大小为3x3。卷积层与上一层之间的连接性是构建网络架构时的设计决策，这将影响每个卷积层的卷积核数量。点击可视化以更好地理解卷积层背后的操作。看看你能否跟随上面的例子！
    </p>
    <h6>理解超参数</h6>
    <p>
      <HyperparameterView />
    </p>
    <ol>
      <li>
        <strong>填充</strong
        >通常在卷积核超出激活图时是必要的。填充保留了激活图边界的数据，从而提高了性能，并且可以帮助<a
          href="https://arxiv.org/pdf/1603.07285.pdf"
          title="见第13页">保留输入的空间大小</a
        >，这使得架构设计者能够构建更深、更高性能的网络。存在<a
          href="https://arxiv.org/pdf/1811.11718.pdf"
          title="概述主要的填充技术">许多填充技术</a
        >，但最常用的方法是零填充，因为它的性能、简单性和计算效率。这种技术涉及在输入的边缘对称地添加零。许多高性能的CNN如<a
          href="https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf"
          title="AlexNet">AlexNet</a
        >采用了这种方法。
      </li>
      <li>
        <strong>卷积核大小</strong
        >，通常也称为滤波器大小，指的是在输入上滑动窗口的尺寸。选择这个超参数对图像分类任务有巨大的影响。例如，小卷积核大小能够从输入中提取包含高度局部特征的更多信息。如上面的可视化所示，较小的卷积核大小也导致层尺寸的较小减少，从而允许更深的架构。相反，较大的卷积核大小提取的信息较少，这导致层尺寸的快速减少，通常导致性能较差。较大的卷积核更适合提取较大的特征。最终，选择适当的卷积核大小将取决于你的任务和数据集，但通常，较小的卷积核大小在图像分类任务中表现更好，因为架构设计者能够堆叠<a
          href="https://arxiv.org/pdf/1409.1556.pdf"
          title="了解为什么更深的网络表现更好！"
          >越来越多的层，以学习越来越复杂的特征</a
        >！
      </li>
      <li>
        <strong>步幅</strong
        >表示每次卷积核应移动多少像素。例如，如上面的卷积层示例所述，Tiny
        VGG的卷积层使用步幅为1，这意味着在输入的3x3窗口上执行点积以生成输出值，然后在每次后续操作中向右移动一个像素。步幅对CNN的影响类似于卷积核大小。随着步幅的减小，学习到的特征更多，因为提取的数据更多，这也导致输出层更大。相反，随着步幅的增加，这导致特征提取更有限，输出层尺寸更小。架构设计者的一个责任是在实现CNN时确保卷积核对称地滑过输入。使用上面的超参数可视化来改变各种输入/卷积核尺寸上的步幅，以理解这一约束！
      </li>
    </ol>
    <h4>激活函数</h4>
    <h6 id="article-relu">ReLU</h6>
    <p>
      神经网络在现代技术中非常普遍——因为它们非常准确！今天表现最好的CNN由大量层组成，能够学习越来越多的特征。这些突破性的CNN能够实现如此<a
        href="https://arxiv.org/pdf/1512.03385.pdf"
        title="ResNet">惊人的准确性</a
      >的部分原因是因为它们的非线性。ReLU将急需的非线性引入模型。非线性对于生成非线性决策边界是必要的，这样输出就不能写成输入的线性组合。如果没有非线性激活函数，深度CNN架构将退化为单个等效的卷积层，其性能远不及前者。ReLU激活函数被专门用作非线性激活函数，而不是其他非线性函数如<em
        >Sigmoid</em
      >，因为<a href="https://arxiv.org/pdf/1906.01975.pdf" title="见第29页"
        >经验观察</a
      >表明，使用ReLU的CNN比它们的对手训练速度更快。
    </p>
    <p>
      ReLU激活函数是逐元素的数学运算：{reluEquation}
    </p>
    <div class="figure">
      <img
        src="PUBLIC_URL/assets/figures/relu_graph.png"
        alt="relu图"
        width="30%"
        height="30%"
      />
      <div class="figure-caption">
        图3. ReLU激活函数的图形，忽略所有负数据。
      </div>
    </div>
    <p>
      这个激活函数逐元素应用于输入张量的每个值。例如，如果对值2.24应用ReLU，结果将是2.24，因为2.24大于0。你可以通过点击上方网络中的ReLU神经元来观察这个激活函数的应用。上述网络架构中的每个卷积层之后都会执行修正线性激活函数（ReLU）。注意这个层对网络中各个神经元激活图的影响！
    </p>
    <h6 id="article-softmax">Softmax</h6>
    <p>
      {softmaxEquation}
      Softmax操作有一个关键目的：确保CNN的输出总和为1。因此，Softmax操作对于将模型输出缩放为概率非常有用。点击最后一层可以看到网络中的Softmax操作。注意展平后的logits没有缩放到0到1之间。为了直观地表示每个logit（未缩放的标量值）的影响，它们使用
      <span style="color:#FFC385;">浅橙色</span>
      &rarr;
      <span style="color:#C44103;">深橙色</span
      >颜色编码。通过Softmax函数后，每个类别现在对应于一个适当的概率！
    </p>
    <p>
      你可能会想标准化和Softmax之间的区别是什么——毕竟，两者都将logits重新缩放到0和1之间。记住反向传播是训练神经网络的关键方面——我们希望正确答案具有最大的“信号”。通过使用Softmax，我们实际上是在“近似”argmax，同时获得可微性。重新缩放不会显著提高最大值的权重，而Softmax会。简单来说，Softmax是一个“更柔和的”argmax——明白我们在说什么了吗？
    </p>
    <div class="figure">
      <img
        src="PUBLIC_URL/assets/figures/softmax_animation.gif"
        alt="softmax交互公式视图"
      />
      <div class="figure-caption">
        图4. <em>Softmax交互公式视图</em
        >允许用户与颜色编码的logits和公式进行交互，以了解展平层后的预测分数如何被归一化以生成分类分数。
      </div>
    </div>
    <h4 id="article-pooling">池化层</h4>
    <p>
      在不同的CNN架构中有许多类型的池化层，但它们的目的都是逐渐减少网络的空间范围，从而减少网络的参数和整体计算量。上述Tiny
      VGG架构中使用的池化类型是最大池化。
    </p>
    <p>
      最大池化操作需要在架构设计期间选择卷积核大小和步幅长度。一旦选择，操作将以指定的步幅滑动卷积核，同时仅选择输入中每个卷积核切片的最大值以生成输出值。可以通过点击上方网络中的池化神经元来查看此过程。
    </p>
    <p>
      在上述Tiny
      VGG架构中，池化层使用2x2的卷积核和步幅为2的设置。这种规格的操作会丢弃75%的激活值。通过丢弃这么多的值，Tiny
      VGG在计算上更高效，并且避免了过拟合。
    </p>
    <h4 id="article-flatten">展平层</h4>
    <p>
      这一层将网络中的三维层转换为一维向量，以适应全连接层的输入进行分类。例如，一个5x5x2的张量将被转换为大小为50的向量。网络之前的卷积层从输入图像中提取特征，但现在是时候对这些特征进行分类了。我们使用softmax函数对这些特征进行分类，这需要一维输入。这就是为什么展平层是必要的。可以通过点击任何输出类别来查看这一层。
    </p>

    <h2>交互功能</h2>
    <ol>
      <li>
        通过选择
        <strong>上传你自己的图片</strong>
        <img
          class="icon is-rounded"
          src="PUBLIC_URL/assets/figures/upload_image_icon.png"
          alt="上传图片图标"
        />
        来了解你的图片是如何被分类到10个类别中的。通过分析整个网络中的神经元，你可以理解激活图和提取的特征。
      </li>
      <li>
        <strong>更改激活图颜色刻度</strong>
        通过调整
        <img
          class="is-rounded"
          width="12%"
          height="12%"
          src="PUBLIC_URL/assets/figures/heatmap_scale.png"
          alt="热图"
        />
        来更好地理解不同抽象层次上激活的影响。
      </li>
      <li>
        通过点击
        <strong>了解网络详情</strong>
        <img
          class="is-rounded"
          width="12%"
          height="12%"
          src="PUBLIC_URL/assets/figures/network_details.png"
          alt="网络详情图标"
        />
        图标来了解层的维度和颜色刻度等网络详情。
      </li>
      <li>
        通过点击
        <strong>模拟网络操作</strong>
        <img
          class="icon is-rounded"
          src="PUBLIC_URL/assets/figures/play_button.png"
          alt="播放图标"
        />
        按钮或在
        <em>交互公式视图</em>
        中与层切片交互，通过悬停在输入或输出的部分来理解映射和底层操作。
      </li>
      <li>
        通过点击
        <strong>学习层功能</strong>
        <img
          class="icon is-rounded"
          src="PUBLIC_URL/assets/figures/info_button.png"
          alt="信息图标"
        />
        从
        <em>交互公式视图</em>
        中阅读文章中的层详细信息。
      </li>
    </ol>

    <!-- <h2>Video Tutorial</h2>
    <ul>
      <li class="video-link" on:click={currentPlayer.play(0)}>
        CNN Explainer Introduction
        <small>(0:00-0:22)</small>
      </li>
      <li class="video-link" on:click={currentPlayer.play(27)}>
        <em>Overview</em>
        <small>(0:27-0:37)</small>
      </li>
      <li class="video-link" on:click={currentPlayer.play(37)}>
        Convolutional <em>Elastic Explanation View</em>
        <small>(0:37-0:46)</small>
      </li>
      <li class="video-link" on:click={currentPlayer.play(46)}>
        Convolutional, ReLU, and Pooling <em>Interactive Formula Views</em>
        <small>(0:46-1:21)</small>
      </li>
      <li class="video-link" on:click={currentPlayer.play(82)}>
        Flatten <em>Elastic Explanation View</em>
        <small>(1:22-1:41)</small>
      </li>
      <li class="video-link" on:click={currentPlayer.play(101)}>
        Softmax <em>Interactive Formula View</em>
        <small>(1:41-2:02)</small>
      </li>
      <li class="video-link" on:click={currentPlayer.play(126)}>
        Engaging Learning Experience: Understanding Classification
        <small>(2:06-2:28)</small>
      </li>
      <li class="video-link" on:click={currentPlayer.play(149)}>
        Interactive Tutorial Article
        <small>(2:29-2:54)</small>
      </li>
    </ul>
    <div class="video">
      <Youtube
        videoId="HnWIHWFbuUQ"
        playerId="demo_video"
        bind:this={currentPlayer}
      />
    </div>

    <h2>How is CNN Explainer implemented?</h2>
    <p>
      CNN Explainer uses <a href="https://js.tensorflow.org/"
        ><em>TensorFlow.js</em></a
      >, an in-browser GPU-accelerated deep learning library to load the
      pretrained model for visualization. The entire interactive system is
      written in Javascript using
      <a href="https://svelte.dev/"><em>Svelte</em></a>
      as a framework and <a href="https://d3js.org/"><em>D3.js</em></a> for visualizations.
      You only need a web browser to get started learning CNNs today!
    </p>

    <h2>Who developed CNN Explainer?</h2>
    <p>
      CNN Explainer was created by
      <a href="https://zijie.wang/">Jay Wang</a>,
      <a href="https://www.linkedin.com/in/robert-turko/">Robert Turko</a>,
      <a href="http://oshaikh.com/">Omar Shaikh</a>,
      <a href="https://haekyu.com/">Haekyu Park</a>,
      <a href="http://nilakshdas.com/">Nilaksh Das</a>,
      <a href="https://fredhohman.com/">Fred Hohman</a>,
      <a href="http://minsuk.com">Minsuk Kahng</a>, and
      <a href="https://www.cc.gatech.edu/~dchau/">Polo Chau</a>, which was the
      result of a research collaboration between Georgia Tech and Oregon State.
      We thank Anmol Chhabria, Kaan Sancak, Kantwon Rogers, and the Georgia Tech
      Visualization Lab for their support and constructive feedback. This work
      was supported in part by NSF grants IIS-1563816, CNS-1704701, NASA NSTRF,
      DARPA GARD, gifts from Intel, NVIDIA, Google, Amazon.
    </p>
  </div> -->
  </div></body
>

<style>
  #description {
    margin-bottom: 60px;
    margin-left: auto;
    margin-right: auto;
    max-width: 78ch;
  }

  #description h2 {
    color: #444;
    font-size: 40px;
    font-weight: 450;
    margin-bottom: 12px;
    margin-top: 60px;
  }

  #description h4 {
    color: #444;
    font-size: 32px;
    font-weight: 450;
    margin-bottom: 8px;
    margin-top: 44px;
  }

  #description h6 {
    color: #444;
    font-size: 24px;
    font-weight: 450;
    margin-bottom: 8px;
    margin-top: 44px;
  }

  #description p {
    margin: 16px 0;
  }

  #description p img {
    vertical-align: middle;
  }

  #description .figure-caption {
    font-size: 13px;
    margin-top: 5px;
  }

  #description ol {
    margin-left: 40px;
  }

  #description p,
  #description div,
  #description li {
    color: #555;
    font-size: 17px;
    line-height: 1.6;
  }

  #description small {
    font-size: 12px;
  }

  #description ol li img {
    vertical-align: middle;
  }

  #description .video-link {
    color: #3273dc;
    cursor: pointer;
    font-weight: normal;
    text-decoration: none;
  }

  #description ul {
    list-style-type: disc;
    margin-top: -10px;
    margin-left: 40px;
    margin-bottom: 15px;
  }

  #description a:hover,
  #description .video-link:hover {
    text-decoration: underline;
  }

  .figure,
  .video {
    width: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
  }
</style>
