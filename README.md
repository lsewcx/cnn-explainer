# CNN Explainer

一个旨在帮助非专家学习卷积神经网络（CNN）的交互式可视化系统

[![构建状态](https://github.com/poloclub/cnn-explainer/workflows/build/badge.svg)](https://github.com/poloclub/cnn-explainer/actions)
[![arxiv 标志](https://img.shields.io/badge/arXiv-2004.15004-red)](http://arxiv.org/abs/2004.15004)
[![DOI:10.1109/TVCG.2020.3030418](https://img.shields.io/badge/DOI-10.1109/TVCG.2020.3030418-blue)](https://doi.org/10.1109/TVCG.2020.3030418)

<a href="https://youtu.be/HnWIHWFbuUQ"  target="_blank"><img src="https://i.imgur.com/sCsudVg.png"  style="max-width:100%;"></a>

更多信息，请查看我们的手稿：

[**CNN Explainer: Learning Convolutional Neural Networks with Interactive Visualization**](https://arxiv.org/abs/2004.15004).
Wang, Zijie J., Robert Turko, Omar Shaikh, Haekyu Park, Nilaksh Das, Fred Hohman, Minsuk Kahng, 和 Duen Horng Chau.
_IEEE Transactions on Visualization and Computer Graphics (TVCG), 2020._

## 在线演示

想要在线演示，请访问：http://poloclub.github.io/cnn-explainer/

## 本地运行

克隆或下载此仓库：

```bash
git clone git@github.com:poloclub/cnn-explainer.git

# 如果你不想下载提交历史，可以使用degit
degit poloclub/cnn-explainer
```

安装依赖：

```bash
npm install
```

然后运行 CNN Explainer：

```bash
npm run dev
```

导航至 [localhost:3000](https://localhost:3000)。你应该可以在浏览器中看到正在运行的 CNN Explainer :)

想要了解我们如何训练 CNN，请访问目录 [`./tiny-vgg/`](tiny-vgg)。
如果你想使用 CNN Explainer 配合自己的 CNN 模型或图像类别，请参考 [#8](/../../issues/8) 和 [#14](/../../issues/14)。

## 致谢

CNN Explainer 是由
<a href="https://zijie.wang/">Jay Wang</a>，
<a href="https://www.linkedin.com/in/robert-turko/">Robert Turko</a>，
<a href="http://oshaikh.com/">Omar Shaikh</a>，
<a href="https://haekyu.com/">Haekyu Park</a>，
<a href="http://nilakshdas.com/">Nilaksh Das</a>，
<a href="https://fredhohman.com/">Fred Hohman</a>，
<a href="http://minsuk.com">Minsuk Kahng</a>, 和
<a href="https://www.cc.gatech.edu/~dchau/">Polo Chau</a> 创建的，
这是佐治亚理工学院和俄勒冈州立大学研究合作的成果。

我们感谢
[Anmol Chhabria](https://www.linkedin.com/in/anmolchhabria),
[Kaan Sancak](https://kaansancak.com),
[Kantwon Rogers](https://www.kantwon.com), 以及
[Georgia Tech Visualization Lab](http://vis.gatech.edu)
的支持和建设性反馈。

## 引用

```bibTeX
@article{wangCNNExplainerLearning2020,
  title = {{{CNN Explainer}}: {{Learning Convolutional Neural Networks}} with {{Interactive Visualization}}},
  shorttitle = {{{CNN Explainer}}},
  author = {Wang, Zijie J. and Turko, Robert and Shaikh, Omar and Park, Haekyu and Das, Nilaksh and Hohman, Fred and Kahng, Minsuk and Chau, Duen Horng},
  journal={IEEE Transactions on Visualization and Computer Graphics (TVCG)},
  year={2020},
  publisher={IEEE}
}
```

## 许可

该软件可在 [MIT License](https://github.com/poloclub/cnn-explainer/blob/master/LICENSE) 下获得。

## 联系方式

如有任何问题，欢迎[提出问题](https://github.com/poloclub/cnn-explainer/issues/new/choose) 或联系 [Jay Wang](https://zijie.wang)。
