# Reinforce Learning Tutorial

我怎么现在才开始看强化学习.jpg

总之这里是一些强化学习的参考代码，然后我的笔记挂在我的 Github Pages 上面，导航：

[Reinforce learning notes](https://ashitemaru.github.io/2022/06/30/note-of-rl/)

## Updates

### 2022.07.02

更新了 Q-Learning、Sarsa 相关内容，主要是一个学习走迷宫的程序。

我觉得他这个代码框架存在一个问题，就是当机器走到角落的时候，实际上它的决策空间会变小，但是代码里面没有体现这一点，只是简单的让机器不动。比如机器在左上角选择向左走，那么只是简单地让机器停在左上角。

这是一个并不好的处理，因为如果此时机器还没有碰巧到达天堂产生正向激励的话，机器完全会缩在角落里来避免地狱的反向激励。

有时间给他代码改改吧。

### 2022.07.05

添加了一部分 DQN 相关代码，目前笔记暂且没有跟进。

他用了很多已经废弃的 TensorFlow 语法，这个也是后面改改比较好，但是一定程度上我不怎么想动了。

### 2022.07.18

由于实习的事情和其他的一些事情搁置了这个 Repo 一段时间。

添加了 Policy Gradients 的实验代码并在笔记里完成了 Policy Gradients 的相关数学推导。

添加了 Actor Critic 的实验代码和简单的数学推导，A2C 和 A3C 还在研究。

### 2022.07.19

将 Actor Critic 直接改成了 A2C，添加了 DDPG 的代码。

### 2022.07.22 - 2022.07.23

修复了 DDPG 代码之中存在的 bug。其实我也不是很清楚我到底做了什么他就通了，网络就收敛了，我也只是不断在把他那些花里胡哨的东西修改成为最原始最基本的东西。

有点怀疑我这样改完之后是不是 DDPG 了。

### 2022.07.25

添加了 A3C 相关代码。但这个代码并不优美，然而我自己本来就十分不擅长调试带有多线程的代码，那就 Whatever 了。说实话，现在这一份代码是存在数据竞争风险的，但实际上没有什么很大的问题，也就暂且不管了。

### 2022.07.27

在学习 PPO 的过程中遇到了点数学问题，目前还没解决，反倒又获得了一堆类似 Bootstrap、Bellman Equation、Off-policy PG 等一堆要学的东西。

之后慢慢再把 PPO 弄上来吧。