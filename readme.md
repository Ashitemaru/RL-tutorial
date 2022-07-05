# Reinforce Learning Tutorial

我怎么现在才开始看强化学习.jpg

总之这里是一些强化学习的参考代码，然后我的笔记挂在我的 Github Pages 上面，导航：

[Reinforce learning notes](https://ashitemaru.github.io/2022/06/30/note-of-rl/)

## Updates

### 2022.07.02

更新了 Q-Learning、Sarsa 相关内容，主要是一个学习走迷宫的程序。

---

我觉得他这个代码框架存在一个问题，就是当机器走到角落的时候，实际上它的决策空间会变小，但是代码里面没有体现这一点，只是简单的让机器不动。比如机器在左上角选择向左走，那么只是简单地让机器停在左上角。

这是一个并不好的处理，因为如果此时机器还没有碰巧到达天堂产生正向激励的话，机器完全会缩在角落里来避免地狱的反向激励。

有时间给他代码改改吧。

### 2022.07.05

添加了一部分 DQN 相关代码，目前笔记暂且没有跟进。

---

他用了很多已经废弃的 TensorFlow 语法，这个也是后面改改比较好，但是一定程度上我不怎么想动了。