---
title: git rebase 用法
date: 2019-03-30 16:21:34
tags: 操作commits
categories: git
---
# 「Git」合并多个 Commit
## 我们可能会由于各种各样的原因提交了许多临时的 commit，而这些 commit 拼接起来才是完整的任务。那么我们为了避免太多的 commit 而造成版本控制的混乱，通常我们推荐将这些 commit 合并成一个。
<!--more-->
### 我们用到命令` git rebase -i  [startpoint]  [endpoint]`其中`-i`的意思是--interactive，即弹出交互式的界面让用户编辑完成合并操作，`[startpoint]  [endpoint]`则指定了一个编辑区间，如果不指定`[endpoint]`，则该区间的终点默认是当前分支HEAD所指向的commit(该区间指定的是一个前开后闭的区间)。也可以用`git rebase -i HEAD~3 `(合并最近三条commit)。
### 接下来进入编辑页面，根据需求按指令编辑commit
* pick：保留该commit（缩写:p）
*  reword：保留该commit，但我需要修改该commit的注释（缩写:r）
* edit：保留该commit, 但我要停下来修改该提交(不仅仅修改注释)（缩写:e）
* squash：将该commit和前一个commit合并（缩写:s）
* fixup：将该commit和前一个commit合并，但我不要保留该提交的注释信息（缩写:f）
* exec：执行shell命令（缩写:x）
* drop：我要丢弃该commit（缩写:d）
### 将 "pick" 改成 "squash" 或者 "s"，意思是将该 commit （add b.php）和 前面的 commit (add a.php) 合并。编辑完成后，保存并退出（wq!）。编辑完合并 commit 的注释之后，就保存退出（:wq!）。ok!
# 将某一段commit粘贴到另一个分支上
###   我们使用命令的形式为:`git rebase   [startpoint]   [endpoint]  --onto  [branchName]`其中，[startpoint]  [endpoint]仍然和上一个命令一样指定了一个编辑区间(前开后闭)，--onto的意思是要将该指定的提交复制到哪个分支上。执行 git rebase 命令之后，我们发现当前的 HEAD 处于游离状态。执行 git rebase 命令之后，我们发现当前的 HEAD 处于游离状态。所以我们需要使用 git reset 命令，将 master 所指向的 commit id 设置为当前 HEAD 所指向的 commit id。
![](git rebase-用法\1.png)