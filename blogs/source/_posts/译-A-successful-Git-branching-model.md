---
title: '[译]A successful Git branching model'
categories:
  - - 翻译
  - - 版本管理
tags:
  - git
abbrlink: aae96086
date: 2019-11-29 19:59:30
---

原文地址：[A successful Git branching model](https://nvie.com/posts/a-successful-git-branching-model/)

>In this post I present the development model that I’ve introduced for some of my projects (both at work and private) about a year ago, and which has turned out to be very successful. I’ve been meaning to write about it for a while now, but I’ve never really found the time to do so thoroughly, until now. I won’t talk about any of the projects’ details, merely about the branching strategy and release management.

在这篇文章中，我介绍了大约一年前为我的一些项目(包括工作项目和私人项目)引入的开发模型，结果证明非常成功。一段时间以来，我一直想写这篇文章，但直到现在，我还没有真正找到时间彻底地写完。我不会谈论任何项目的细节，仅仅是分支策略和发布管理

![](/imgs/gitflow/git-model@2x.png)

## Why git?

为什么是git?

>For a thorough discussion on the pros and cons of Git compared to centralized source code control systems, see the web. There are plenty of flame wars going on there. As a developer, I prefer Git above all other tools around today. Git really changed the way developers think of merging and branching. From the classic CVS/Subversion world I came from, merging/branching has always been considered a bit scary (“beware of merge conflicts, they bite you!”) and something you only do every once in a while.

有关Git相对于集中式源码控制系统的优缺点的详细讨论，请参见[网站](http://git.or.cz/gitwiki/GitSvnComparsion)。那里正在进行大量的战争。作为一名开发人员，我更喜欢Git，而不是现在的所有其他工具。Git确实改变了开发人员对合并和分支的想法。从我来自的经典CVS/Subversion世界来看，合并/分支一直被认为有点可怕(“当心合并冲突，它们会咬你！”)以及你偶尔才会做的事情

>But with Git, these actions are extremely cheap and simple, and they are considered one of the core parts of your daily workflow, really. For example, in CVS/Subversion books, branching and merging is first discussed in the later chapters (for advanced users), while in every Git book, it’s already covered in chapter 3 (basics).

但是使用Git，这些操作非常便宜和简单，而且它们被认为是你日常工作流程的核心部分之一，真的。例如，在CVS/Subversion[书籍](http://svnbook.red-bean.com/)中，分支和合并首先在后面的章节中讨论(对于高级用户)，而在每本Git[书籍](http://pragprog.com/titles/tsgit/pragmatic-version-control-using-git)中，它已经在第3章(基础)中讨论过了

>As a consequence of its simplicity and repetitive nature, branching and merging are no longer something to be afraid of. Version control tools are supposed to assist in branching/merging more than anything else.

由于它的简单性和重复性，分支和合并不再是什么可怕的事情。版本控制工具应该比任何其他工具都更有助于分支/合并

>Enough about the tools, let’s head onto the development model. The model that I’m going to present here is essentially no more than a set of procedures that every team member has to follow in order to come to a managed software development process.

关于工具，我们先来看看开发模型。我将在这里展示的模型基本上只不过是一组程序，每个团队成员都必须遵循这些程序才能进入托管软件开发过程

## Decentralized but centralized

去中心化的同时进行中心化

>The repository setup that we use and that works well with this branching model, is that with a central "truth" repo. Note that this repo is only considered to be the central one (since Git is a DVCS, there is no such thing as a central repo at a technical level). We will refer to this repo as origin, since this name is familiar to all Git users.

我们使用的存储库设置与这个分支模型配合得很好，那是一个中心“真正的”仓库。请注意，这种仓库只被认为是中间仓库(因为Git是DVCS的，所以在技术层面上不存在中心仓库)。我们将把这个仓库称为origin，因为这个名字是所有Git用户都熟悉的

![](/imgs/gitflow/centr-decentr@2x.png)

>Each developer pulls and pushes to origin. But besides the centralized push-pull relationships, each developer may also pull changes from other peers to form sub teams. For example, this might be useful to work together with two or more developers on a big new feature, before pushing the work in progress to origin prematurely. In the figure above, there are subteams of Alice and Bob, Alice and David, and Clair and David.

每一个开发人员都向origin拉取和推送代码。但是除了集中式的推拉关系之外，每个开发人员也可以从其他同事那里拉取变化来组成子项目。例如，在将正在进行的工作过早推向origin之前，与两个或更多的开发人员一起开发一个大的新特性可能会很有用。在上图中，有Alice和Bob、Alice和David、Clair和David的子项目

>Technically, this means nothing more than that Alice has defined a Git remote, named bob, pointing to Bob’s repository, and vice versa.

从技术上讲，这仅仅意味着Alice已经定义了一个远程Git，名为bob，指向Bob的存储库，反之亦然

## The main branches

主要分支

>At the core, the development model is greatly inspired by existing models out there. The central repo holds two main branches with an infinite lifetime:

这个分支模型极大的受之前开发模型的启发。中央仓库仅持有两个最重要的分支:

* master
* develop

![](/imgs/gitflow/main-branches@2x.png)

>The master branch at origin should be familiar to every Git user. Parallel to the master branch, another branch exists called develop.

每个Git用户都应该熟悉origin的master分支。与master分支并行的是另一个分支，叫做develop

>We consider origin/master to be the main branch where the source code of HEAD always reflects a production-ready state.

我们认为origin/master是主分支，其HEAD指针指向的源代码已经用于后续的生产环节

>We consider origin/develop to be the main branch where the source code of HEAD always reflects a state with the latest delivered development changes for the next release. Some would call this the “integration branch”. This is where any automatic nightly builds are built from.

同时origin/develop也是主分支，其HEAD指针指向的源代码反映的是将用于下一个版本的最新交付的开发变更状态。有人称之为“集成分支”。这个分支可作用于自动构建

>When the source code in the develop branch reaches a stable point and is ready to be released, all of the changes should be merged back into master somehow and then tagged with a release number. How this is done in detail will be discussed further on.

当develop分支中的源代码到达一个稳定点并准备发布时，所有的变更应该以某种方式合并回master，并且标记发布号。如何详细完成将在下面进一步讨论

>Therefore, each time when changes are merged back into master, this is a new production release by definition. We tend to be very strict at this, so that theoretically, we could use a Git hook script to automatically build and roll-out our software to our production servers everytime there was a commit on master.

因此，每次将变更合并回master时，根据定义，这是一个新的生产版本。我们倾向于对此非常严格，所以理论上，在每次提交到master时，我们可以使用Git hook脚本来自动构建软件并将其部署到生产服务器上

## Supporting branches

支持分支

>Next to the main branches master and develop, our development model uses a variety of supporting branches to aid parallel development between team members, ease tracking of features, prepare for production releases and to assist in quickly fixing live production problems. Unlike the main branches, these branches always have a limited life time, since they will be removed eventually.

除了主分支master和develop之外，我们的开发模型还使用各种支持分支来帮助团队成员之间的并行开发，简化特性跟踪，为生产发布做准备，并帮助快速修复实时生产问题。与主分支不同，这些分支的生命周期总是有限的，因为它们最终会被移除

>The different types of branches we may use are:
>* Feature branches
>* Release branches
>* Hotfix branches

我们可以使用的不同类型的分支有:

* 特征分支
* 版本分支
* 热修复分支

>Each of these branches have a specific purpose and are bound to strict rules as to which branches may be their originating branch and which branches must be their merge targets. We will walk through them in a minute.

这些分支中的每一个都有特定的目的，并受严格规则的约束，即哪些分支可能是它们的起始分支，哪些分支必须是它们的合并目标

>By no means are these branches "special" from a technical perspective. The branch types are categorized by how we use them. They are of course plain old Git branches.

从技术角度来看，这些分支绝不是“特殊的”。分支类型是根据我们如何使用它们来分类的。它们当然是普通的老式Git分支

### Feature branches

特征分支

>May branch off from: **develop**
>Must merge back into: **develop**

特征分支从develop分支fork过来，同时必须合并回develop分支

>Branch naming convention:
>* anything except master, develop, release-*, or hotfix-*

分支命名规范：**可以是任何名字**，除了`master, develop, release-*以及hotfix-*`

>Feature branches (or sometimes called topic branches) are used to develop new features for the upcoming or a distant future release. When starting development of a feature, the target release in which this feature will be incorporated may well be unknown at that point. The essence of a feature branch is that it exists as long as the feature is in development, but will eventually be merged back into develop (to definitely add the new feature to the upcoming release) or discarded (in case of a disappointing experiment).

特性分支(或有时称为主题分支)用于为即将到来或遥远的未来版本开发新特性。当开始开发一个特性时，这个特性将被包含在其中的目标版本在那个时候可能是未知的。特性分支的本质是，只要特性还在开发中，它就一直存在，但最终会被合并回develop中(在即将发布的版本中明确添加的新特性)或被丢弃(在实验令人失望的情况下)

>Feature branches typically exist in developer repos only, not in origin.

特征分支通常只存在于开发人员的个人仓库中，而不存在于origin中

![](/imgs/gitflow/fb@2x.png)

#### Creating a feature branch

创建特征分支

>When starting work on a new feature, branch off from the develop branch.

当开始一个新特性的工作时，从fork develop分支开始

```
$ git checkout -b myfeature develop
Switched to a new branch "myfeature"
```

#### Incorporating a finished feature on develop

在develop中加入已完成的特征

>Finished features may be merged into the develop branch to definitely add them to the upcoming release:

完成的功能可能会合并到develop分支中，以明确地将其添加到即将发布的版本中:

```
$ git checkout develop
Switched to branch 'develop'

$ git merge --no-ff myfeature
Updating ea1b82a..05e9557
(Summary of changes)

$ git branch -d myfeature
Deleted branch myfeature (was 05e9557).

$ git push origin develop
```

>The --no-ff flag causes the merge to always create a new commit object, even if the merge could be performed with a fast-forward. This avoids losing information about the historical existence of a feature branch and groups together all commits that together added the feature. Compare:

**虽然合并可以通过fast-forward来执行，但是最好是使用-no-ff标志使合并操作仅创建一个新的提交对象**。这避免了丢失关于特征分支的历史存在的信息，并将所有一起添加该特征的提交组合在一起。比较如下:

![](/imgs/gitflow/merge-without-ff@2x.png)

>In the latter case, it is impossible to see from the Git history which of the commit objects together have implemented a feature—you would have to manually read all the log messages. Reverting a whole feature (i.e. a group of commits), is a true headache in the latter situation, whereas it is easily done if the --no-ff flag was used.

在后一种情况下，无法从Git历史中发现哪几次提交对象一起实现了一个特性 - 此时必须手动读取所有日志消息。同时，恢复整个功能(即一组提交)是一个真正令人头疼的问题，而如果使用-no-ff标志，这很容易做到（仅包含单次提交）

>Yes, it will create a few more (empty) commit objects, but the gain is much bigger than the cost.

虽然它会创建更多(空的)提交对象，但是收益远大于成本

### Release branches

版本分支

>May branch off from: **develop**
>Must merge back into: **develop and master**

版本分支可以从develop分支进行fork，它必须合并到develop和master分支

> Branch naming convention:
>* release-*

分支命名规范：**release-***

>Release branches support preparation of a new production release. They allow for last-minute dotting of i’s and crossing t’s. Furthermore, they allow for minor bug fixes and preparing meta-data for a release (version number, build dates, etc.). By doing all of this work on a release branch, the develop branch is cleared to receive features for the next big release.

release分支支持新产品发布的准备。它们允许在最后一分钟dotting of i’s and crossing t’s，此外，它们还允许小错误修复和为发布准备元数据（版本号、构建日期等）。在release分支上完成所有这些工作的同时，develop分支可以准备接收下一个大型release的特性

>The key moment to branch off a new release branch from develop is when develop (almost) reflects the desired state of the new release. At least all features that are targeted for the release-to-be-built must be merged in to develop at this point in time. All features targeted at future releases may not—they must wait until after the release branch is branched off.

从develop中分出一个新的release分支的关键时刻是develop(几乎)反映了新版本的期望状态。至少所有针对待构建版本的特性都已经合并进develop分支。针对未来版本的所有特性不会加入进来 - 它们必须等到release分支分离之后

>It is exactly at the start of a release branch that the upcoming release gets assigned a version number—not any earlier. Up until that moment, the develop branch reflected changes for the “next release”, but it is unclear whether that “next release” will eventually become 0.3 or 1.0, until the release branch is started. That decision is made on the start of the release branch and is carried out by the project’s rules on version number bumping.

只有在版本分支的开始，即将发布的版本才被分配了一个版本号 - 而不是更早。在此之前，develop分支反映了“下一个版本”变化，但是在版本分支开始之前，尚不清楚“下一个版本”最终会变成0.3还是1.0。这个决定是在版本分支开始时做出的，并由项目的版本号碰撞规则来执行

#### Creating a release branch

创建版本分支

>Release branches are created from the develop branch. For example, say version 1.1.5 is the current production release and we have a big release coming up. The state of develop is ready for the “next release” and we have decided that this will become version 1.2 (rather than 1.1.6 or 2.0). So we branch off and give the release branch a name reflecting the new version number:

release分支从develop分支中fork。例如，假设版本1.1.5是当前的生产版本，我们即将发布一个大版本。develop分支已经为“下一个版本”做好了准备，我们已经决定这将成为1.2版(而不是1.1.6或2.0版)。因此，从develop分支fork一个版本分支，并使用反映新版本号的名称:

```
$ git checkout -b release-1.2 develop
Switched to a new branch "release-1.2"

$ ./bump-version.sh 1.2
Files modified successfully, version bumped to 1.2.

$ git commit -a -m "Bumped version number to 1.2"
[release-1.2 74d9424] Bumped version number to 1.2
1 files changed, 1 insertions(+), 1 deletions(-)
```

>After creating a new branch and switching to it, we bump the version number. Here, bump-version.sh is a fictional shell script that changes some files in the working copy to reflect the new version. (This can of course be a manual change—the point being that some files change.) Then, the bumped version number is committed.

在创建了一个新的分支并切换到它之后，我们会增加版本号。这里，bump-version.sh是一个虚构的shell脚本，它会更改工作副本中的一些文件以反映新版本(这当然可以是手动更改)。然后，提交新的版本分支

>This new branch may exist there for a while, until the release may be rolled out definitely. During that time, bug fixes may be applied in this branch (rather than on the develop branch). Adding large new features here is strictly prohibited. They must be merged into develop, and therefore, wait for the next big release.

这个新的分支可能会在那里存在一段时间，直到发行版明确推出。在此期间，bug修复可能会应用于这个分支(而不是develop分支)。严禁在此添加大型新功能。它们必须被合并到develop中，然后等待下一个大版本

#### Finishing a release branch

结束版本分支

>When the state of the release branch is ready to become a real release, some actions need to be carried out. First, the release branch is merged into master (since every commit on master is a new release by definition, remember). Next, that commit on master must be tagged for easy future reference to this historical version. Finally, the changes made on the release branch need to be merged back into develop, so that future releases also contain these bug fixes.

当release分支的状态准备好成为真正的版本时，需要执行一些操作。首先，release分支被合并到master中(因为根据定义，master上的每个提交都是新的release，请记住)。接下来，master上的提交必须被标记（tagged），以便于将来参考这个历史版本。最后，在release分支上所做的更改需要合并回develop中，以便将来的发布也包含这些bug修复

>The first two steps in Git:

前两步Git操作如下：

```
$ git checkout master
Switched to branch 'master'

$ git merge --no-ff release-1.2
Merge made by recursive.
(Summary of changes)

$ git tag -a 1.2
```

>The release is now done, and tagged for future reference.

新版本完成后需要标记，以便后续参考

>Edit: You might as well want to use the -s or -u <key> flags to sign your tag cryptographically.

新增：不妨使用-s或-u <key>标志对标签进行加密

>To keep the changes made in the release branch, we need to merge those back into develop, though. In Git:

为了保持版本分支中的变更，需要进一步合并回develop分支

```
$ git checkout develop
Switched to branch 'develop'
$ git merge --no-ff release-1.2
Merge made by recursive.
(Summary of changes)
```

>This step may well lead to a merge conflict (probably even, since we have changed the version number). If so, fix it and commit.

这一步很可能会导致合并冲突(甚至可能，因为我们已经更改了版本号)。如果存在，修复它并提交

>Now we are really done and the release branch may be removed, since we don’t need it anymore:

完成上述操作后就可以移除版本分支了

```
$ git branch -d release-1.2
Deleted branch release-1.2 (was ff452fe).
```

### Hotfix branches

热修复分支

>May branch off from: **master**
>Must merge back into: **develop and master**

热修复分支可以从master分支进行fork，它必须合并到develop和master分支

> Branch naming convention:
>* hotfix-*

分支命名规范：**hotfix-***

![](/imgs/gitflow/hotfix-branches@2x.png)

>Hotfix branches are very much like release branches in that they are also meant to prepare for a new production release, albeit unplanned. They arise from the necessity to act immediately upon an undesired state of a live production version. When a critical bug in a production version must be resolved immediately, a hotfix branch may be branched off from the corresponding tag on the master branch that marks the production version.

热修复分支非常类似于版本分支，因为它们也意味着为新的生产发布做准备，尽管是计划外的。它们产生于必须立即对实时生产版本的不期望状态采取行动。当生产版本中的关键错误必须立即解决时，热修复分支会从master分支上标记生产版本的相应标记中分离出来

>The essence is that work of team members (on the develop branch) can continue, while another person is preparing a quick production fix.

本质上团队成员(在develop分支上)的工作可以继续，而另一个人正在准备快速的生产修复

#### Creating the hotfix branch

创建热修复分支

>Hotfix branches are created from the master branch. For example, say version 1.2 is the current production release running live and causing troubles due to a severe bug. But changes on develop are yet unstable. We may then branch off a hotfix branch and start fixing the problem:

热修复分支是从master分支创建的。例如，假设版本1.2是当前运行的生产版本，由于一个严重的错误导致了问题。但是在develop分支上的变化仍然不稳定，我们可以fork一个热修复分支，并开始修复问题:

```
$ git checkout -b hotfix-1.2.1 master
Switched to a new branch "hotfix-1.2.1"

$ ./bump-version.sh 1.2.1
Files modified successfully, version bumped to 1.2.1.

$ git commit -a -m "Bumped version number to 1.2.1"
[hotfix-1.2.1 41e61bb] Bumped version number to 1.2.1
1 files changed, 1 insertions(+), 1 deletions(-)
```

>Don’t forget to bump the version number after branching off!

不要忘记在fork分支后修改版本号

>Then, fix the bug and commit the fix in one or more separate commits.

然后修复bug并在一次或多次提交中提交

```
$ git commit -m "Fixed severe production problem"
[hotfix-1.2.1 abbe5d6] Fixed severe production problem
5 files changed, 32 insertions(+), 17 deletions(-)
```

#### Finishing a hotfix branch

结束热修复分支

>When finished, the bugfix needs to be merged back into master, but also needs to be merged back into develop, in order to safeguard that the bugfix is included in the next release as well. This is completely similar to how release branches are finished.

完成后，修复的bug需要合并回master，也需要合并回develop，以确保bugfix也包含在下一个版本中。这完全类似于release分支的完成方式

>First, update master and tag the release.

首先，更新master并进行标记

```
$ git checkout master
Switched to branch 'master'

$ git merge --no-ff hotfix-1.2.1
Merge made by recursive.
(Summary of changes)

$ git tag -a 1.2.1
```

>Edit: You might as well want to use the -s or -u <key> flags to sign your tag cryptographically.

新增：不妨使用-s或-u <key>标志对标签进行加密签名

>Next, include the bugfix in develop, too:

下一步，合并热修复分支到develop

```
$ git checkout develop
Switched to branch 'develop'

$ git merge --no-ff hotfix-1.2.1
Merge made by recursive.
(Summary of changes)
```

>The one exception to the rule here is that, **when a release branch currently exists, the hotfix changes need to be merged into that release branch, instead of** develop. Back-merging the bugfix into the release branch will eventually result in the bugfix being merged into develop too, when the release branch is finished. (If work in develop immediately requires this bugfix and cannot wait for the release branch to be finished, you may safely merge the bugfix into develop now already as well.)

**这里存在一个额外的规则，当一个release分支当前存在时，修补程序更改需要合并到该release分支中，而不是develop**。将bugfix合并到release分支中最终会导致bugfix也被合并到develop中。(如果develop分支的工作立即需要此bugfix，并且不能等待release分支完成，那么也可以将该bugfix安全地合并到现在的develop分支中)

>Finally, remove the temporary branch:

最后，移除临时的热修复分支

```
$ git branch -d hotfix-1.2.1
Deleted branch hotfix-1.2.1 (was abbe5d6).
```

## Summary

总结

>While there is nothing really shocking new to this branching model, the "big picture" figure that this post began with has turned out to be tremendously useful in our projects. It forms an elegant mental model that is easy to comprehend and allows team members to develop a shared understanding of the branching and releasing processes.

虽然这个分支模型没有什么真正令人震惊的新东西，但是这篇文章开头的“大图”在我们的项目中已经证明是非常有用的。它形成了一个优雅的脑图，易于理解，并允许团队成员对分支和发布过程形成共同的理解