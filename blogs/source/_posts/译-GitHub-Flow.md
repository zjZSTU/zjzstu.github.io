---
title: '[译]GitHub Flow'
categories:
  - [翻译]
  - [版本控制, 规范, workflow]
  - [版本控制, 托管平台]
  - [工具]
tags:
  - git
  - github
  - github-flow
abbrlink: a20843e9
date: 2019-11-30 17:23:50
---

原文地址：[GitHub Flow](http://scottchacon.com/2011/08/31/github-flow.html)

## Issues with git-flow

git-flow的问题

>I travel all over the place teaching Git to people and nearly every class and workshop I’ve done recently has asked me what I think about [git-flow](http://nvie.com/posts/a-successful-git-branching-model/). I always answer that I think that it’s great - it has taken a system (Git) that has a million possible workflows and documented a well tested, flexible workflow that works for lots of developers in a fairly straightforward manner. It has become something of a standard so that developers can move between projects or companies and be familiar with this standardized workflow.

我走遍各地教人们Git，几乎我最近做的每一堂课和研讨会都问我对git-flow的看法。我总是回答，我认为它是伟大的 - 它采用了一个有一百万种可能的工作流程的系统（Git），并记录了一个测试良好的，灵活的工作流程，以相当简单的方式为许多开发人员工作。它已经成为一种标准，开发人员可以在项目或公司之间移动，并熟悉这种标准化的工作流

>However, it does have its issues. I have heard a number of opinions from people along the lines of not liking that new feature branches are started off of develop rather than master, or the way it handles hotfixes, but those are fairly minor.

然而，它确实有它的问题。我听到了很多人的意见，他们不喜欢新的特性分支是从develop开始的，而不是从master开始的，或者它处理hotfix的方式，但是这些都是相当小的问题

>One of the bigger issues for me is that it’s more complicated than I think most developers and development teams actually require. It’s complicated enough that a big [helper script](https://github.com/nvie/gitflow) was developed to help enforce the flow. Though this is cool, the issue is that it cannot be enforced in a Git GUI, only on the command line, so the only people who have to learn the complex workflow really well, because they have to do all the steps manually, are the same people who aren’t comfortable with the system enough to use it from the command line. This can be a huge problem.

对我来说，一个更大的问题是它比我认为大多数开发人员和开发团队实际需要的要复杂。它已经足够复杂到需要开发一个大的助手脚本来帮助实施流程了。尽管这很酷，但问题是它不能在Git图形用户界面中强制执行，只能在命令行上强制执行，所以人们必须非常好地学习这个复杂的工作流程，因为必须手动完成所有步骤，如果对系统不太适应，将无法从命令行使用它。这可能是个大问题

>Both of these issues can be solved easily just by having a much more simplified process. At GitHub, we do not use git-flow. We use, and always have used, a much simpler Git workflow.

这两个问题都可以通过简单得多的流程轻松解决。在GitHub，我们不使用git-flow。我们使用并且一直使用一个简单得多的Git工作流

>Its simplicity gives it a number of advantages. One is that it’s easy for people to understand, which means they can pick it up quickly and they rarely if ever mess it up or have to undo steps they did wrong. Another is that we don’t need a wrapper script to help enforce it or follow it, so using GUIs and such are not a problem.

它的简单性给了它许多优点。一是人们很容易理解，这意味着他们可以很快学会，而且他们很少会把事情搞砸或者不得不撤销他们做错的步骤。另一个是，我们不需要辅助脚本来帮助执行或遵循它，所以可以通过图形用户界面使用

## GitHub Flow

>So, why don’t we use git-flow at GitHub? Well, the main issue is that we deploy all the time. The git-flow process is designed largely around the "release". We don’t really have "releases" because we deploy to production every day - often several times a day. We can do so through our chat room robot, which is the same place our CI results are displayed. We try to make the process of testing and shipping as simple as possible so that every employee feels comfortable doing it.

那么，为什么我们不在GitHub上使用git-flow呢？主要问题是我们一直在部署。git-flow主要是围绕“release”设计的。我们并没有真正的“release”，因为我们每天都部署到生产中 - 通常一天几次。我们可以通过聊天室机器人这样做，这也是我们的显示CI结果的地方。我们努力使测试和发布过程尽可能简单，以便每位员工都能轻松完成

>There are a number of advantages to deploying so regularly. If you deploy every few hours, it’s almost impossible to introduce large numbers of big bugs. Little issues can be introduced, but then they can be fixed and redeployed very quickly. Normally you would have to do a 'hotfix' or something outside of the normal process, but it’s simply part of our normal process - there is no difference in the GitHub flow between a hotfix and a very small feature.

定期部署有很多好处。如果每隔几个小时部署一次，几乎不可能引入大量的大错误。可以引入一些小问题，但是可以很快修复和重新部署。通常情况下，您必须在正常流程之外做一个‘热修复’，但这只是我们正常流程的一部分 - 热修复在GitHub流程中是一个非常小的特性

>Another advantage of deploying all the time is the ability to quickly address issues of all kinds. We can respond to security issues that are brought to our attention or implement small but interesting feature requests incredibly quickly, yet we can use the exact same process to address those changes as we do to handle normal or even large feature development. It’s all the same process and it’s all very simple.

持续部署的另一个优势是能够快速解决各种问题。我们可以对引起我们注意的安全问题做出响应，或者以惊人的速度实现小而有趣的特征请求，但是我们可以使用与处理正常甚至大的特征开发完全相同的过程来解决这些变化。这都是同一个过程，都很简单

## How We Do It

如何实现它

>So, what is GitHub Flow?
>* Anything in the master branch is deployable
>* To work on something new, create a descriptively named branch off of master (ie: new-oauth2-scopes)
>* Commit to that branch locally and regularly push your work to the same named branch on the server
>* When you need feedback or help, or you think the branch is ready for merging, open a [pull request](http://help.github.com/send-pull-requests/)
>* After someone else has reviewed and signed off on the feature, you can merge it into master
>* Once it is merged and pushed to 'master', you can and should deploy immediately

所以，什么是GitHub流？

* master分支中的任何东西都是可部署的
* 要开发新的东西，从master分支中创建一个描述性命名的分支(比如：new-oauth2-scopes)
* 在本地提交到该分支，并定期将您的工作推送到服务器上的同一个命名分支
* 当您需要反馈或帮助，或者您认为分支已经准备好合并时，可以提交一个推送请求（PR）
* 在其他人审阅并签署了该功能后，可以将其合并到master中
* 一旦它被合并并推送到"主服务器"，就可以并且应该立即部署

>That is the entire flow. It is very simple, very effective and works for fairly large teams - GitHub is 35 employees now, maybe 15-20 of whom work on the same project (github.com) at the same time. I think that most development teams - groups that work on the same logical code at the same time which could produce conflicts - are around this size or smaller. Especially those that are progressive enough to be doing rapid and consistent deployments.

这就是全部流程。它非常简单，非常有效，并且为相当大的团队工作 - GitHub现在有35名员工，其中可能有15-20人同时在同一个项目(github.com)上工作。我认为大多数开发团队 - 在同一时间处理同一逻辑代码的团队可能会产生冲突 - 都在这个规模或更小的范围内。尤其是那些进步到足以进行快速的一致性部署的公司

>So, let’s look at each of these steps in turn.

让我们依次看看这些步骤

### anything in the master branch is deployable

>This is basically the only hard rule of the system. There is only one branch that has any specific and consistent meaning and we named it master. To us, this means that it has been deployed or at the worst will be deployed within hours. It’s incredibly rare that this gets rewound (the branch is moved back to an older commit to revert work) - if there is an issue, commits will be reverted or new commits will be introduced that fixes the issue, but the branch itself is almost never rolled back.

这基本上是该系统唯一的硬性规定。只有一个分支具有任何特定和一致的含义，我们称之为“master”。对我们来说，这意味着它已经部署，或者最坏的情况是将在几小时内部署。这种情况非常罕见(分支被移回到旧的提交来恢复工作) - 如果有问题，提交将被恢复或者引入新的提交来修复问题，但是分支本身几乎从来没有被回滚过

>The master branch is stable and it is always, always safe to deploy from it or create new branches off of it. If you push something to master that is not tested or breaks the build, you break the social contract of the development team and you normally feel pretty bad about it. Every branch we push has tests run on it and reported into the chat room, so if you haven’t run them locally, you can simply push to a topic branch (even a branch with a single commit) on the server and wait for Jenkins to tell you if it passes everything.

master分支是稳定的，从master分支部署或从master分支创建新分支总是安全的。如果你把一些未经测试或破坏构建的东西推给master，你就破坏了开发团队的约定，你应该为此感到非常难过。我们推送给master的每个分支都经过测试并报告给聊天室，所以如果您没有在本地运行它们，您只要简单地推送至服务器上的指定分支(甚至是具有单个提交的分支)，并等待Jenkins告诉您它是否通过了所有操作

>You could have a deployed branch that is updated only when you deploy, but we don’t do that. We simply expose the currently deployed SHA through the webapp itself and curl it if we need a comparison made.

您可以拥有一个只在部署时才更新的已部署分支，但我们不会这样做。我们只需通过webapp公开当前部署的SHA，如果我们需要进行比较，就可以下载它

### create descriptive branches off of master

>When you want to start work on anything, you create a descriptively named branch off of the stable master branch. Some examples in the GitHub codebase right now would be user-content-cache-key, submodules-init-task or redis2-transition. This has several advantages - one is that when you fetch, you can see the topics that everyone else has been working on. Another is that if you abandon a branch for a while and go back to it later, it’s fairly easy to remember what it was.

当您想要开始任何工作时，您可以从稳定的master分支创建一个描述性命名的分支。GitHub代码库中的一些例子是用户内容缓存键、子模块初始化任务或redis2转换。这有几个优点 - 一是当你获取时，你可以看到其他人都在研究的主题。另一个是，如果你暂时放弃一个分支，然后再回到它，很容易记住它是什么

>This is nice because when we go to the GitHub branch list page we can easily see what branches have been worked on recently and roughly how much work they have on them.

这很好，因为当我们转到GitHub分支列表页面时，我们可以很容易地看到哪些分支最近被处理过，以及它们在这些分支上有多少工作

![](/imgs/github-flow/7988902c-d0a8-11e4-94c9-dc132461ffe4.png)

>It’s almost like a list of upcoming features with current rough status. This page is awesome if you’re not using it - it only shows you branches that have unique work on them relative to your currently selected branch and it sorts them so that the ones most recently worked on are at the top. If I get really curious, I can click on the 'Compare' button to see what the actual unified diff and commit list is that is unique to that branch.

这几乎就像是一个当前粗略状态的即将推出的功能列表。如果你不使用它，这个页面会很棒 - 它只显示相对于你当前选择的分支在它们上面有独特工作的分支，它会对它们进行排序，使得最近工作的分支在顶部。如果我真的很好奇，我可以点击“比较”按钮，看看真正的统一差异和提交列表是哪个分支独有的

>So, as of this writing, we have 44 branches in our repository with unmerged work in them, but I can also see that only about 9 or 10 of them have been pushed to in the last week.

因此，截至本文撰写之时，我们的存储库中有44个分支，其中有未合并的工作，但我也可以看到，在过去的一周中，只有大约9到10个分支被推送

### push to named branches constantly

>Another big difference from git-flow is that we push to named branches on the server constantly. Since the only thing we really have to worry about is master from a deployment standpoint, pushing to the server doesn’t mess anyone up or confuse things - everything that is not master is simply something being worked on.

与git-flow的另一大区别是，我们不断地在服务器上推送命名分支。因为从部署的角度来看，我们唯一真正需要担心的是部署时候的master，所以向服务器推送不会让任何人混乱或事物混淆 - 所有不是master的东西都只是正在处理的事情

>It also make sure that our work is always backed up in case of laptop loss or hard drive failure. More importantly, it puts everyone in constant communication. A simple 'git fetch' will basically give you a TODO list of what every is currently working on.

它还能确保在笔记本电脑丢失或硬盘出现故障时，我们的工作始终得到备份。更重要的是，它让每个人都保持持续的沟通。一个简单的“git fetch”基本上会给你一个待办事项列表，列出每个人目前正在做的事情

```
$ git fetch
remote: Counting objects: 3032, done.
remote: Compressing objects: 100% (947/947), done.
remote: Total 2672 (delta 1993), reused 2328 (delta 1689)
Receiving objects: 100% (2672/2672), 16.45 MiB | 1.04 MiB/s, done.
Resolving deltas: 100% (1993/1993), completed with 213 local objects.
From github.com:github/github
 * [new branch]      charlock-linguist -> origin/charlock-linguist
 * [new branch]      enterprise-non-config -> origin/enterprise-non-config
 * [new branch]      fi-signup  -> origin/fi-signup
   2647a42..4d6d2c2  git-http-server -> origin/git-http-server
 * [new branch]      knyle-style-commits -> origin/knyle-style-commits
   157d2b0..d33e00d  master     -> origin/master
 * [new branch]      menu-behavior-act-i -> origin/menu-behavior-act-i
   ea1c5e2..dfd315a  no-inline-js-config -> origin/no-inline-js-config
 * [new branch]      svg-tests  -> origin/svg-tests
   87bb870..9da23f3  view-modes -> origin/view-modes
 * [new branch]      wild-renaming -> origin/wild-renaming
```

>It also lets everyone see, by looking at the GitHub Branch List page, what everyone else is working on so they can inspect them and see if they want to help with something.

它还可以让每个人通过查看GitHub分支列表页面，看到其他人正在做什么，这样就可以检查并且看看他们是否需要帮忙

### open a pull request at any time

>GitHub has an amazing code review system called [Pull Requests](http://help.github.com/send-pull-requests/) that I fear not enough people know about. Many people use it for open source work - fork a project, update the project, send a pull request to the maintainer. However, it can also easily be used as an internal code review system, which is what we do.

GitHub有一个惊人的代码审查系统，叫做Pull Requests(PRs)，恐怕没有足够的人知道。许多人将它用于开源工作 - fork一个项目，更新项目，向维护者发送一个拉请求。然而，它也可以很容易地用作内部代码审查系统，这就是我们所做的

>Actually, we use it more as a branch conversation view more than a pull request. You can send pull requests from one branch to another in a single project (public or private) in GitHub, so you can use them to say "I need help or review on this" in addition to "Please merge this in".

实际上，我们更多地将其用作分支对话视图，而不是录取请求。您可以在GitHub中的单个项目(公共或私有)中将请求从一个分支发送到另一个分支，这样除了“请求合并”之外，您还可以使用它们来说“我需要帮助或审阅”

![](/imgs/github-flow/61a2dcba-d0a8-11e4-9924-3576232053ee.png)

>Here you can see Josh cc’ing Brian for review and Brian coming in with some advice on one of the lines of code. Further down we can see Josh acknowledging Brian’s concerns and pushing more code to address them.

在这里，你可以看到乔什给布莱恩发了评论，布莱恩进来了，并对其中一行代码提出了一些建议。在更远的地方，我们可以看到乔希承认布赖恩的担忧，并推动更多的代码来解决它们

![](/imgs/github-flow/5054b4ba-d0a8-11e4-8d38-548ecf157018.png)

>Finally you can see that we’re still in the trial phase - this is not a deployment ready branch yet, we use the Pull Requests to review the code long before we actually want to merge it into master for deployment.

最后，您可以看到，我们仍然处于试验阶段 - 这还不是一个部署就绪分支，我们使用PRs来检查代码，直到我们真正想要将它合并到master代码中进行部署

>If you are stuck in the progress of your feature or branch and need help or advice, or if you are a developer and need a designer to review your work (or vice versa), or even if you have little or no code but some screenshot comps or general ideas, you open a pull request. You can cc people in the GitHub system by adding in a @username, so if you want the review or feedback of specific people, you simply cc them in the PR message (as you saw Josh do above).

如果你被困在你的特性或分支的进程中，需要帮助或建议，或者如果你是一个开发人员，需要一个设计者来回顾你的工作(反之亦然)，或者即使你只有很少或没有代码，但有一些截图或一般想法，你可以打开一个拉取请求。您可以在GitHub系统中添加@username来抄送人，所以如果您想要特定人员的评论或反馈，您只需在公关信息中抄送他们(正如您在上面看到的Josh所做的那样)

>This is cool because the Pull Request feature let’s you comment on individual lines in the unified diff, on single commits or on the pull request itself and pulls everything inline to a single conversation view. It also let you continue to push to the branch, so if someone comments that you forgot to do something or there is a bug in the code, you can fix it and push to the branch, GitHub will show the new commits in the conversation view and you can keep iterating on a branch like that.

这很酷，因为PR功能让您可以在统一差异、单次提交或PR本身中对单独的行进行评论，并将所有内容内嵌到单个对话视图中。它还允许您继续推送到分支，因此如果有人评论您忘记做某事或代码中有错误，您可以修复它并推送到分支，GitHub将在对话视图中显示新提交，您可以继续这样迭代分支

>If the branch has been open for too long and you feel it’s getting out of sync with the master branch, you can merge master into your topic branch and keep going. You can easily see in the pull request discussion or commit list when the branch was last brought up to date with the 'master'.

如果分支已经打开太久，并且您觉得它与主分支不同步，您可以将主分支合并到您的主题分支中，然后继续。您可以很容易地在PR讨论或提交列表中看到分支最后一次更新为“master”时的情况

![](/imgs/github-flow/2162f69e-d0a8-11e4-8c98-d2bb581f7152.png)

>When everything is really and truly done on the branch and you feel it’s ready to deploy, you can move on to the next step.

当分支上的所有工作都真正完成，并且您觉得已经准备好部署时，您可以继续下一步

### merge only after pull request review

>We don’t simply do work directly on master or work on a topic branch and merge it in when we think it’s done - we try to get signoff from someone else in the company. This is generally a +1 or emoji or ":shipit:" comment, but we try to get someone else to look at it.

我们不只是直接在master上工作或者在主题分支上工作，当我们认为已经完成的时候，我们会把它合并进来 - 我们会试着从公司的其他人那里获得签名。这通常是+1或表情符号或“:shipit:”注释，但我们试图让其他人看它

![](/imgs/github-flow/0ea37c4a-d0a8-11e4-8b61-7aa73b7e3b03.png)

>Once we get that, and the branch passes CI, we can merge it into master for deployment, which will automatically close the Pull Request when we push it.

一旦我们得到它，并且分支已通过CI，我们就可以将它合并到master节点进行部署，当我们推送代码时，master将自动关闭PR功能

### deploy immediately after review

...
...

## Conclusion

>Git itself is fairly complex to understand, making the workflow that you use with it more complex than necessary is simply adding more mental overhead to everybody’s day. I would always advocate using the simplest possible system that will work for your team and doing so until it doesn’t work anymore and then adding complexity only as absolutely needed.

Git本身很难理解，让您使用的工作流比必要的更复杂只会给每个人的一天增加更多的精神负担。我总是主张使用对你的团队有用的最简单的系统，这样做直到它不再工作，然后只在绝对需要的时候增加复杂性

>For teams that have to do formal releases on a longer term interval (a few weeks to a few months between releases), and be able to do hot-fixes and maintenance branches and other things that arise from shipping so infrequently, git-flow makes sense and I would highly advocate it’s use.

对于那些必须在更长的时间间隔内(几周到几个月之间)进行正式发布，并且能够进行热修复和维护分支以及其他不频繁部署的团队来说，git-flow是有意义的，我会大力提倡使用它

>For teams that have set up a culture of shipping, who push to production every day, who are constantly testing and deploying, I would advocate picking something simpler like GitHub Flow.

对于已经建立了部署文化的团队，他们每天都在推动生产，不断地测试和部署，我主张选择一些更简单的东西，比如GitHub Flow