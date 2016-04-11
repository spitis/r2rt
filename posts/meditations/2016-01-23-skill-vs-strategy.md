---
title: Skill vs Strategy
author: Silviu Pitis
---

_In this post I discuss several real life examples of strategies that cannot be achieved through mere practice of an inferior strategy. Backpropagation is an algorithm akin to such "mere practice"--backpropagation develops skill at a specific strategy (i.e., it learns a specific local minimum). Like practice, backpropagation alone cannot result in a switch to a superior strategy. I look at how strategy switches are achieved in real examples and ask what algorithm might allow machines to effectively switch strategies._

<!--more-->

## Shooting hoops

As a basketball player, I found the difference between the shots of 12 year olds and NBA players fascinating. Compare the shot of 12 year old Julian Newman to that of Dirk Nowitzki:

<img src="/images/bball_dirk.gif" alt="Dirk Nowitzki basketball shot" style="max-width: 45%; float: right">
<img src="/images/bball_julian.gif" alt="Julian Newman basketball shot" style="max-width: 45%;">

<!-- Source Videos:
https://youtu.be/FjPNPTcxz1I
http://youtu.be/eDgRMdT1QVI
-->

Julian's shot is as skilled as it gets given his age and physical characteristics. But notice how Julian shoots from his chest. He leans forward and pushes the ball away from his body (a chest shot). This is different from Dirk, who stands tall, raises the ball above his head, and shoots by extending his arm and flicking his wrist (a wrist shot).

As he grows, Julian will soon find opponents' hands inconveniently placed in the path of his shot. He'll also soon be tall enough and strong enough to shoot like Dirk.

What's interesting is that Julian's chest shot skill will not transfer perfectly. The only way he can switch to a wrist shot is by training the new style.  

To us observers, this switch may seem obvious. Yet in my experience and that of my basketball-playing friends, as once-skilled chest-shooters, it is anything but. When we were Julian's age, our chest shots came naturally and with higher accuracy (even with an opponent's hand in front). We did switch eventually, perhaps due to a combination of logic and faith in the results of continued practice, or maybe as a simple matter of monkey see monkey do, or perhaps it was Coach Carter's shooting drills that did it. Whatever the reason, the ball now leaves my hands from above my head with a flick of the wrist. I shoot like Dirk, except the part where the ball goes in.

## The Fosbury Flop

The [Fosbury Flop](https://en.wikipedia.org/wiki/Fosbury_Flop) is to Olympic high jumping as the wrist shot is to shooting a basketball. Though the mid-1960s, Olympic high jumpers would jump using the straddle technique, Western Roll, Eastern cut-off or scissors jump. But in 1968, a young Dick Fosbury set a new world record with his unorthodox new method of flopping backward over the bar. The Fosbury Flop was born.

Compare the dominant pre-Fosbury style, the straddle technique, to the Fosbury Flop:

<img src="/images/fosbury.gif" alt="Fosbury Flop" style="max-width: 45%; min-width: 90px; float: right">
<img src="/images/straddle.gif" alt="Straddle technique" style="max-width: 45%; min-width: 90px;">

<!-- Source Videos:
https://www.youtube.com/watch?v=d6lpk_9T5hM
https://www.youtube.com/watch?v=Id4W6VA0uLc
-->

Unlike the chest shot, which is clearly more prone to being caught under an opponent's hand than the wrist shot, it's not clear that the Fosbury Flop is any better than the straddle technique. Indeed, when Fosbury first used the Flop, he was met with warnings from his coaches. And even today, you will find high jumpers debating the relative merits of each jumping style. But the Fosbury Flop caught on, and has held every world record since the straddle technique's last record in 1978.

## As easy as 17 plus 24

Another example, one of my favorites, concerns a fundamental human skill. Unlike the wrist shot for basketball players and the Fosbury Flop for high jumpers, this technique is relatively unknown among those who stand to benefit.

Sometime in middle school, my music teacher recounted the story of a student who did arithmetic in his head faster than with a calculator. His secret? While school had taught us to add right-to-left, this student added left-to-right. A proud skeptic, I tested his technique out for myself. After a little practice, I too was summing numbers faster than with a calculator. It turns out that adjusting your answers for carried 1s isn't as hard as it seems. In math class I had just  started algebra, but in music, I had finally learned to add correctly.

## Mayan astronomy

As a final fun example, take a minute to watch Richard Feynman's [imaginary discussion between a Mayan astronomer and his student](https://youtu.be/NM-zWTU7X-k?t=3m52s):

<iframe width="420" height="315" src="https://www.youtube.com/embed/NM-zWTU7X-k?start=232" frameborder="0"></iframe>

## Well-executed strategies are local minima

Similar situations are common. For most any task there exist multiple strategies of varying effectiveness: there are different ways to perform in competitive sports, different ways to study, and even different ways to think about life and pursue happiness. In the language of machine learning, this is simply to say that real world situations involve non-convex objective functions with multiple local minima. The dip around each local minimum corresponds to a distinct strategy, and the minimum itself corresponds to the perfect execution of that strategy. Under this lens, _backpropagation is an algorithm for improving skill_.

Perhaps unsurprisingly, a skillfully-executed next-best strategy works in each of the above examples. It may not be as effective as the best strategy, but it gets the job done (else it wouldn't really be a strategy). This is similar to the empirical result that despite the non-convex nature of a neural network's error surface, backpropagation generally converges to a local minimum that performs well. A good way to think about this might be that until something "sort of works" (i.e., it has some semblance of a strategy), it won't even start to converge; you can't be skilled at a non-strategy.

In spite of the efficacy of the next-best strategy, the examples above demonstrate that a superior strategy can perform significantly better. A superior strategy can also open doors that were previously closed in related domains (e.g., our ancestors needed to develop opposable thumbs before they could develop writing). This is critical to creating strong artificial intelligence, and so begs the question: what prompts the switch to a superior strategy and how can we use this to build stronger AI?

## What's in a strategy; why random guesses won't work

Implicit in a strategy is an _objective_, which can be shooting a basketball through a hoop or minimizing an arbitrary loss function.

A strategy is executed within a _context_, which can make reaching the objective easier. It's easier to shoot a basketball if you are a foot taller, and its easier to train a model when you have more data.

Because some characteristics of the context are within our control, our strategy will often first involve developing these characteristics. In basketball, you can shoot from close up by passing the ball or crossing your opponent. In training a neural network, you can gather more data or select better features.

As developing these characteristics can take time, and different strategies often rely on different characteristics, switching strategies can be costly. While changing the structure of a small neural network may be relatively inexpensive, retraining a lawyer to be a software engineer can be difficult due to the large investment required to develop domain expertise. Therefore, we'll often need a compelling reason to prompt a strategy switch; something more than a random guess.

Another reason to rule out random guesses is the very reason we need the backpropagation algorithm to being with. There are infinitely many random guesses we could make. We simply do not have infinite time to make them, and so whereas backpropagation and its variants are required meta-strategies that tell us how to converge to a working strategy, we might wonder whether there are other meta-strategies that tell us how to effectively switch strategies.

## Switching strategies through social learning

One such meta-strategy is social learning, or learning by external influence. Whether it's the Coach's shooting drills, seeing other players hit threes using their wrist, watching Dick Fosbury set a world record with his Flop, or hearing a story of how a math whiz does mental addition faster than a calculator. In my limited experience, there is not yet a true parallel for this in the world of artificial intelligence.

Our machine learning models do switch strategies due to the external influence of their creators; for example, vision researches were presented with the so called "AlexNet" architecture in 2012 (described in this [paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)), and the model has since been copied and further improved. Unfortunately, such improvement is not machine autonomous, but requires the direct interference of a human. There are cases where a model is improved by mimicking a database of past samples, but it is difficult to call such programmed mimicry autonomous or intelligent.

One big question then is, how can we get a machine to learn socially? Is there a way to do this on a more sophisticated level than averaging models or copying weight variables? There must be, because humans can do it.

Being able to accept external influence requires the ability to understand the process that is producing the result. We can see and communicate the precise difference between a chest shot and a wrist shot and so we can accept the external influences and learn via imitation. But even human communication is limited to concepts that the receiver understands. A chess amateur may gain little by watching a grandmaster play whereas a master might learn a great deal. I think the day our programs are able to understand higher order concepts so as to learn by external influence we will be _very_ close to artificial general intelligence.

## Strategies of first convergence, or why social learning is not enough

With social learning, our machines will be able to stand on the shoulders of giants. But what if there are no giants?

While it is certainly possible to first converge to the best strategy, the chest shot in basketball shows us that our first strategy may be biased. As kids grow up, they are not strong or tall enough to pull of a wrist shot and inevitably converge to a chest shot. By the time they grow up, their chest shot may be a local minima that they can never escape without some creativity or outside help.

Suppose we are trying to classify images of people as male or female, and train an image classification neural network with backpropagation to do so. By starting with a random initialization, the network reaches a local minimum. The question is: can we be sure that this local minimum is close to the best one? We might try testing this by reinitializing the weights and retraining the network. Let's say we retrain the network one million times, and each of the local minima reached leads to approximately the same performance. Is this enough for us to conclude that the resulting strategies are close to the best? I would answer in the negative; we cannot be certain that a random initialization will ever lead to an optimal strategy via backpropagation. It may be a situation like the chest shot, where in order to reach an optimal strategy, the network must be trained again after it has learned some useful hidden features.

It's possible, for example, that height is such a good first proxy that neural networks trained with backpropagation immediately learn to use, and even heavily rely, on height as a feature. Humans know that while height is correlated with gender, more subtle characteristics like facial structure are superior predictors. It's possible that neural networks trained with just backpropagation, even if they eventually learn to use facial structure, will never be able to change their strategy completely and "unlearn" the use of height.

Therefore, even if machines are able to learn strategies from each other, it may not be enough to produce the Fosbury Flop or the theory of General Relativity. _Monkey see, monkey do_ is not enough for true intelligence: intelligent machines must be able to produce new strategies independently.  

## Switching strategies through creativity

The ability to switch strategies without external influence is the ultimate mark of intelligence. It takes something more than training to stop what one is currently doing and try something else entirely. You need to have a hypothesis that another method will be superior before you try it. In the basketball example, the opponent might stick a hand in your face while you're trying to shoot, and it might prompt the thought: "if only I could shoot from higher up." In isolation, backpropagating neural networks cannot have these sorts of thoughts about their weights and structures.

The key to independent strategy switches is the hypothesis--a guess. Per Feynman, this is the core of the scientific method:

<iframe width="420" height="315" src="https://www.youtube.com/embed/EYPapE-3FRw" frameborder="0"></iframe>

As Feynman notes, however, the guesses are not random--some guesses are better than others. Our task then, is to figure out an algorithm for making effective guesses: an algorithm for creativity.
