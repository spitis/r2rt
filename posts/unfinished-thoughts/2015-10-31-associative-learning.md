---
title: Associative Learning
author: Silviu Pitis
---

_The distinction between supervised vs unsupervised learning, which presumably arose due to the structure of data and early input-output models, is not well suited in the context of general intelligence. It may be better to consider intelligence not as a function, f(x) = y, but as a (semi-)symmetric relation (i.e., an association)._

<!--more-->

Pick up a book or a course on machine learning and they all start with the same distinction: _supervised learning_ vs _unsupervised learning_ (some add _reinforcement learning_ as a third category). In the context of intelligence (as opposed to learning from data), this distinction is silly.

Are humans supervised or unsupervised learners? A baby learning to call mom "mama" and dad "dada" is supervised learning in the following sense: the baby hears the label "ma" primarily when interacting with mom and the label "da" primarily when interacting with dad.

But rewind a bit. The baby can distinguish between mom and dad a lot sooner than they know the labels _mom_ and _dad_. This learning occurs with labels, so it's unsupervised. Or is it? Why is the label _dad_ any more significant than the _image of dad_, the _sound of dad's voice_, the _smell of dad's aftershave_?

Although these other projections of dad cannot be communicated as easily as the label _dad_ (consider Charades), they can serve, to different degrees, as labels for dad. Imagine a world in which humans, instead of making arbitrary sounds with their vocal chords, emitted projections of visual images--what role would the _image of dad_ play?

It seems weird to think that humans learn in two different ways. If we must force a dichotomy on methods of human learning, I think there are better distinctions. For example, we might think of learning as either associative (e.g., inductive) or logical (e.g., deductive). This dichotomy is not perfect either (e.g., analogical reasoning inductive or deductive?), but it seems more reasonable and covers a wider range of human learning than does the unsupervised/supervised distinction: both supervised and unsupervised learning, as commonly understood, form a part of associative or inductive reasoning; logical or deductive thought is something else entirely.

The human is an _association engine_. We do not simply classify things into "dogs" or "cats" (e.g., is a wolf a dog?). Nor do we say those dogs are “unlabeled Group A” and those cats are “unlabeled Group B.” Instead we say, that is a dog because it looks like this animal and that animal, which I've associated with the word dog. Both tasks are occurring. And actually, both tasks are very much the same task: New animal looks like dog. Dog is labelled a “dog”. “Looks like” and “is labelled" are both the same as “produces the thought of" or "is associated with."

A dog sound vector, a dog word (grammatical usage) vector, a dog image vector, etc. should be associated, and it's not clear that the pure concept of dog (i.e., the label "dog") should be an intermediary between them. In reality, there is no true concept of a dog; only dog-like objects. Or if such a dog concept does exist, it is a flexible concept.

The _association engine_ model, or _associative learning_ both supervised or unsupervised. It is unsupervised in the sense that there is no special y-vector with labels. Rather the label is itself is part of the input data.

Associative learning is saying that instead of a one-hot label, we should use a more flexible distributed representation of the label.
