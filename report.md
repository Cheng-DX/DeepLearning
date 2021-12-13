# ImageCaptioning

------

### 1.摘要

​		ImageCaptioning（看图说话）是一个较为流行的深度学习任务，该任务概述原文如下：

> ***Image captioning is the task of describing the visual content of an image in natural language, employing a visual understanding system and a language model capable of generating meaningful and syntactically correct sentences.Neuroscience research has clarified the link between human vision and language generation only in the last few years.Similarly, in Artificial Intelligence, the design of architectures capable of processing images and generating language is a very recent matter. The goal of these research efforts is to find the most effective pipeline to process an input image,represent its content, and transform that into a sequence of words by generating connections between visual and textual elements while maintaining the fluency of language.***
>
> [原文链接](**https://arxiv.org/pdf/2107.06912.pdf**)

​		其主要目标是找到最有效的方法来处理输入图像，表示其内容，并通过在保持语言流畅性的同时生成视觉和文本元素之间的联系将其转换为句子并输出。总的来说，该任务是一个集计算机视觉、深度学习、自然语言处理的综合任务，有较高的门槛。本文基于笔者所掌握的知识（见2 技术栈概述），实现了一个小规模但具有较好完成度 Image Captioning模型，同时对Seq2Seq模型进行了改进，使之更加适合本任务的需求。

------

### 2.技术栈概述

#### 2.1 循环神经网络

​		神经网络的出现，可以说是一场革命。因为它可以轻易表达那些难以用数学公式拟合映射关系。如今，有相当多的领域使用神经网络搭建了性能大大优于已有解决方案的模型。那么为什么我们还需要循环神经网络（RNN）呢？这是因为传统神经网络难以处理含有明显序列信息的数据。传统神经网络处理句子时，只能单独的取处理一个个的输入，前一个输入和后一个输入是完全无关。但是，某些任务需要能够更好的处理序列信息，即当前输出依赖于前一个结果，例如经典的隐马尔可夫模型；甚至依赖于前两个结果，即三维隐马尔可夫模型。

​		一个简单的RNN结构如下[(图源)](https://pic4.zhimg.com/80/v2-3884f344d71e92d70ec3c44d2795141f_720w.jpg)

<img src="https://pic4.zhimg.com/80/v2-3884f344d71e92d70ec3c44d2795141f_720w.jpg" alt="img" style="zoom: 45%;" />

​		隐藏层中的**w**可以认为是实现序列中前后单元联系的部分，即隐藏层的值s不仅仅取决于当前这次的输入x，还取决于上一次隐藏层的值s。权重矩阵 W就是隐藏层上一次的值作为这一次的输入的权重。使用公式表达即为：
$$
O_t = g(V\cdot S_t)\\
S_t = f(U\cdot X_t +W\cdot S_{t-1})
$$

#### 2.2 迁移学习	

​		基于本任务的特殊性，需要计算机视觉的基础知识和模型训练经验，对于笔者来说，这两方面的知识相对匮乏，此时可以使用迁移学习来简化模型后见过程。迁移学习，指的是使用现有模型的一部分来帮助搭建现有模型，把为任务 A 开发的模型作为初始点，重新使用在为任务 B 开发模型的过程中。这样可以帮助我们提升训练模型的效率，因为现有的模型可以保障模型的速度、效果、准确率。

​		本文中，笔者使用了Pytorch中内置的预训练卷积神经网络（ ResNet-101）来对图像进行编码，最终将结果加上了一层线性层和用来分类的softmax，以便于和解码器适配。[下图是迁移学习的形象表述👇](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X2pwZy9QbjRTbTBSc0F1aWFkOFplb2R4ZWszNVdtYWN6R29pY0pLM2YwbUR4aWNPelRYaWJ1bmdSaWEwTkk4VnZnUlBWV1RPeFl3V1BvYW41eDdJWHJHbEsxVlpRczlRLzY0MA?x-oss-process=image/format,png)

<img src="https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X2pwZy9QbjRTbTBSc0F1aWFkOFplb2R4ZWszNVdtYWN6R29pY0pLM2YwbUR4aWNPelRYaWJ1bmdSaWEwTkk4VnZnUlBWV1RPeFl3V1BvYW41eDdJWHJHbEsxVlpRczlRLzY0MA?x-oss-process=image/format,png" alt="图片" style="zoom: 67%;" />

#### 2.3 Seq2Seq模型的基本概念

​		Seq2Seq模型是自然语言处理中最为常用，也是最体现自然语言处理特殊之处的语言模型。其基本思想是将一个序列转换为另一个序列。它通过使用循环神经网络或更常见的LSTM或GRU来避免梯度消失的问题。经典的Seq2Seq模型由一个编码器encoder和一个解码器decoder组成，其中编码器将每个项目转换为包含当前数据及其上下文的相应隐藏向量；解码器则可以理解为该过程的逆过程，即将向量转换为输出项，同时使用前一个输出作为下一个输入的上下文，最终生成一个完整的序列。

​		[下图是Seq2Seq模型应用于机器翻译的简单示范👇](https://pic4.zhimg.com/80/v2-3884f344d71e92d70ec3c44d2795141f_720w.jpg)

​	![Image for post](https://miro.medium.com/max/942/1*KtWwvLK-jpGPSnj3tStg-Q.png)

​			过程解释：

1. **在Embedding层，我们根据预先准备的词汇表将单词编码，随后转化为张量**
2. **将张量输入Encoder，得到输入的隐藏状态，送与Decoder**
3. **在解码过程中，首先输入的必须是start令牌，通常标识为<SOS>**
4. **在每个循环步骤中，将上一步（或者上几步）的输出作为输入的一部分，结合当前单元的数据来预测下一个单元的结果**
5. **最终，当输出end令牌（通常标识为<EOS>）时，结束**
6. **将输出的张量经过词汇表转化为单词，输出序列**

#### 2.4 Seq2Seq模型的缺陷及其改进

Seq2Seq模型概念清晰，效果良好，但是仍有许多较为明显的问题，好在目前这些问题都有较为成熟的解决方案。Seq2Seq模型的常见问题及其改良手段如下：

- **信息冗余**。我们使用Seq2Seq的经典任务机器翻译举例：解码器的输入是整个句子，这就意味着每个单词是在整个句子的影响下输出的，但这显然和我们的常识不符。翻译任务中某个单词的具体含义应该和对应部分周围的几个单词关系度更高，如果我们可以为每个部分赋权，增加关系度较高部分对于输出的影响度，应该就可以提升模型的性能。这就是注意力机制[Attention]()的基本思想。

  [下图是注意力机制在机器翻译中的应用👇](https://pic2.zhimg.com/80/v2-a6db10fb178f5e5e486e165051dfa829_720w.jpg)

  ​		我们可以明显地看到，不同单词将更多的权重赋予了相关度更高的单词。通常，注意力机制加持下的Seq2Seq拥有更高的精确度和更快的速度。

  <img src="https://pic2.zhimg.com/80/v2-a6db10fb178f5e5e486e165051dfa829_720w.jpg" alt="img"  />

- **解码器训练效果差**。这通常指在训练过程中，解码器的输出难以控制。例如出现收敛速度极慢、模型不稳定、泛化效果不好等问题。这是解码器的循环结构导致的。我们将前一步的输入当作下一步的输出，尽管这符合了序列的特性，但这也是一个将误差几何倍数放大的过程。通常是某个步骤的输出有一个小差异，差异在循环过程中不断累积放大，最终使得整个输出的差异度极大。于是我们引入[TeacherForcing]()机制。

  > **An interesting technique that is frequently used in dynamical supervised learning tasks is to replace the actual output y(t) of a unit by the teacher signal d(t) in subsequent computation of the behavior of the network, whenever such a value exists. We call this technique teacher forcing.**
  >
  > ​                         *—— [A Learning Algorithm for Continually Running Fully Recurrent Neural Networks](http://ieeexplore.ieee.org/document/6795228/), 1989.*

  ![img](https://img2020.cnblogs.com/blog/1630237/202104/1630237-20210422182310408-901727114.png)简单来说，就是在训练过程中不使用或者是不完全使用上一步的输出作为输入，而是使用标准答案，即教师信号d(t)，这样就会使得每个单元的误差不会被累积。同时，只要对齐教师信号，就能够在不损失序列效果的前提下得到更快的收敛和更佳的效果。

#### 2.5 Pytorch

> 以下关于Pytorch的简介来源于https://www.yiibai.com/pytorch/pytorch_introduction.html

​		PyTorch是一个Python的开源机器学习库。它用于自然语言处理等应用程序。它最初由Facebook人工智能研究小组开发，而优步的Pyro软件则用于概率编程。最初，PyTorch由Hugh Perkins开发，作为基于Torch框架的LusJIT的Python包装器。有两种PyTorch变种。PyTorch在Python中重新设计和实现Torch，同时为后端代码共享相同的核心C库。PyTorch开发人员调整了这个后端代码，以便有效地运行Python。他们还保留了基于GPU的硬件加速以及基于Lua的Torch的可扩展性功能。

​		下面是TensorFlow和PyTorch之间的主要区别：

| PyTorch                                                      | TensorFlow                                                   |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| PyTorch与基于lua的Torch框架密切相关，该框架在Facebook中广泛使用。 | TensorFlow由Google Brain开发，并在Google上积极使用。         |
| 与其他竞争技术相比，PyTorch相对较新。                        | TensorFlow并不是新的，但许多研究人员和行业专业人士视为一种前沿工具。 |
| PyTorch以强制性和动态的方式包含所有内容。                    | TensorFlow包含静态和动态图形作为组合。                       |
| PyTorch中的计算图是在运行时定义的。                          | TensorFlow不包含任何运行时选项。                             |
| PyTorch包括针对移动和嵌入式框架的部署。                      | TensorFlow更适用于嵌入式框架。                               |

### 3.实现

#### 3.1 数据收集

​		训练集数据由8万张图片及其对应的若干个描述文本构成，基于训练时间和GPU资源考虑，笔者仅选择了40%作为自己的训练集，效果虽不及已有的模型，但也差强人意。

![image-20211213201515901](C:\Users\猫丞\Desktop\image-20211213201515901.png)

> 描述文本（此处仅展示每个图像对应的一条描述）
>
> ```json
> {
>  ...
>  "many plates full of food on the table",
>  "a giraffe eating leaves",
>  ...
> }
> ```

> 感谢来自[MSCOCO](https://cocodataset.org)的数据集支持

#### 3.2 准备过程

​		我们首先需要做的就是将图片转化成预训练模型需要的格式。ImageNet预训练模型需要像素值范围为0~1之间，我们需要根据要求对图像进行处理，另外，我们还要将全部图像调整为256×256大小来保障我们输入的一致性。代码示例如下：

> ```python
> normalize = transforms.Normalize(
>     mean=[0.485, 0.456, 0.406],
>     std=[0.229, 0.224, 0.225])
> ......
> img = imresize(img, (256, 256))
> ```

​		随后，我们要基于训练集图片的标题构建语料库，同时提供了将单词转化为张量的方法，来保障后续接口调用的方便性。另外，我们在训练和测试过程中保障了数据源的一致性，以确保词库中张量和单词映射关系保持不变，避免了不必要的错误。注意，我们需要将特殊的<SOS>和<EOS>加入到语料库中：

> ```python
> words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
> word_map = {k: v + 1 for v, k in enumerate(words)}
> word_map['<SOS>'] = len(word_map) + 1
> word_map['<EOS>'] = len(word_map) + 1
> ```

​		最后，建立一个将最终输出的张量转化为句子的工具方法，可以增强代码的复用性👇

> ```python
> EOS_token = '<EOS>'
> def transform(max_length,output_tensors):
>     sentence = ''
>     for i in range(max_length):
>         top_v, top_i = output_tensors.data.topk(1)
>         if top_i.item() == EOS_token:
>             sentence += EOS_token
>             break
>         else:
>             sentence += index2word[top_i.item()]
>     return sentence
> ```

​		至此，关于文本的输入及输出的处理已经结束，随后可专注于模型的搭建、训练和测试。

#### 3.3 Encoder

​		由于我们的编码器使用的是Pytorch内置的resnet101网络，所以我们的编码器设计得较为简单，只是将预训练模型的后两层去除而已。代码如下：

> ```python
> class Encoder(nn.Module):
>     def __init__(self, encoded_image_size=14):
>         super(Encoder, self).__init__()
>         self.enc_image_size = encoded_image_size
>         # 使用既有的预训练模型
>         resnet = torchvision.models.resnet101(pretrained=True)  
>         # 去除了后面两层，因为我们不需要最后的分类结果
>         modules = list(resnet.children())[:-2]
>         self.resnet = nn.Sequential(*modules)
>         self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size,encoded_image_size))
>         self.fine_tune()
>     def forward(self, images):
>         out = self.resnet(images)  
>         out = self.adaptive_pool(out)  
>         out = out.permute(0, 2, 3, 1)  
>         return out
> ```
>

#### 3.4 带有Attention机制的Decoder

<img src="https://pytorch.org/tutorials/_images/attention-decoder-network.png" alt="img" style="zoom: 67%;" />

```python
class DecoderWithAttention(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        super(DecoderWithAttention, self).__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim) 
        self.embedding = nn.Embedding(vocab_size, embed_dim) 
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  
        self.init_c = nn.Linear(encoder_dim, decoder_dim) 
        self.f_beta = nn.Linear(decoder_dim, encoder_dim) 
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size) 
        self.init_weights() 
    
    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)
    def load_pretrained_embeddings(self, embeddings):
        self.embedding.weight = nn.Parameter(embeddings)
    def fine_tune_embeddings(self, fine_tune=True):
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune
    def init_hidden_state(self, encoder_out):

        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        encoder_out = encoder_out.view(batch_size, -1, encoder_dim) 
        num_pixels = encoder_out.size(1)

        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        embeddings = self.embedding(encoded_captions)
        h, c = self.init_hidden_state(encoder_out) 
        decode_lengths = (caption_lengths - 1).tolist()
       
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t])) 
            preds = self.fc(self.dropout(h)) 
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind

```

#### 3.6 损失函数



### 4.实验结果

