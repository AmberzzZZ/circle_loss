1. formulation

    1.0 similarity
        cosine distance in this paper
        github上的野生实现中：
            1.0.1 l2_norm the feature-embeddings before circle loss
            1.0.2 dot similarity: 区别就是除不除分母的abs [https://github.com/xiangli13/circle-loss/blob/master/circle_loss.py]


    1.1 scale factor $\lambda$
        控制similarity score的最大值，定值，论文里面32-512，default是256
        实验中发现scale要从小往大调，mnist上32就好，大了会nan


    1.2 decay factor $\alpha$
        控制梯度，与到optimum的距离成正比，Op是1+m，On是-m
        clip0: decay factor取正数部分


    1.3 margin factor $\Delta$
        控制decision boundary的宽窄，Dp是1-m，Dn是m，m取值default是0.25


    1.4 decouple
        sn和sp不同时计算，因此可以解耦成两个部分



2. visualize
    sklearn的tsne，观察不出啥


3. inference
    circle_loss model的输出是feature vector
    要与pre stored的标准模版做cosine计算，找到相似度最高的样本作为归属
    标准模版怎么准备？质心还是啥？