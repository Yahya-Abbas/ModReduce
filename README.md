# ModReduce
## Covered in this repo
1. An implementation for ModReduce, a novel knowledge distillation technique that combines three offline knowledge distillation techiques: 
   * Response knowledge distillation by [Hinton et al.](https://arxiv.org/abs/1503.02531)
   * Relational knowledge distillation, specifically Contrastive Representation Distillation by [Yonglong Tian et al.](https://arxiv.org/abs/1910.10699) 
   * Feature knoweldge distillation, specifically Cross-Layer Distillation with Semantic Calibration by [Defang Chen et al.](https://arxiv.org/abs/2012.03236v1)
through online learning knowledge distillation using a Weighted Averaging method. 

![Alt text](images/Architecture.png?raw=true "ModReduce Architecture")

2. An impementaion for 4 different aggregation methods to be used inside ModReduce for the online learning step (PCL-ONE-FC-WAvg).

3. Alongside the three aforementioned offline knoweldge distillation methods, the code integrates the classical relational knowledge distillaion RKD by [Wonpyo Park et al.](https://arxiv.org/abs/1904.05068) and FitNets: Hints for Thin Deep Nets by [Adriana Romero et al.](https://arxiv.org/abs/1412.6550)

4. Benchmarking Hinton, SemCKD, CRD, and ModReduce on CIFAR-100 with 15 experiments combining Teacher-Student architectures from both [CRD](https://arxiv.org/abs/1910.10699) and [SemCKD](https://arxiv.org/abs/2012.03236v1) papers.

## Installation
1. The conda enviroment needed to run our code could be installed using pyt.txt
2. <pre>pip install torch-summary</pre>
3. <pre>pip install matplotlib</pre>

## CIFAR-100 results
The student model with highest accuracey in each teacher-student combination is shown in **bold**.
|                          | Exp: 1                      | Exp: 2                        | Exp: 3   | Exp: 4    | Exp: 5    | Exp: 6      | Exp: 7       | Exp: 8       |
| ------------------------ | --------------------------- | ----------------------------- | -------- | --------- | --------- | ----------- | ------------ | ------------ |
| Experiments Source       | CRD (Similar Atchitectures)                                                                    ||||| CRD (Different Architectures)         |
| Teacher Model            | wrn\_40\_2      | wrn\_40\_2    | resnet56 | resnet110 | resnet110 | vgg13       | resnet32x4   | wrn\_40\_2   |
| Teacher Accuracy   | 75.61           | 75.61         | 72.34    | 74.31     | 74.31     | 74.64       | 79.42        | 75.61        |
| Student Model            | wrn\_16\_2      | wrn\_40\_1    | resnet20 | resnet20  | resnet32  | MobileNetV2 | ShuffleNetV1 | ShuffleNetV1 |
| Student Accuracy   | 73.26           | 71.98         | 69.06    | 69.06     | 71.14     | 64.6        | 70.5         | 70.5         |
| Hinton Accuracy    | 75.39           | 74.21         | 71.7     | 70.99     | 73.66     | 68.72       | 74.59        | 75.45        |
| SemCKD Accuracy    | 75.1            | 73.11         | 70.91    | 70.95     | 73.47     | 68.66       | **77.21**        | 76.93        |
| CRD Accuracy       | **76.12**           | **74.91**         | 71.72    | 71.35     | 73.65     | **69.66**       | 75.77        | 76.59        |
| ModReduce (WAvg) Accuracy | 75.44           | 74.84         | **71.99**    | **72.01**     | **74.34**     | 69.23       | 76.96        | **77.14**        |

|                          | Exp: 9       | Exp: 10 | Exp: 11      | Exp: 12    | Exp: 13    | Exp: 14      | Exp: 15     |
| ------------------------ | ------------ | ------- | ------------ | ---------- | ---------- | ------------ | ----------- |
| Experiments Source       | SemCKD & CRD                          ||| SemCKD  |
| Teacher Model            | resnet32x4   | vgg13   | resnet32x4   | resnet32x4 | resnet32x4 | vgg13        | wrn\_40\_2  |
| Teacher Accuracy   | 79.42        | 74.64   | 79.42        | 79.42      | 79.42      | 74.64        | 75.61       |
| Student Model            | resnet8x4    | vgg8    | ShuffleNetV2 | vgg8       | vgg13      | ShuffleNetV2 | MobileNetV2 |
| Student Accuracy   | 73.09        | 70.46   | 72.6         | 70.46      | 74.82      | 72.6         | 65.43       |
| Hinton Accuracy    | 74.32        | 73.62   | 75.73        | 72.48      | 77.21      | 75.89        | 69.02       |
| SemCKD Accuracy    | 75.55        | 74.08   | 78.07        | 75.02      | 79.14      | 76.24        | 69.77       |
| CRD Accuracy       | 74.97        | 74.39   | 76.57        | 73.68      | 77.71      | 76.26        | **70.13**       |
| ModReduce (WAvg) Accuracy | **75.78**        | **74.64**   | **78.23**        | **75.21**      | **79.51**      | **76.76**        | 69.37       |

## To fetch pretrained teacher models run
<pre>sh scripts/fetch_pretrained_teachers.sh</pre>

## To run the code 
consult the files under <pre>scripts/"online_learning_method"/</pre> 

## Code Acknoweldgments
This code is based upon the source codes of CRD https://github.com/HobbitLong/RepDistiller and SemCKD https://github.com/DefangChen/SemCKD
