# Introduction

이 수업에서는 Deep Learning(DL) 과 Computer Vision(CV) 을 중심으로 다룬다.  
단순히 알고리즘을 나열하는 것이 아니라, _왜 이런 접근이 필요해졌는지_,  
그리고 _어떤 수학적·개념적 배경 위에서 발전해왔는지_를 함께 이해하는 것이 목표이다.

이때 가장 중요한 수학적 도구는 Machine Learning(ML) 이며, 보다 정확히 말하면 Statistical Machine Learning 이다.

---

# History of Vision

Vision의 출발점은 매우 오래되었다.  
약 **580만 년 전**, 빛을 sensing하는 기관을 가진 생물이 등장하면서  
“보는 능력”은 생존과 직결되는 핵심 기능이 되었다.

#### Hubel and Wiesel (1959)

Hubel과 Wiesel은 시각 피질을 연구하며 중요한 사실을 발견했다.

-   모든 뉴런이 전체 시야를 담당하는 것이 아니라  
    **특정 위치, 특정 패턴에만 반응하는 receptive field**가 존재한다.
-   시각 정보 처리가 **계층적** 구조를 가진다.

이 발견은 이후 **CNN의 inductive bias**로 이어진다.  
즉, 국소적 패턴 → 점점 추상적인 표현으로 올라가는 구조는  
생물학적 시각 시스템에서 영감을 받은 것이다.

#### Larry Roberts (1963) ~ David Marr (1970s)

이 시기에는 “Vision이란 무엇인가?” 에 대한 이론적 접근이 이루어졌다.

-   이미지(image)와 특징(feature)을 구분
-   형태, 원근감, 3D 구조에 대한 표현
-   2D image로부터 3D world를 이해하려는 시도

핵심 문제는 다음과 같다.

> 2D 정보만을 이용하여 3D 정보를 recover할 수 있는가?

이 문제는 본질적으로 매우 어렵다.  
인간은 양안을 사용하지만, 그마저도 극도로 정확한 3D 복원을 하지는 못한다.

특히 intrinsic calibration 없이 3D 형상을 복원하는 문제는 전형적인 **ill-posed problem**이다.

> Ill-posed problem  
> 해가 존재하지 않거나, 유일하지 않거나,  
> 입력의 작은 변화에 해가 크게 변하는 문제.  
> 수학적으로는 regularization이나 constraint 없이는  
> solution space가 무한하거나 아예 정의되지 않는다.

이 지점이 Vision이 Language보다 어려운 근본적인 이유 중 하나이다.  
Language는 비교적 명시적인 구조를 모델링할 수 있는 반면,  
Vision은 물리 세계의 투영 문제를 다루기 때문에  
모델링 자체가 훨씬 복잡하다.

#### Canny (1980s)

1980년대에는 고전적인 Computer Vision 알고리즘들이 등장했다.  
대표적인 예가 Canny Edge Detector이다.

핵심 아이디어는 단순하다.

-   이미지에서 gradient가 급격히 변하는 지점 = edge

Pipeline:

1.  Gaussian kernel smoothing (noise 제거)
2.  Gradient 계산
3.  Non-Maximum Suppression (NMS)
4.  Thresholding
5.  Edge tracking

오늘날 기준으로는 단순해 보이지만,  
당시에는 매우 체계적이고 강력한 접근이었다.

#### Normalized Cuts (1997) ~ Caltech101, PASCAL (2004/2007)

1990년대 후반~2000년대 초반에는

-   이미지 segmentation
-   point matching
-   face recognition

과 같은 다양한 Vision 기술들이 발전했다.

특히 Caltech101, PASCAL VOC 같은 데이터셋의 등장은 인터넷 보급과 함께 Vision 연구의 방향을 크게 바꾸었다.  
→ Data-driven perspective

---

# History of Deep Learning

#### Perceptron (1958)

Perceptron은 인간의 신경세포를 단순화한 모델로, 신경망 기반 컴퓨팅의 출발점이다.

하지만 1969년, 단층 퍼셉트론은 XOR 문제를 해결할 수 없음이 증명되었다.  
이는 곧 **Multi-layer Perceptron** 연구의 계기가 된다.

#### Neocognitron (1980)

Neocognitron은 **CNN의 전신**이라 할 수 있다.

-   국소 수용영역
-   계층적 구조
-   위치 변화에 강인한 표현

현대 CNN에서도 이 아이디어는 핵심적인 구조이다.  
Vision 분야에서 CNN이 오랫동안 지배적인 이유이기도 하다.

#### Backpropagation (1985)

Backpropagation의 등장으로 multilayer perceptron을 실제로 학습시키는 방법이 확립되었다.

이 시점부터 “깊은 모델을 쓸 수 있다”는 가능성이 열렸다.

#### LeNet (1998)

LeNet은

-   Neocognitron 구조
-   Backpropagation

을 결합한 초기 CNN의 완성형이다. 실제 문제(손글씨 숫자 인식 - MNIST)에 성공적으로 적용되었다.

#### Deep Learning (2000s)

2000년대에는 신경망을 더 깊게 쌓으려는 시도가 있었지만,

-   데이터 부족
-   overfitting

문제가 심각했다.  
당시에는 아키텍처에만 집중하고 데이터의 중요성을 간과하는 경향도 있었다.

이 문제를 바꾼 것이 **ImageNet**의 등장이다.  
100만 장 이상의 대규모 데이터셋과 ImageNet Challenge는 DL의 전환점이 되었다.

현재 ImageNet classification 자체는 사실상 정복 단계에 접어들었다.

---

# Nowadays

오늘날 Deep Learning은 거의 모든 CV task에 적용된다.

-   Object Detection
-   Semantic Segmentation
-   Video Classification
-   Pose Recognition
-   Medical Imaging
-   Image Generation

이러한 발전의 핵심 배경 중 하나는 압도적인 computing power(GPU, TPU) 이다.

---

# What We Will Cover

앞으로 이 수업에서는 다음 내용을 다룬다.

1.  **DL Basics**
2.  **Perceiving and Understanding the Visual World**
3.  **Generative and Interactive Visual Intelligence**
4.  **Human-Centered Applications and Implications**

→ 단순한 모델 구현을 넘어, 시각 지능을 어떻게 이해하고, 어떻게 사회에 적용할 것인가까지 다루는 것이 목표이다.