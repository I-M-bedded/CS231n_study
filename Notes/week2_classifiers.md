# CS231n\_Week2

### 문제 정의 : Semantic Gap

우리가 고양이라는 객체를 사진으로 찍는다고 하여보자.  
이때 카메라의 위치에 따라서 사진의 각 픽셀들의 정보는 완전히 변할 것이다.  
사람은 카메라의 위치가 변하더라도 고양이 사진이 고양이를 찍은 것임을 안다.

하지만 픽셀들의 정보는 완전히 변하기 때문에, 컴퓨터의 입장에서는 고양이라고 판단할 근거가 부족하다. 즉, 객체가 지닌 정보와 실제로 이미지의 픽셀들이 지닌 정보의 의미적 격차(Semantic Gap)이 존재하는 것이 주된 문제점인 것이다.

이러한 semantic gap은 여러 요인으로 발생할 수 있을 것이다.

위에서 설명한 카메라의 시점이나, 조명, 배경, 일부분이 가려지는 Occlusion, 객체의 형태가 달라지는 defomation등.

또한, 같은 고양이라도 여러 고양이가 있는 문제, (실제로는 픽셀 정보가 꽤 다르더라도 같은 고양이), 고양이처럼 보이지만 사실은 고양이 귀를 한 다른 동물이라는 그런 맥락을 읽는 문제 등.

즉 사람이 이미지에서 인식할 수 있는 정보는 매우 다양하기 때문에, 이러한 다양하고 방대한 정보를 컴퓨터도 인식할 수 있도록 하는 것이 Computer vision에서의 주된 challenge일 것이다.

## Image Classification

고전적인 cv에서는 edge detection하고 주요 pattern을 인식하고 이러한 pattern이 몇 개면 고양이, 이러한 pattern이 몇 개면 강아지. 즉 휴리스틱이나 잘 설계된 패턴 인식 모델에 기반했다.

CV에 혁신적인 machine learning paradigm은 data-driven perspective로 휴리스틱을 배제하고, 모델 기반으로 task를 수행하는 것이 아니라, 데이터 기반으로 컴퓨터가 스스로 이해하고 패턴을 찾아내서 필터를 설계하도록 ‘최적화’ 한다. 이는 다음과 같이 작동하는데

1.  dataset을 모으고
2.  필터를 학습시키고
3.  이 필터를 평가한다.

우리가 해야 하는 과제는 dataset을 잘 모으는 방법, 그리고 필터 → 신경망을 잘 학습시키는 방법을 찾는 것이다.

### kNN classifier

Nearest Neighbor 이라는 말에서 알 수 있듯, label이 있는 training data들이 label별로 모여 있고, 어떠한 data가 어떤 label집단에 **가까운지** 판별해서 가까운 label로 classify한다고 생각할 수 있다.

그럴려면 distance를 정의해야 한다. 새로운 data가 기존의 traning data point들과 얼마나 가까운지 알아야하기 때문이다.

가장 일반적인 distance function중 하나는 L1 norm이다.

$\\underset{i,j}\\sum |X\_{i,j}-Y\_{i,j}|$

위의 수식에서 X가 data point, Y가 query라고 했을 때, 각 픽셀마다 difference를 구하고 합쳐준다.

이제 Nearest Neighbor classifier를 작성해보자.

```
import numpy as np

class NN_classifier:
    def __init__(self):
        pass

    def train(self, img, label): #img : N*D(가로 세로 pixel을 flatten 했다고 생각) label : N
        self.Xtr = img 
        self.Ytr = label #단순하게 저장

    def predict(self, img):
        num_pred = img.shape[0]
        pred=np.zeros(num_pred, dtype=self.Ytr.dtype)
        for i in range(num_y):
            dist = np.sum(np.abs(self.Xtr - y[i,:]),axis=1) #L1 norm 계산
            min_index = np.argmin(dist) #argmin으로 거리가 최소인 index찾기
            pred[i]=self.ytr[min_index] #거리가 최소인 index의 label이 예측 label
        return pred
```

앞에서 설명했던 3단계에 입각하자면, data를 준비한 후에  
train → 여기서는 단순히 data를 저장한다.  
prediction → 모든 train data에 대해 L1 norm을 계산하고, L1 norm중 가장 작은 값의 index를 찾아, 해당하는 index의 label을 반환한다.

각 단계의 알고리즘 시간복잡도를 보자면, train은 데이터를 저장한다고 되어 있지만 사실은 복사가 아니라 단순히 pointer를 넘겨주는 방식이므로 (shallow copy) O(1)이다.  
이에 비해서 predict의 경우 L1 norm 계산은 N에 비례하므로 O(N),  
argmin은 배열 요소 N개를 한 번씩 들려서 비교하므로 O(N),  
이후 예측으로 값을 복사해주는 건 O(1)의 연산을 prediction sample마다 하게 된다.

> 실제로는 거리계산의 시간 복잡도는 O(ND)일 것이다. D개의 요소를 하나로 합쳐야 하기 때문이다. 하지만 D는 고정된 값이므로 여기서는 상수로 취급했다. 실제로는 D에 비례하는 연산 부하가 엄연히 존재하므로, image 처리에서는 차원을 많이 늘리지 않는다. (일반적으로 HD화질도 안 씀)

이러한 구조는 ‘나쁘다’

학습에 긴 시간이 걸리는 것은 괜찮을 수도 있지만,  
추론에 긴 시간이 걸린다는 것은, dataset이 클 수록 실제 결과를 얻는 사용자 입장에서 많이 기다려야 한다는 것이다.

이러한 문제는 이후에 notation될 linear classifier와 같은 방법으로 해결해 볼 것이다.

위의 알고리즘을 그대로 적용하면 다음과 같이 나올 것이다.

[##_Image|kage@cIGP6W/dJMcad1OvTG/AAAAAAAAAAAAAAAAAAAAANjmrYSokNRnQzte6-WC-Z3GtF6DZz62ACP0LG4ocM9e/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1769871599&amp;allow_ip=&amp;allow_referer=&amp;signature=i59QUtaupBS6gaILazG3O6kj%2Bi0%3D|CDM|1.3|{"originWidth":645,"originHeight":510,"style":"alignCenter","width":364,"height":288}_##]

이미지를 보면, 뭔가 이상한 point가 있다. 초록영역에 떡하니 하나 있는 노란색 point이다. 이는 outlier일 것이다. 실제 문제 해결과정에서 이런 outlier가 다수 존재 할 것이다.

이러한 outlier에 강건하도록 영역을 분할 하려면 어떻게 해야할까? 단순하다. 가까운점을 여러 개(k) 뽑아보고 이들이 어디에 속해 있는지 보고, 다수결로 정해주자.

이를 k-Nearest Neighbors 알고리즘이라고 한다.

[##_Image|kage@PTCd6/dJMcadAJ6ho/AAAAAAAAAAAAAAAAAAAAAJKUTSkogInGNliEAOhIlT7CTXz0g4TVb9ziS7vLmosK/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1769871599&amp;allow_ip=&amp;allow_referer=&amp;signature=kR%2FVdNRotpqW6NtAeFOgOnNoYXE%3D|CDM|1.3|{"originWidth":1180,"originHeight":329,"style":"alignCenter"}_##]

k에 따라서 이상치가 해결되고, 더욱 smooth한 경계선을 가지는 것을 확인할 수 있다. 하지만 문제점이 보인다. 바로 결정되지 않는 영역이 생기는 것이다.

이러한 문제는 거리 함수를 L2 norm - Euclidean distance로 변경하면 어느 정도 해결 할 수 있다.

$$  
\\underset{i,j}\\sum \\sqrt{(X\_{i,j}-Y\_{i,j})^2}  
$$

manhattan의 경우, 같은 거리로 표시되는 등거선이 선의 형태이다.  
이에 비해 euclidean의 경우 등거선이 원의 형태이므로, 서로 다른 두 점에서 뻗어 나왔을 때 manhattan은 선으로 만날 확률 이 높지만, euclidean은 점으로 만난다.

이러한 차이 때문에 L2 norm을 사용하면 경계선이 깔끔한 선으로 떨어지면서 결정되지 않는 영역이 줄어들게 된다.

또 하나의 challenge는, 좋은 k를 찾는 것이다.

### hyperparameter & Cross Validation

kNN에서의 k와 같이, 머신러닝에서 그 결과에 크게 영향을 미치는 사전 parameter를 hyperparameter라고 한다. hyperparameter 튜닝이라고 하는 것은, 이러한 hyperparameter중 가장 모델의 일반적인 성능을 좋게 만드는 hyperparameter를 찾는 과정이다.

가장 좋은 hyperparameter를 어떻게 찾을 수 있을까??

몇 가지 idea를 보자.

Idea 1 : train set accuaracy가 가장 높은 k → ❌ if k=1, accuracy is 100%  
Idea 2 : test set accuaracy가 가장 높은 k → ❌ cheating, test set is fully new data  
Idea 3 : validation set accuaracy가 가장 높은 k → ✅

따라서 우리는 val set을 이용해 k를 찾을 것인데.. 문제가 있다. val set이 고정되어 있다면? 우리가 가진 train set에는 오히려 잘 맞지 않아서 정확도가 그리 높지 않게 나올 수도 있다. 가장 일반적으로 좋은 k를 찾아야 한다.

그래서 사용하는 idea가 k-fold Cross Validation이다.

dataset을 k개의 fold로 나누고 번갈아가면서 validation set역할을 하게 한다. 이러면 k를 결정하는 데에 가지고 있는 모든 데이터셋이 참여하게 되면서 더욱 일반적인 성능의 k를 기대할 수 있을 것이다.

[##_Image|kage@Elk3d/dJMcagddyl0/AAAAAAAAAAAAAAAAAAAAAD94nRWHxinNfiOxLR4v3trPCd7-cBYsEZgC6TqtraaV/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1769871599&amp;allow_ip=&amp;allow_referer=&amp;signature=OO9GaTYvhTXtUNdGr5Ep%2FLDHhJA%3D|CDM|1.3|{"originWidth":1110,"originHeight":317,"style":"alignCenter"}_##]

_하지만 deep learning에서는 자주 쓰이지는 않는다_.

### kNN classifier의 한계

하지만 img에서 pixel간 거리로 classify하는 kNN classifier는 거의 사용하지 않는다. 왜냐하면 pixel간 거리는 informative하지 않기 때문이다.

거기다가 위에서 설명했듯, 추론이 오래 걸리는 단점도 존재한다.

## Linear Classifier

선형이라는 말에서 알 수 있듯, 데이터를 잘 구분하는 선을 찾는 것과 같다.  
이 linear classifier는 deep neural network의 근간을 이룬다.  
사실 신경망이란, 이러한 선형 레이어사이에 비선형 활성화 함수를 끼워 넣은 것이다.

데이터를 잘 구분하는 선이란 건, 여러 개의 선형 연립 방정식을 의미한다.

선형 연립방정식은 선형대수의 언어를 빌리자면 다음과 같을 것이다.

$$  
f(\\mathbf{x},W)=W\\mathbf{x}+b  
$$

여기서 $W$가 바로 파라미터(weight)이다. data vector(혹은 tensor) x에 대해서 parameter행렬 W를 적절하게 찾아 데이터를 잘 구분하는 선을 찾는 것이다.

이 W를 어떻게 적절하게 찾을까?

이를 위해서는 크게 두 가지가 필요하다.

1.  Loss function
2.  optimization

Loss function은 이 parameter 행렬이 얼마나 좋은지 혹은 나쁜지 알려주는 평가지표다.

optimization은 이 parameter 행렬을 loss를 줄이는 방향으로 학습하는 알고리즘을 의미한다.

이번에는 Loss function에 집중을 해보자.

이 parameter matrix가 좋은지 안 좋은지 판별을 하려면, 이 parameter matrix가 trainset을 잘 구별하는지 알아야 한다.

지금은 image를 classify 하는 것이 목적이다. 즉, 최종적으로 결과를 해당 이미지일 확률로 냈으면 좋겠다. 이 떄 정답은 1 아니면 0으로 나타낼 수 있을 것이다. (one-hot encoding)

  
이미지 데이터 벡터가 parameter와 곱해졌을 때 계산되는 확률이 정답과 얼마만큼 차이나는지를 loss로 설정한다면, 분명히 이는 parameter행렬을 평가해줄 수 있다.

하지만 문제는 $Wx+b$의 결과 값은 확률이 아니다. 확률은 non negative여야 하고, 최대값이 1이어야 한다. 우리의 정답도 0 아니면 1이기 때문에 이는 지켜져야 한다.

이를 위해서 logit에 Softmax함수를 사용한다.

$$  
s=f(x\_i;W) \\quad Softmax= P(Y=k|X=x\_i)=\\frac{e^{s\_k}}{\\sum\_j e^{s\_j}}  
$$

non negative는 exponential을 통해, 확률로 mapping은 정규화를 통해 해결하게 된다. 여기서 우리는 데이터에 대해서 정답 레이블 $y\_i$가 최대화 되는 파라미터를 찾아야 한다.

$$  
\\text{maximize}\\space P(Y=y\_i|X=x\_i)  
$$

이는 수학적으로, weight가 input에 대한 likelihood를 최대화하는 문제인 MLE를 적용해서 풀 수 있다.

우리의 likelihood는 softmax함수이고, MLE는 이 likelihood를 negative log를 적용하여 푼다. 따라서 softmax loss는 다음과 같이 정의 될 수 있다.

$$  
L=-\\log{\\frac{e^{s\_{y\_i}}}{\\sum\_j e^{s\_j}}}  
$$

이는 정보이론적 관점에서 cross-entropy와 식이 동일하다.  
따라서 cross-entropy loss라고 불리기도 한다.

> **cross-entropy 복습**  
> 목표 분포에 대해서 제안 분포가 얼마나 목표 분포에 대한 정보를 가지고 있는지 나타내는 것이 cross-entropy이다.  
> $H(p, q) = - \\sum\_{k} p(k) \\log q(k)$  
> 이때 one-hot encoding에 의해서 목표 분포 p의 경우 정답은 1, 외에는 모두 0이다.  
> 따라서 cross entropy에서는 제안 분포의 정답 확률만이 남게 된다.  
> $H(p, q) = - \\log q(y\_i)$  
> 이는 위의 softmax loss와 정확히 같은 식이다.

이 유사성은 KL Divergence로 설명된다. 두 분포 간 거리를 의미하는 KL을 최소화 하는 것과 수학적으로 동일한데,

$$  
H(p, q) = - \\sum\_{k} p(k) \\log q(k)= D\_{KL}(P||Q)+H(Q)  
$$

위의 식에서 $H(Q)$가 one hot encoding에서는 0이므로, 사실 classification에서는 cross-entropy = KL-divergence이다.

MLE와 linear regression의 관계등은 ML에서 더욱 자세히 다뤄보자.

위와 같은 softmax loss말고도 한 가지 더, SVM loss가 있다.

우리가 본래 ML등에서 다루는 SVM같은 경우에는 Dual form을 풀기 위해서 KKT조건을 만족하는~ kernel function을 통과하고~ 하는 복잡한 것이 아니라,

soft margin을 수학적인 식으로 표현하는 Hinge loss만 가져다 쓰고 최적화는 parameter를 SGD와 같은 알고리즘으로 최적해를 구하도록 하는 것이 목적이다.

Hinge Loss를 수식으로 나타내면,

$$  
L\_i = \\sum\_{j \\neq y\_i} \\max(0, s\_j - s\_{y\_i} + \\Delta)  
$$

위와 같다. 여기서 $\\Delta$가 soft margin의 slack variable과 같은 역할이라고 생각하면 된다.

### Discussion

Q4 : (SVM) What if the sum was over all classes? (including j = y\_i)

loss는 상수 delta만큼 전체적으로 커질 것인데, 이는 변수가 아니므로 최적화 과정에서 미분을 통해 전파되진 않으므로, 학습에는 영향이 없다.

Q5: What if we used mean instead of sum?

전체적인 loss크기가 작아지므로 학습이 더뎌지는 효과가 있다. learning rate을 적절히 조절하거나 꼭 scaling을 해줘야 할 필요가 있다.

Q6: What if we used L2-hinge loss

gradient를 생각해보면, Delta내부로 들어온 애들은 조금 덜 당기지만,  
Delta에서 벗어난 애들은 2배로 당길 것이다. 즉 Outlier에게 가혹하고, Inlier에게 관대하다.