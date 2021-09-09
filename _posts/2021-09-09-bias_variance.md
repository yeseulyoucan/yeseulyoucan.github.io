---
layout: single
title:  "Bias-Variance Trade off(편향-분산 트레이드오프/ 딜레마)란?"
---

머신러닝을 공부하다보면 아래와 같은 그래프를 자주 보게 된다.  
이 그래프는 모델의 복잡성에 따라 편향(bias)와 분산(variance), 그리고 이를 합친 전체 에러가 어떻게 변화하는지 보여주고 있다.  
그래프를 해석해보자면 모델의 복잡성이 증가할수록 편항의 제곱은 지수적으로 감소하고, 분산은 반대로 증가하며, 전체 에러는 어느 시점까지는 감소하다가 다시 특정 포인트를 기점으로 증가하게 된다.  
우리가 목표로 하는 최적화된 모델은 이 전체 에러가 최소화되는 지점의 모델이다.

![](https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Bias_and_variance_contributing_to_total_error.svg/330px-Bias_and_variance_contributing_to_total_error.svg.png)

머신러닝 모델을 훈련시킬 때는 'training error'와 'test error' 를 측정한다. (여기서는 일단 cross-validation 관련 사항은 제외한다.)  
training error란 훈련데이터를 이용해 모델을 피팅한 후 이 모델을 통해 예측한 예측값과 실제값의 차이를 의미한다.  
반면, test error는 훈련 시 사용되지 않은 데이터(unseen data)를 훈련데이터를 통해 피팅한 모델에 넣어 예측한 예측값과 테스트 데이터의 실제값의 차이를 의미한다.  
이 때 bias는 training error와 관련이 있고, variance는 test error과 연관이 되어있다. 사실 우리가 모델을 피팅하는 근본적인 이유는 아직 관측되지 않은 데이터(unseen data)를 이용해 예측을 하기 위해서다. 여러 unseen dataset일 이용해 예측을 진행했는데, 예측정확도가 높았다면 우리는 이 모델을 '일반화가 잘 된 모델'이라고 말할 수 있다.

물론 training error도 매우 작고, test error 도 매우 작은 모델을 만들면 좋겠지만, 이건 편향-분산 트레이드 오프라는 특성때문에 현실적으로 불가능하다. 왜냐하면 편향이 커지면 분산이 작아지고, 편향이 작아지면 분산이 커지기 때문이다.  
각 용어들의 정의를 하기 전에 우선 MSE(Mean Squared Error)r에 관해 이야기를 해보자. MSE는 연속형 Y 예측모델 성능평가에 사용되는 대표적인 지표들 중 하나이다. 이 MSE의 구성요소를 분해해보면 아래와 같다.

[##_Image|kage@bipWKu/btreqJ6xlOi/3cUbiFCOCVpOOPF8AgwdRK/img.png|alignCenter|data-origin-width="616" data-origin-height="226" data-ke-mobilestyle="widthOrigin"|MSE 분해 (decomposition)||_##]

우리는 훈련데이터가 아닌 unseen data에서도 일반화될 수 있는 최적의 예측모델을 찾는 것이 궁극적인 목표이다. 위 식에서 여기서 x는 training set과 test set을 포괄하는 dataset이다. 훈련데이터로 피팅한 f hat 모델에 x를 넣으면 예측값이 구해지며, y는 해당 데이터셋의 실제값을 의미한다.

MSE는 편향^2과 분산, 그리고 sigma^2으로 구성되어있다. 먼저 편향은 쉽게 말해 예측값의 기대값과 '실제' 모델 간의 오차이다. 여기서 f(x)가 무엇인지 헷갈릴 수 있는데, f(x)는 우리가 결론적으로 알아내고자 하는 x와 y 간 관계를 나타내는 함수이며, f(x)우리는 결국 이 실제 관계와 가장 가까운 예측모델을 찾고자 하는 것이다.

y = F\*(X) + e , e ~ N(0, sigma^2) 와 같은 관계를 가정해보자.

여기서 F\*(X)는 우리가 알고자 하는, 그러나 실제로는 관측할 수 없는 실제 X와 y의 관계를 나타내는 함수이다.

그리고 그 뒤에는 자연적으로 발생하는 노이즈 error가 존재한다.

우리는 X와 y의 실제 관계와 가장 가까운 모델을 찾기 위해 여러 dataset을 바꿔가며 모델링을 할 수 있다. 하지만 각 데이터셋마다 노이즈가 다르기 때문에 이들이 내뱉는 모델 및 예측값들도 당연히 서로 다르다.

이 때, 이 예측값들의 기댓값과 실제값의 차이를 편향이라고 한다. 즉, 편향은 dataset을 변경해가며 모델링을 했을 때 그 예측값들의 기댓값과 실제값과의 차이를 의미한다.

반면, 분산은 dataset을 바꿔가며 모델링을 했을 때 각각의 예측값과 예측값의 기대값과의 차이 제곱의 평균을 의미한다. 즉, 데이터셋이 달라질 때마다 뱉어내는 예측값들과 전체 예측값들의 평균과의 분산을 의미한다.

마지막으로 편향과 분산으로 설명할 수 없는 error는 자연적으로 존재하는 줄일 수 없는 (irreducible)한 오차를 의미한다.

![](https://t1.daumcdn.net/cfile/tistory/99CDCC33599AC28F07)

아마 위 그림도 많이 보셨을 것이다. 가운데 붉은 원은 실제값을 의미하고, 파란색 점들은 예측값들을 의미한다. Lo Variance-Low Bias 부분을 보면 예측값들이 실제값에 가깝고, 예측값들의 분산도 매우 작다. 하지만 두번 째 High Variance-Low Bias의 경우 예측값들이 실제값에 대체로 가깝지만, 예측값들 간의 분산이 크다. 세 번째 High Bias - Low Variance의 경우 예측값들이 실제값과 멀지만 예측값들 간의 분산은 매우 작다. 앞서 말했듯 Low Variance-Low Bias가 가장 이상적인 형태이지만 분산과 편향을 트레이드오프 관계에 있기 때문에 우리는 전체 에러 MSE가 가장 작아지는 지점의 모델을 찾아야 한다. 

[##_Image|kage@bd2s79/btrevH1eScf/pzqnhkxtuUnKgvTfsLX5rK/img.png|alignCenter|data-origin-width="1217" data-origin-height="445" width="751" height="275" data-ke-mobilestyle="widthOrigin"|출처) 유튜브 StatQuest with Josh Starmer||_##]

위 그림에서 파란색 점들은 training set을, 연두색 점들은 test set을 의미한다. 왼쪽 그래프의 경우 훈련데이터에 완전히 피팅된 형태인데, 이 모델을 이용해 test set의 예측을 진행하면 오른쪽 그래프와 같이 실제값과 예측값들 간 차이가 크다(파란색 점으로 예측한 값과 연두색 점으로 예측한 값들 간 분산이 크다). 즉, 이 모델은 bias는 작지만 variance가 크다. 이런 형태의 모델을 overfitting(과적합)된 모델이라고 한다. overfitting된 모델은 훈련데이터에서는 예측을 아주 잘하지만, 테스트데이터에서는 형편없는 예측을 할 가능성이 매우 크다. 

[##_Image|kage@bX3y2l/btreAJcuF5A/1vrOjmu7swLLFkx6lJkUV0/img.png|alignCenter|data-origin-width="1209" data-origin-height="389" width="760" height="244" data-ke-mobilestyle="widthOrigin"|출처)&nbsp;유튜브 StatQuest with Josh Starmer||_##]

반대로 위 그림의 왼쪽 그래프를 보면 linear한 모델이 피팅되었는데, 이는 실제 x와 y의 관계(파란 nonlinear한 선)를 잘 설명하지 못하고 있다. 이 때문에 예측값과 실제값의 차이가 크다, 즉 bias가 크다. 반면, 테스트데이터로 예측을 한 경우에는 예측값과 실제값의 차이가 훈련데이터로 한 경우와 크게 차이가 없다. 즉, variance가 작다. 이런 경우를 underfitting 되었다, 즉 실제 모델의 관계를 충분히 배우지 못했다고 한다. 

---
