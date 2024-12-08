# 이미지 색상화 및 손실 부분 복원 (Image Colorization and Inpainting)
 --- 

 

# **CONTENST**

  
* **DACON 대회 배경 및 소개** : 이미지 색상화 및 손실 부분 복원의 개요 

  
* **DACON 이미지 색상화 및 손시 부분 복원 규칙** : 전체적인 규칙 소개 

  
* **모델 선정 이유 및 설명** : 모델 선정 배경 및 각종 함수(파라미터) 설명


## DACON 대회 배경 및 소개 

<p align="center">
  <img src="https://raw.githubusercontent.com/senya-ashukha/senya-ashukha.github.io/master/projects/lama_21/ezgif-4-0db51df695a8.gif" />
</p>

<p align="center">
![image](https://github.com/user-attachments/assets/2b855d05-3e12-4909-b3c2-35fa2c15a49c)
</p>

**이미지 색상화(Image Colorization)**
이미지 색상화는 흑백 또는 단색 이미지에 각 픽셀의 절적한 색상을 예측하여 완전한 컬러 이미지로 변환하는 작업
이미지의 구조적 특징과 맥락 정보를 이해하여 자연스러운 색상 결과를 생성하는데 사용


## **접근 방식 정리**


**전통적인 방법**
* 규칙 기반 색상 지정 및 수동 편집


**딥러닝 기반 방법**
* CNN(합성곱 신경망): 픽셀 단위의 색상 예측을 위해 공간적 특징 추출
* GAN(생성적 적대 신경망): 생성자와 판별자의 상호작용을 통해 현실감 있는 색상화 이미지 생성
* U-Net: 로컬 및 글로벌 이미지 특징을 결합하는 Encoder-Decoder 아키텍처 활용
* 사전 학습 모델(e.g., CLIP): 이미지-텍스트 임베딩을 활용하여 문맥을 고려한 색상화


## **응용 분야**
* 역사 사진 및 영화 복원: 오래된 흑백 사진이나 영화를 컬러화
* 의료 영상: CT, X-ray 등 흑백 데이터를 시각적으로 개선
* 패션 및 마케팅: 제품의 색상 시뮬레이션
* 창작 분야: 흑백 아트워크 또는 게임 디자인에 색상 추가
* 이외 다른 분야에도 사용





