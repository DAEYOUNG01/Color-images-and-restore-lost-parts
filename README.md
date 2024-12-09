# 이미지 색상화 및 손실 부분 복원 (Image Colorization and Inpainting)
 --- 

 

# **CONTENST**

  
* **DACON 대회 배경 및 소개** : 이미지 색상화 및 손실 부분 복원의 개요 

  
* **DACON 이미지 색상화 및 손시 부분 복원 규칙** : 전체적인 규칙 소개 

  
* **모델 선정 이유 및 설명** : 모델 선정 배경 및 설명


## DACON 대회 배경 및 소개 

<p align="center">
  <img src="https://raw.githubusercontent.com/senya-ashukha/senya-ashukha.github.io/master/projects/lama_21/ezgif-4-0db51df695a8.gif" width="1000", height="600" />
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/2b855d05-3e12-4909-b3c2-35fa2c15a49c" width="1000", height="600" />
</p>


# **이미지 색상화(Image Colorization)**


이미지 색상화는 흑백 또는 단색 이미지에 각 픽셀의 절적한 색상을 예측하여 완전한 컬러 이미지로 변환하는 작업


이미지의 구조적 특징과 맥락 정보를 이해하여 자연스러운 색상 결과를 생성하는데 사용

---


## **접근 방식 정리**


**전통적인 방법**
* 규칙 기반 색상 지정 및 수동 편집


**딥러닝 기반 방법**
* CNN(합성곱 신경망): 픽셀 단위의 색상 예측을 위해 공간적 특징 추출
* GAN(생성적 적대 신경망): 생성자와 판별자의 상호작용을 통해 현실감 있는 색상화 이미지 생성
* U-Net: 로컬 및 글로벌 이미지 특징을 결합하는 Encoder-Decoder 아키텍처 활용
* 사전 학습 모델(e.g., CLIP): 이미지-텍스트 임베딩을 활용하여 문맥을 고려한 색상화

---


## **응용 분야**
* 역사 사진 및 영화 복원: 오래된 흑백 사진이나 영화를 컬러화
* 의료 영상: CT, X-ray 등 흑백 데이터를 시각적으로 개선
* 패션 및 마케팅: 제품의 색상 시뮬레이션
* 창작 분야: 흑백 아트워크 또는 게임 디자인에 색상 추가
* 이외 다른 분야에도 사용

---


# **이미지 손실 복원(Image Inpaintion)**


이미지 손실 복원은 손상되거나 누락된 이미지 부분을 원래와 일관성 있게 채우는 작업


물리적으로 손상된 사진 복원, 불필요한 객체 제거, 데이터 전송 중 발생한 결함 복구 등 활용

---


## **접근 방식 정리**


**전통적인 방법**
* 주변 픽셀을 기반으로 하는 보간법(Interpolation)
* 패턴 매칭(PatchMatch Algoritm)을 이용한 결함 영역 복구


**딥러닝 기반 방법**
* CNN 기반 모델: 주변 컨텍스트를 학습해 픽셀 단위 복구
* GAN 기반 모델: PatchGAN 및 Contextua GAN을 활용해 자연스러운 복구 결과 생성
* Ttansformers 및 Attemtion Mechanis: 더 넓은 범위의 문맥 고려
* LaMa 모델: 대규모 손실 영역에서도 고품질 복구를 제공하는 최신 기술

---


## **응용분야**
* 사진 복원: 오래된 사진 및 예술 작품 복원
* 비디오 편집: 불필요한 객체 제거 및 자연스러운 배경 복원
* 의료 영상: 결함 있는 의료 데이터 보완
* 자율 주행: 센서 데이터의 결함 복원
* AR/VR: 가상 환경의 결손 영역 복구로 현실감 향상

---


# **DACON 이미지 색상화 및 손시 부분 복원 규칙** : 전체적인 규칙 소개 


<p align="center">
  <img src="https://github.com/user-attachments/assets/f4127315-6e41-495a-b9e1-e19eebc9aab2" width="500", height="500" />
  <img src="https://github.com/user-attachments/assets/603c460b-0b0b-4231-81bb-5b0feadbd7cf" width="500", height="500" />
</p>


# **모델 선정 이유 및 설명** : 모델 선정 배경 및 설명


## 1. U-Net 모델 선택
* 강점
  1. Encoder-Decoder 구조: U-Net은 입력 이미지를 점점 압축하며 고수준 특징을 학습(Encoding)한 뒤, 이를 복원하며(Denormalization) 세밀한 정보를 결합


     이미지 복원(인페인팅, 색 복원 등)에 최적화된 구조


  2. Skip Connection: Encoder에서 학습한 세부 정보를 Decoder로 직접 연결하여 원본 이미지의 세부 정보를 효과적으로 복구
  3. 다양한 응용 가능성: Segmentation, Inpainting, Super-Resolution 등 이미지 처리 전반에서 성능이 검증된 모델

---

    적용 이유
    
    
    이미지 복원 및 생성: 손상된 이미지 복구(Colorization 및 Loss Restoration)에 적합
    
    
    해상도 보존: 고해상도 이미지에 대해 정보를 손실하지 않고 복구 가능


## 2. PatchGAN Discriminator
* 강점
  1. 지역별 판별(Local Patch Evaluation): PatchGAN은 이미지의 전체적인 진위 여부가 아닌, 이미지를 작은 패치로 나누어 각 패치의 진위 여부를 학습


     고해상도 이미지에서도 세부 정보를 판별하는 데 유리


  2. GAN 구조와의 조합: U-Net과 같은 Generator와 조합 시 실제와 같은 고품질의 복원된 이미지를 생성하는 데 효과적

---
     
    적용 이유
    
    
    Fine-Grained Analysis: 각 패치별로 진짜/가짜 여부를 학습해 세부적인 복원 품질 향상 가능
    
    
    Adversarial Loss 제공: Generator가 더 사실적인 이미지를 생성하도록 유도
   
