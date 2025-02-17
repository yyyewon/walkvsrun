# walkvsrun 데이터

- **date**: 데이터가 기록된 날짜 (YYYY-MM-DD 형식)
- **time**: 데이터가 기록된 시간 (HH:MM:SS:나노초 형식)
- **username**: 데이터를 제공한 사용자 이름 (여기서는 "viktor")
- **wrist**: 손목 장치를 착용한 손(0은 왼손, 1은 오른손)
- **activity**: 활동 유형(0은 걷기, 1은 달리기)
- **acceleration_x**: X축 방향의 가속도 값 (m/s² 단위)
- **acceleration_y**: Y축 방향의 가속도 값 (m/s² 단위)
- **acceleration_z**: Z축 방향의 가속도 값 (m/s² 단위)
- **gyro_x**: X축 방향의 자이로스코프 값 (회전 속도, rad/s 단위)
- **gyro_y**: Y축 방향의 자이로스코프 값 (회전 속도, rad/s 단위)
- **gyro_z**: Z축 방향의 자이로스코프 값 (회전 속도, rad/s 단위)
- **x축**: 왼쪽-오른쪽 방향의 움직임.
- **y축**: 앞뒤 방향의 움직임.
- **z축**: 위아래 방향의 움직임.
- **가속도계 (Accelerometer)**: 물체의 가속도를 측정. 예를 들어, 손목을 빠르게 위로 올리거나 내릴 때 z축 가속도 값이 변함.
- **자이로스코프 (Gyroscope)**: 물체의 각속도(회전 속도)를 측정. 예를 들어, 손목을 돌릴 때 x, y, z축 각속도 값이 달라짐.

# walkvsrun_sorted.csv

full_datetime 컬럼 추가: 시간 정렬 및 가공 (시간 순서대로 정리)

# encoder_visualization.ipynb
## Transformer 구조 설명
**1. 입력차원변환**

`self.embedding = nn.Linear(input_dim, d_model)`


**2. Transformer Encoder Layer 생성**

`self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)`

`self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)`


**3. 순전파 과정**

`x = self.embedding(x)  # [batch_size, seq_length, d_model]`

`x = x.permute(1, 0, 2)  # Transformer expects [seq_length, batch_size, d_model]`

`x = self.transformer(x)  # [seq_length, batch_size, d_model]`

`x = x[-1]  # 마지막 시점 출력 사용`

`return self.fc(x)`


**4. 데이터 읽기 및 전처리**


**5. 학습 데이터 준비**

`seq_length = 130  # 130개의 연속된 샘플을 하나의 입력으로 사용`

`input_dim = len(features)`

`num_classes = data[target].nunique()`


`train_dataset = TimeSeriesDataset(train_data, features, target, seq_length)`

`test_dataset = TimeSeriesDataset(test_data, features, target, seq_length)`



**6. 모델 학습**

AdamW 옵티마이저 사용, 배치마다 역전파 진행


**7. 학습 결과 시각화**


# walk_mlp.ipynb
## mlp 구조 설명
