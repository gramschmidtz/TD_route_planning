import numpy as np
import matplotlib.pyplot as plt

# 가우시안 함수 정의
def gaussian(x, a, mu, sigma):
    return a * np.exp(-((x - mu)**2) / (2 * sigma**2))

# x축 범위 설정
x = np.linspace(-10, 10, 1000)

# 3개의 서로 다른 봉우리 생성 (진폭, 평균, 표준편차를 각각 다르게 설정)
peak1 = gaussian(x, a=1.0, mu=-4.0, sigma=1.5)
peak2 = gaussian(x, a=0.8, mu=1.0, sigma=1.0)
peak3 = gaussian(x, a=0.5, mu=6.0, sigma=2.0)

# 3개의 함수를 더해서 다중 봉우리 함수(F(x)) 완성
y_combined = peak1 + peak2 + peak3

# 시각화
plt.plot(x, y_combined, label='Combined Function', color='black', linewidth=2)
plt.plot(x, peak1, '--', label='Peak 1')
plt.plot(x, peak2, '--', label='Peak 2')
plt.plot(x, peak3, '--', label='Peak 3')
plt.legend()
plt.show()