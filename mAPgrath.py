import pandas as pd

data = pd.read_csv('/works/ultralytics/runs/detect/train24/results.csv')
# Assuming you've already loaded your data into the DataFrame named 'data'
print(data.columns)
import matplotlib.pyplot as plt

data.columns = data.columns.str.strip()

# Now plot using the correct column names
plt.figure(figsize=(10, 5))
plt.plot(data['epoch'], data['metrics/mAP50-95(B)'], marker='o', linestyle='-', color='b')
plt.title('Epoch vs. mAP50-95(B)')
plt.xlabel('Epoch')
plt.ylabel('mAP50-95(B)')
plt.grid(True)
# 그래프를 파일로 저장
plt.savefig('mAP50-95(B).png')  # 저장 경로와 파일 이름 지정