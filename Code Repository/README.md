Code Repository

📗Machine Learning
วัตถุประสงค์ของ Code

เพื่อเปลี่ยน "ข้อมูล" ให้เป็น "คำทำนาย" (Prediction) การนำข้อมูลในอดีตมาให้คอมพิวเตอร์ประมวลผล เพื่อหา "แนวโน้ม" ในอนาคต
เพื่อหา "รูปแบบ" ที่มนุษย์มองไม่เห็น (Pattern Recognition) ใช้ Scikit-learn ในการวิเคราะห์ข้อมูลจำนวนมหาศาล เพื่อง่ายต่อการทำงาน
เพื่อ "ลดขั้นตอน" การเขียนโปรแกรม (Automation) Scikit-learn ช่วยลดขั้นตอนการป้อนคำสั่งในการเขียนโค้ด เช่น if-else
หลักการทำงานของ code

Machine Learning คือ การสั่งค่าคอมพิวเตอร์ให้เรียนรู้รูปแบบจากข้อมูลเพื่อทำนายผลเองได้

Scikit-learn คือ เครื่องมือใน Python ที่รวมอัลกอริทึมสำเร็จรูปไว้ให้เรียกใช้ได้ทันที

สรุปด้วยขั้นตอนมาตรฐาน (Scikit-learn Workflow)

Prepare: เตรียมข้อมูล โดยการ import model

Fit: ป้อนคำสั่งให้โมเดล เพื่อให้ทำการเรียนรู้ข้อมูลเหล่านั้น
Predict: หลังการเรียนรู้ผล แล้วจึงสามารถทำนายผลได้

วิธีการใช้งานโค้ด

นำเข้าข้อมูล (Data) เปรียบเทียบ: คุณมี กองผลไม้ อยู่ตรงหน้า (ข้อมูล) และคุณรู้ว่าลูกไหน หวาน ลูกไหน เปรี้ยว (เฉลย)
X (Features): คือลักษณะ (สี, ผิวขรุขระ, กลิ่น)

y (Target): คือคำตอบ (หวาน หรือ เปรี้ยว)
ป้อนคำสั่ง(Fit) เปรียบเทียบ: คุณหยิบผลไม้ส่งให้ "ลิ้นสมองกล" (Model) ลองชิมทีละลูก เพื่อให้มันจำว่า "อ๋อ... ผิวขรุขระแบบนี้คือเปรี้ยวนะ" ใน Scikit-learn เราใช้คำสั่งสั้นๆ แค่ model.fit(X, y) สมองเสี้ยวนี้ทำงานแค่: "จำรูปแบบ" (Pattern)

ทำนายผล (Predict) เปรียบเทียบ: คราวนี้หยิบผลไม้ลูกใหม่ที่ "ไม่เคยเห็น" มาวาง แล้วถามลิ้นนั้นว่า "ลูกนี้รสอะไร?" ถ้าลิ้นจำเก่ง (Model ดี) มันจะทายถูกว่า "ลูกนี้หวานแน่ๆ!" ใน Scikit-learn คือคำสั่ง model.predict(ข้อมูลใหม่)
การวิเคราะห์ปัจจัยเชิงพื้นที่และโมเดลการเรียนรู้ของเครื่องเพื่อทำนายผลสัมฤทธิ์ทางการศึกษาของนักเรียน

(Spatial Analysis and Machine Learning Models for Predicting Student Educational Outcomes)

1. การนำเข้าไลบรารี (Import Libraries)
ส่วนนี้เป็นการนำเข้าไลบรารีที่จำเป็นสำหรับการทำงานในโค้ดนี้:

numpy as np: ใช้สำหรับการคำนวณทางคณิตศาสตร์ โดยเฉพาะการทำงานกับอาร์เรย์และเมทริกซ์
pandas as pd: ใช้สำหรับจัดการและวิเคราะห์ข้อมูล โครงสร้างข้อมูลหลักคือ DataFrame ซึ่งคล้ายตารางข้อมูล
matplotlib.pyplot as plt: ใช้สำหรับสร้างกราฟและแผนภาพต่างๆ
seaborn as sns: ใช้สำหรับสร้างกราฟสถิติที่สวยงามและใช้งานง่าย สร้างขึ้นบน Matplotlib อีกที
sklearn.model_selection.train_test_split: ฟังก์ชันสำหรับแบ่งข้อมูลออกเป็นชุดสำหรับฝึกโมเดล (Training set) และชุดสำหรับทดสอบโมเดล (Test set)
sklearn.linear_model.LinearRegression: คลาสสำหรับสร้างแบบจำลองการถดถอยเชิงเส้น ซึ่งเป็นอัลกอริทึม Machine Learning สำหรับทำนายค่าตัวเลขต่อเนื่อง
sklearn.metrics.mean_squared_error, r2_score: ฟังก์ชันสำหรับคำนวณตัวชี้วัดประสิทธิภาพของโมเดลการถดถอย คือ Mean Squared Error (MSE) และ R-squared

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. เตรียมข้อมูลจำลอง (Dataset)
# Hours: ชั่วโมงการอ่าน, Distance: ระยะทางจากบ้าน(กม.), Score: คะแนนสอบ
data = {
    'Hours_Studied': [2, 3, 4, 5, 6, 7, 8, 9, 10, 3, 5, 2, 8, 4, 1],
    'Distance_KM': [15, 12, 10, 5, 2, 3, 1, 0.5, 1, 20, 8, 25, 2, 10, 30],
    'Exam_Score': [55, 60, 68, 75, 82, 88, 92, 95, 98, 50, 70, 45, 90, 65, 35]
}

df = pd.DataFrame(data)

# 2. กำหนดระยะทางเป็นอัตรภาคชั้น
bins = [0, 5, 10, 15, 20, np.inf] # Bins: <5, 6-10, 11-15, 16-20, >20
labels = ['<5 KM', '6-10 KM', '11-15 KM', '16-20 KM', '>20 KM']
df['Distance_Category'] = pd.cut(df['Distance_KM'], bins=bins, labels=labels, right=True, include_lowest=True)

# Perform one-hot encoding on 'Distance_Category'
distance_dummies = pd.get_dummies(df['Distance_Category'], prefix='Distance')

# 3. กำหนดตัวแปร X (Features) และ y (Target)
# รวม Hours_Studied กับ One-Hot Encoded Distance Categories
X = pd.concat([df[['Hours_Studied']], distance_dummies], axis=1)
y = df['Exam_Score']

# 4. แบ่งข้อมูลเป็นชุด Train (80%) และ Test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. สร้างและฝึกสอนโมเดล (Training)
model = LinearRegression()
model.fit(X_train, y_train)

# 6. ทำนายผลและประเมินประสิทธิภาพ
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"--- ผลการทดสอบโมเดล ---")
print(f"ค่าความคลาดเคลื่อน (MSE): {mse:.2f}")
print(f"ความแม่นยำของโมเดล (R-squared): {r2:.2f}") # ยิ่งใกล้ 1.0 ยิ่งแม่นยำ

# 7. แสดงผลการวิเคราะห์ด้วยกราฟ (กราฟสำหรับ Distance_KM จะแสดงก่อนการแปลงเป็นประเภท)
plt.figure(figsize=(10, 5))

# กราฟความสัมพันธ์ระหว่าง ชั่วโมงการอ่าน กับ คะแนน
plt.subplot(1, 2, 1)
sns.regplot(x='Hours_Studied', y='Exam_Score', data=df, color='blue')
plt.title('Hours vs Score')

# กราฟความสัมพันธ์ระหว่าง ระยะทาง (แบบต่อเนื่องเดิม) กับ คะแนน
plt.subplot(1, 2, 2)
sns.regplot(x='Distance_KM', y='Exam_Score', data=df, color='red')
plt.title('Distance (Original) vs Score')

plt.tight_layout()
plt.show()

# กราฟสำหรับ Distance Category vs Score
plt.figure(figsize=(8, 5))
sns.boxplot(x='Distance_Category', y='Exam_Score', data=df.sort_values('Distance_Category'), palette='viridis')
plt.title('Distance Category vs Score')
plt.xlabel('Distance Category')
plt.ylabel('Exam Score')
plt.show()


# 8. ลองนำไปใช้งาน (Prediction)
# สมมติว่านักเรียน A: อ่านหนังสือ 6 ชม. และบ้านไกล 15 กม.
# ฟังก์ชันช่วยแปลงระยะทางเป็น One-Hot Encoding สำหรับการทำนาย
def preprocess_new_distance(distance_km, bins, labels, all_distance_categories):
    category = pd.cut([distance_km], bins=bins, labels=labels, right=True, include_lowest=True)[0]
    # Create a dummy dataframe with all possible categories, then fill the one for the new distance
    dummy_df = pd.DataFrame(0, index=[0], columns=[f'Distance_{cat}' for cat in all_distance_categories])
    dummy_df[f'Distance_{category}'] = 1
    return dummy_df

# Get all possible categories from the training data for consistent encoding
all_distance_categories = sorted(distance_dummies.columns.str.replace('Distance_', '').tolist())

new_student_hours = 6
new_student_distance_km = 15

# Preprocess the new student's distance
new_distance_encoded = preprocess_new_distance(new_student_distance_km, bins, labels, all_distance_categories)

# Prepare the feature array for the new student
# Ensure the order of columns matches X_train
new_student_features = pd.DataFrame([[new_student_hours]], columns=['Hours_Studied'])
new_student_features = pd.concat([new_student_features, new_distance_encoded], axis=1)

# Reindex to ensure column order consistency with X_train
new_student_features = new_student_features.reindex(columns=X_train.columns, fill_value=0)

predicted_score = model.predict(new_student_features)
print(f"\nทำนายคะแนนของนักเรียน A: {predicted_score[0]:.2f} คะแนน")

ผู้จัดทำ

จิตราภรณ์ วรรณวิลัย

ชลิดา ทิตศานติกุล

จิรัชยา โชคกำเนิด

นรีกานต์ นึกสม
