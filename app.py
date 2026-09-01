import streamlit as st
import pandas as pd
from io import StringIO
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="วิเคราะห์ดิน+พยากรณ์ผลผลิตลำไย")
st.title("ระบบวิเคราะห์สภาพดินและพยากรณ์ผลผลิตลำไย")

# ---------- ข้อมูลตัวอย่างสวนลำไย (แทนด้วยข้อมูลจริงได้) ----------
ข้อมูล = """pH,N,P,K,OM,moisture,temp,rain,density,chlorate,yield
6.0,45,25,180,2.5,55,27,1200,45,8,920
5.5,30,18,140,1.8,48,29,900,50,6,720
6.2,55,30,210,3.0,60,26,1400,40,10,1080
5.8,40,22,160,2.2,52,28,1100,48,7,850
6.4,60,32,230,3.2,62,25,1500,38,11,1150
5.2,25,15,120,1.5,45,30,800,55,5,600
6.1,50,28,195,2.8,58,27,1300,42,9,1000
5.6,35,20,150,2.0,50,28,1000,50,6,780
6.3,58,31,220,3.1,61,26,1450,39,10,1120
5.4,28,17,130,1.7,47,29,850,52,5,660
5.9,43,24,170,2.4,54,27,1150,46,8,890
6.5,62,33,240,3.3,63,25,1550,37,12,1200
5.7,38,21,155,2.1,51,28,1050,49,7,820
6.0,48,27,185,2.6,56,27,1250,44,9,960
"""
df = pd.read_csv(StringIO(ข้อมูล))
X = df.drop(columns=["yield"])
y = df["yield"]
model = RandomForestRegressor(n_estimators=200, random_state=42).fit(X, y)

# ===== ส่วนที่ 1: วิเคราะห์สภาพดิน =====
st.header("1) วิเคราะห์สภาพดิน")
c1, c2 = st.columns(2)
pH = c1.number_input("ค่า pH ดิน", 3.0, 9.0, 6.0, 0.1)
N  = c1.number_input("ไนโตรเจน N (mg/kg)", 0, 200, 40)
P  = c1.number_input("ฟอสฟอรัส P (mg/kg)", 0, 200, 25)
K  = c2.number_input("โพแทสเซียม K (mg/kg)", 0, 400, 180)
OM = c2.number_input("อินทรียวัตถุ OM (%)", 0.0, 10.0, 2.5, 0.1)
moisture = c2.number_input("ความชื้นดิน (%)", 0, 100, 55)

st.subheader("คำแนะนำจัดการดิน (สำหรับลำไย)")
if pH < 5.5:
    st.warning("ดินเป็นกรดเกินไป → ใส่ปูนขาวปรับ pH (ลำไยชอบ 5.5–6.5)")
elif pH > 6.5:
    st.warning("ดินค่อนข้างด่าง → ระวังธาตุอาหารบางตัวถูกตรึง")
else:
    st.success("pH เหมาะกับลำไย")
if OM < 2.0:
    st.write("• อินทรียวัตถุต่ำ → เพิ่มปุ๋ยคอก/ปุ๋ยหมัก")
if N < 30:
    st.write("• ไนโตรเจนต่ำ → บำรุงใบด้วยปุ๋ย N")
if K < 150:
    st.write("• โพแทสเซียมต่ำ → สำคัญต่อการติดผลลำไย ควรเพิ่ม K")

# ===== ส่วนที่ 2: พยากรณ์ผลผลิต =====
st.header("2) พยากรณ์ผลผลิตลำไย")
c3, c4 = st.columns(2)
temp = c3.number_input("อุณหภูมิเฉลี่ย (°C)", 15, 40, 27)
rain = c3.number_input("ปริมาณน้ำฝน (มม./ปี)", 0, 3000, 1200)
density  = c4.number_input("ความหนาแน่น (ต้น/ไร่)", 10, 100, 45)
chlorate = c4.number_input("โพแทสเซียมคลอเรต (กก./ไร่)", 0, 30, 8)

x_new = pd.DataFrame([[pH, N, P, K, OM, moisture, temp, rain, density, chlorate]],
                     columns=X.columns)
yhat = model.predict(x_new)[0]
st.metric("ผลผลิตคาดการณ์", f"{yhat:.0f} กก./ไร่")

# ปัจจัยสำคัญ
st.subheader("ปัจจัยที่มีผลต่อผลผลิตมากที่สุด")
imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
st.bar_chart(imp)
