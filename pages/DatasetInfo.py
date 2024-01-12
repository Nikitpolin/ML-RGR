import streamlit as st

st.title("Информация о наборе данных")
st.header("Тематика датасета")
st.write("Информация об объектах НАСА")
st.header("Описание признаков")
st.write("- id: идентификатор ")
st.write("- name: название, данное НАСА")
st.write("- est_diameter_min: минимальный расчетный диаметр в километрах")
st.write("- est_diameter_max: максимальный расчетный диаметр в километрах")
st.write("- relative_velocity: скорость относительно земли")
st.write("- miss_distance: расстояние в километрах от Земли")
st.write("- absolute_magnitude: собственная светимость")
st.write("- hazardous: является ли астериод потенциально опасным или нет")
st.header("Особенности предобработки данных")
st.write("В датасете необходимо предугатывать был ли является ли астериод потенциально опасным или нет. Опасность опеределяется показателем 1 или 0.")
st.write("1 - астероид опасен")
st.write("0 - астероид не опасен")
st.write("В датасете присутствовали категориальные признаки, так что был проведен One-hot кодирование")
st.write("В датасете были пропущенные значения. Они были заполнены:")
st.write("- медианой для целых чисел")
st.write("- средним значением для действительных чисел")
st.write("- модой для остальных типов данных")
st.write("Были удалены дубликаты")
st.write("Был проведен EDA (см. Vusualization) и удалены выбросы")
st.write("Числовые признаки были масштабированны. Дисбаланс был устраннен алгоритмом NearMiss")