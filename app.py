import streamlit as st
import pandas as pd
import pickle

st.title("Sektör Emisyon Kümeleme :factory:")
st.write("Bir sektörün sera gazı emisyon faktörlerini girin (kg CO2e / USD); model sektörün hangi kümeye girdiğini bulsun.")

model=pickle.load(open('ghg_kmeans.pkl','rb'))
scaler=pickle.load(open('ghg_scaler.pkl','rb'))

without_margins=st.number_input('Marjsız emisyon faktörü',0.0,50.0,0.5)
margins=st.number_input('Marj emisyon faktörü',0.0,10.0,0.05)
with_margins=st.number_input('Marjlı emisyon faktörü',0.0,50.0,0.55)

if st.button('Kümeyi bul'):
    veri=pd.DataFrame([[without_margins,margins,with_margins]],columns=['without_margins','margins','with_margins'])
    kume=model.predict(pd.DataFrame(scaler.transform(veri),columns=veri.columns))[0]
    st.success(f'Bu sektör {kume}. kümeye ait.')
