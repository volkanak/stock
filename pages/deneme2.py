import streamlit as st
import time

with st.spinner(text="In progress"):
       time.sleep(3)
       st.success("Done")
bar = st.progress(50)
time.sleep(3)
bar.progress(100)

with st.status("Authenticating...") as s:
  time.sleep(2)
  st.write("Some long response.")
  s.update(label="Response")

# st.balloons()
# st.snow()
st.toast("Warming up...")
st.error("Error message")
st.warning("Warning message")
st.info("Info message")
st.success("Success message")
# st.exception(e)