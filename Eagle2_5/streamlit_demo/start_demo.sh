python controller.py &
python model_worker.py --model-path nvidia/Eagle2.5-8B \
--model-name Eagle2_5  --port 6214 --worker-address http://127.0.0.1:6214 &
streamlit run app.py
