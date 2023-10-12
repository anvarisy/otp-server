activate:
	.\myenv\Scripts\Activate

run: 
	uvicorn main:app --host 0.0.0.0 --port 2000 --reload