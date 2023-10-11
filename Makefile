activate:
	.\myenv\Scripts\Activate

run: 
	uvicorn main:app --port 2000 --reload