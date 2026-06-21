from fastapi import FastAPI
from app.config import get_settings

settings = get_settings()
app = FastAPI(title = settings.app_name)

'''
run the below function everytime the tag /health is called, making sure that the backend is responding
'''
@app.get('/health') 
def health():
    return {'status': 'gud'}