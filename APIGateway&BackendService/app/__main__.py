from .main import app, GATEWAY_HOST, GATEWAY_PORT

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=GATEWAY_HOST, port=GATEWAY_PORT, reload=False)
