from fastapi import FastAPI
from Service import Service
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = Service()
@app.on_event("startup")
async def startup_event():
   
    service.run()

@app.get("/df")
async def df():
    return service.get_df()

@app.get("/order_info")
async def order_info():
    return service.get_order_info()