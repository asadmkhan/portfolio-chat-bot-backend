from fastapi import APIRouter   

router = APIRouter()

@router.get("/health", summary="Health Check", description="Check the health status of the application.")
async def health_check():
    return {"status": "healthy"}