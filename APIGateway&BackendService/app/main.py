import os
from typing import Dict, Optional, Any, Tuple
import json
import asyncio

from fastapi import FastAPI, Request, Depends, HTTPException, status, WebSocket, WebSocketDisconnect, Response
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRouter
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field
import httpx

# -----------------------------
# Configuration via ENV
# -----------------------------
GATEWAY_HOST: str = os.getenv("GATEWAY_HOST", "0.0.0.0")
GATEWAY_PORT: int = int(os.getenv("GATEWAY_PORT", "8080"))

USER_SERVICE_URL: str = os.getenv("USER_SERVICE_URL", "http://localhost:9001")
NOTIFICATION_SERVICE_URL: str = os.getenv("NOTIFICATION_SERVICE_URL", "http://localhost:9002")
HOTEL_MENU_SERVICE_URL: str = os.getenv("HOTEL_MENU_SERVICE_URL", "http://localhost:9003")
ORDER_SERVICE_URL: str = os.getenv("ORDER_SERVICE_URL", "http://localhost:9004")
PAYMENT_SERVICE_URL: str = os.getenv("PAYMENT_SERVICE_URL", "http://localhost:9005")
PROMOTION_SERVICE_URL: str = os.getenv("PROMOTION_SERVICE_URL", "http://localhost:9006")
LOCATION_SERVICE_URL: str = os.getenv("LOCATION_SERVICE_URL", "http://localhost:9007")
REVIEW_RATING_SERVICE_URL: str = os.getenv("REVIEW_RATING_SERVICE_URL", "http://localhost:9008")

# Optional service-to-service auth token
INTERNAL_SERVICE_TOKEN: Optional[str] = os.getenv("INTERNAL_SERVICE_TOKEN")

HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT_SECONDS", "20"))


# -----------------------------
# Models for OpenAPI clarity
# -----------------------------

class RefreshRequest(BaseModel):
    refreshToken: str = Field(..., description="Refresh token issued by UserService")


# -----------------------------
# JWT Auth dependency (verification delegating to UserService)
# -----------------------------
async def verify_jwt(request: Request) -> Optional[Dict[str, Any]]:
    """
    Simple bearer token presence check and optional introspection via UserService.
    If USER_SERVICE_URL exposes /auth/introspect, we call it; otherwise accept presence of token.
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid Authorization header")
    token = auth_header.split(" ", 1)[1].strip()
    # Attempt introspection (best-effort). If it fails, still allow to proceed to downstream that will enforce auth
    try:
        url = f"{USER_SERVICE_URL.rstrip('/')}/auth/introspect"
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.post(url, json={"token": token}, headers=_service_headers())
            if resp.status_code == 200:
                return resp.json()
    except Exception:
        # swallow errors - rely on downstream services to enforce auth if needed
        pass
    return None


def _service_headers(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    headers = {}
    if INTERNAL_SERVICE_TOKEN:
        headers["X-Internal-Token"] = INTERNAL_SERVICE_TOKEN
    if extra:
        headers.update(extra)
    return headers


def _proxy_headers_from_request(request: Request, override: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    # Forward most headers except host-related hop-by-hop headers
    excluded = {"host", "content-length", "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
                "te", "trailers", "transfer-encoding", "upgrade"}
    headers = {k: v for k, v in request.headers.items() if k.lower() not in excluded}
    # add internal header
    headers.update(_service_headers())
    if override:
        headers.update(override)
    return headers


async def _read_body_bytes(request: Request) -> bytes:
    body = await request.body()
    return body or b""


async def _forward(
    request: Request,
    method: str,
    target_url: str,
    include_body: bool = True,
) -> Response:
    """
    Forward the incoming request to the given target_url using httpx and stream back the response.
    """
    params = dict(request.query_params)
    headers = _proxy_headers_from_request(request)
    content = await _read_body_bytes(request) if include_body else None

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
        try:
            resp = await client.request(method=method, url=target_url, params=params, headers=headers, content=content)
        except httpx.ReadTimeout:
            raise HTTPException(status_code=504, detail="Upstream timeout")
        except httpx.ConnectError:
            raise HTTPException(status_code=502, detail="Cannot connect to upstream service")
        except httpx.HTTPError as ex:
            raise HTTPException(status_code=502, detail=f"Upstream error: {str(ex)}")

    # Build downstream response
    excluded_resp_headers = {"content-encoding", "transfer-encoding", "connection", "keep-alive"}
    response_headers = {k: v for k, v in resp.headers.items() if k.lower() not in excluded_resp_headers}
    return Response(content=resp.content, status_code=resp.status_code, headers=response_headers, media_type=resp.headers.get("content-type"))


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title="Food Delivery API Gateway",
    description="Central API Gateway for the Food Delivery platform. Proxies to domain services and enforces security.",
    version="1.0.0",
    openapi_tags=[
        {"name": "Auth", "description": "Authentication and token management"},
        {"name": "Users", "description": "User profile and account endpoints"},
        {"name": "Hotels", "description": "Hotel discovery and details"},
        {"name": "Menus", "description": "Menu listing"},
        {"name": "Cart", "description": "Shopping cart operations"},
        {"name": "Orders", "description": "Order placement and tracking"},
        {"name": "Payments", "description": "Payment intent and status"},
        {"name": "Reviews", "description": "Review and rating operations"},
        {"name": "Promotions", "description": "Promotion validation"},
        {"name": "Notifications", "description": "Notification publishing and streaming"},
        {"name": "Location", "description": "Location tracking"},
    ],
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("CORS_ALLOW_ORIGINS", "*")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Health and docs helper
# -----------------------------
@app.get("/health", tags=["Health"], summary="Health check", description="Returns gateway health status")
# PUBLIC_INTERFACE
def health() -> Dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "ok"}


# -----------------------------
# Auth routes -> UserService
# -----------------------------
auth_router = APIRouter(prefix="/auth", tags=["Auth"])

@auth_router.post("/register", summary="Register a new user")
# PUBLIC_INTERFACE
async def register(request: Request):
    """Proxy user registration to UserService."""
    url = f"{USER_SERVICE_URL.rstrip('/')}/auth/register"
    return await _forward(request, "POST", url)

@auth_router.post("/login", summary="Authenticate user")
# PUBLIC_INTERFACE
async def login(request: Request):
    """Proxy login to UserService."""
    url = f"{USER_SERVICE_URL.rstrip('/')}/auth/login"
    return await _forward(request, "POST", url)

@auth_router.post("/refresh", summary="Refresh access token", dependencies=[])
# PUBLIC_INTERFACE
async def refresh_token(request: Request, _: RefreshRequest):
    """Proxy token refresh to UserService. This endpoint is public (no bearer required)."""
    url = f"{USER_SERVICE_URL.rstrip('/')}/auth/refresh"
    return await _forward(request, "POST", url)

app.include_router(auth_router)


# -----------------------------
# Users -> UserService
# -----------------------------
users_router = APIRouter(prefix="/users", tags=["Users"], dependencies=[Depends(verify_jwt)])

@users_router.get("/me", summary="Get current user")
# PUBLIC_INTERFACE
async def get_me(request: Request):
    """Proxy to UserService to get current user profile."""
    url = f"{USER_SERVICE_URL.rstrip('/')}/users/me"
    return await _forward(request, "GET", url, include_body=False)

@users_router.patch("/me", summary="Update current user")
# PUBLIC_INTERFACE
async def update_me(request: Request):
    """Proxy to UserService to update current user profile."""
    url = f"{USER_SERVICE_URL.rstrip('/')}/users/me"
    return await _forward(request, "PATCH", url)

app.include_router(users_router)


# -----------------------------
# Hotels & Menus -> Hotel&MenuService
# -----------------------------
hotels_router = APIRouter(tags=["Hotels", "Menus"])

@hotels_router.get("/hotels", summary="Search and list nearby hotels")
# PUBLIC_INTERFACE
async def list_hotels(request: Request):
    """Proxy to Hotel & Menu Service for hotel listing/search."""
    url = f"{HOTEL_MENU_SERVICE_URL.rstrip('/')}/hotels"
    return await _forward(request, "GET", url, include_body=False)

@hotels_router.get("/hotels/{hotelId}", summary="Get hotel details")
# PUBLIC_INTERFACE
async def get_hotel(hotelId: str, request: Request):
    """Proxy to Hotel & Menu Service for hotel details."""
    url = f"{HOTEL_MENU_SERVICE_URL.rstrip('/')}/hotels/{hotelId}"
    return await _forward(request, "GET", url, include_body=False)

@hotels_router.get("/hotels/{hotelId}/menu", summary="Get menu for a hotel")
# PUBLIC_INTERFACE
async def get_menu(hotelId: str, request: Request):
    """Proxy to Hotel & Menu Service for menu listing."""
    url = f"{HOTEL_MENU_SERVICE_URL.rstrip('/')}/hotels/{hotelId}/menu"
    return await _forward(request, "GET", url, include_body=False)

app.include_router(hotels_router)


# -----------------------------
# Cart & Orders -> OrderService
# -----------------------------
orders_router = APIRouter(tags=["Cart", "Orders"], dependencies=[Depends(verify_jwt)])

@orders_router.get("/cart", summary="Get current cart")
# PUBLIC_INTERFACE
async def get_cart(request: Request):
    """Proxy to Order Service for retrieving current cart."""
    url = f"{ORDER_SERVICE_URL.rstrip('/')}/cart"
    return await _forward(request, "GET", url, include_body=False)

@orders_router.post("/cart", summary="Replace cart items")
# PUBLIC_INTERFACE
async def replace_cart(request: Request):
    """Proxy to Order Service for replacing cart."""
    url = f"{ORDER_SERVICE_URL.rstrip('/')}/cart"
    return await _forward(request, "POST", url)

@orders_router.post("/cart/items", summary="Add item to cart")
# PUBLIC_INTERFACE
async def add_cart_item(request: Request):
    """Proxy to Order Service to add an item to cart."""
    url = f"{ORDER_SERVICE_URL.rstrip('/')}/cart/items"
    return await _forward(request, "POST", url)

@orders_router.delete("/cart/items/{itemId}", summary="Remove item from cart")
# PUBLIC_INTERFACE
async def remove_cart_item(itemId: str, request: Request):
    """Proxy to Order Service to remove an item from cart."""
    url = f"{ORDER_SERVICE_URL.rstrip('/')}/cart/items/{itemId}"
    return await _forward(request, "DELETE", url, include_body=False)

@orders_router.post("/orders", summary="Create a new order")
# PUBLIC_INTERFACE
async def create_order(request: Request):
    """Proxy to Order Service to create new order."""
    url = f"{ORDER_SERVICE_URL.rstrip('/')}/orders"
    return await _forward(request, "POST", url)

@orders_router.get("/orders", summary="List my orders")
# PUBLIC_INTERFACE
async def list_orders(request: Request):
    """Proxy to Order Service for listing orders of current user."""
    url = f"{ORDER_SERVICE_URL.rstrip('/')}/orders"
    return await _forward(request, "GET", url, include_body=False)

@orders_router.get("/orders/{orderId}", summary="Get order by id")
# PUBLIC_INTERFACE
async def get_order(orderId: str, request: Request):
    """Proxy to Order Service to get order details."""
    url = f"{ORDER_SERVICE_URL.rstrip('/')}/orders/{orderId}"
    return await _forward(request, "GET", url, include_body=False)

@orders_router.patch("/orders/{orderId}", summary="Update order status")
# PUBLIC_INTERFACE
async def update_order(orderId: str, request: Request):
    """Proxy to Order Service to update order (e.g., cancel)."""
    url = f"{ORDER_SERVICE_URL.rstrip('/')}/orders/{orderId}"
    return await _forward(request, "PATCH", url)

app.include_router(orders_router)


# -----------------------------
# Payments -> PaymentService
# -----------------------------
payments_router = APIRouter(prefix="/payments", tags=["Payments"], dependencies=[Depends(verify_jwt)])

@payments_router.post("/intent", summary="Create payment intent")
# PUBLIC_INTERFACE
async def create_payment_intent(request: Request):
    """Proxy to Payment Service to create payment intent."""
    url = f"{PAYMENT_SERVICE_URL.rstrip('/')}/payments/intent"
    return await _forward(request, "POST", url)

@payments_router.get("/{paymentId}", summary="Get payment status")
# PUBLIC_INTERFACE
async def get_payment_status(paymentId: str, request: Request):
    """Proxy to Payment Service to retrieve payment status."""
    url = f"{PAYMENT_SERVICE_URL.rstrip('/')}/payments/{paymentId}"
    return await _forward(request, "GET", url, include_body=False)

app.include_router(payments_router)


# -----------------------------
# Reviews -> Review&RatingService
# -----------------------------
reviews_router = APIRouter(prefix="/reviews", tags=["Reviews"], dependencies=[Depends(verify_jwt)])

@reviews_router.post("", summary="Create a review")
# PUBLIC_INTERFACE
async def create_review(request: Request):
    """Proxy to Review & Rating Service to submit a review."""
    url = f"{REVIEW_RATING_SERVICE_URL.rstrip('/')}/reviews"
    return await _forward(request, "POST", url)

@reviews_router.get("", summary="List reviews for a hotel")
# PUBLIC_INTERFACE
async def list_reviews(request: Request):
    """Proxy to Review & Rating Service to list reviews for a hotel."""
    url = f"{REVIEW_RATING_SERVICE_URL.rstrip('/')}/reviews"
    return await _forward(request, "GET", url, include_body=False)

app.include_router(reviews_router)


# -----------------------------
# Promotions -> PromotionService
# -----------------------------
promotions_router = APIRouter(prefix="/promotions", tags=["Promotions"])

@promotions_router.post("/validate", summary="Validate a promotion", dependencies=[Depends(verify_jwt)])
# PUBLIC_INTERFACE
async def validate_promotion(request: Request):
    """Proxy to Promotion Service to validate a promo code."""
    url = f"{PROMOTION_SERVICE_URL.rstrip('/')}/promotions/validate"
    return await _forward(request, "POST", url)

app.include_router(promotions_router)


# -----------------------------
# Notifications -> NotificationService
# -----------------------------
notifications_router = APIRouter(prefix="/notifications", tags=["Notifications"], dependencies=[Depends(verify_jwt)])

@notifications_router.post("", summary="Send a notification")
# PUBLIC_INTERFACE
async def send_notification(request: Request):
    """Proxy to Notification Service to send a notification."""
    url = f"{NOTIFICATION_SERVICE_URL.rstrip('/')}/notifications"
    return await _forward(request, "POST", url)

app.include_router(notifications_router)


# -----------------------------
# Location -> LocationService
# -----------------------------
location_router = APIRouter(prefix="/location", tags=["Location"], dependencies=[Depends(verify_jwt)])

@location_router.post("/updates", summary="Submit a live location update")
# PUBLIC_INTERFACE
async def submit_location_update(request: Request):
    """Proxy to Location Service to submit live location updates."""
    url = f"{LOCATION_SERVICE_URL.rstrip('/')}/location/updates"
    return await _forward(request, "POST", url)

@location_router.get("/track/{orderId}", summary="Get latest tracking location")
# PUBLIC_INTERFACE
async def get_tracking(orderId: str, request: Request):
    """Proxy to Location Service to get the latest tracking location for an order."""
    url = f"{LOCATION_SERVICE_URL.rstrip('/')}/location/track/{orderId}"
    return await _forward(request, "GET", url, include_body=False)

app.include_router(location_router)


# -----------------------------
# WebSocket bridge to NotificationService
# -----------------------------
# PUBLIC_INTERFACE
@app.websocket("/ws/notifications")
async def ws_notifications(websocket: WebSocket):
    """
    WebSocket bridge that connects the client to the downstream NotificationService WebSocket.
    It forwards messages bi-directionally.
    Usage: connect to ws://<gateway>/ws/notifications?userId=<id>&token=<bearer_token_optional>
    """
    await websocket.accept()

    # Compose downstream URL (assuming NotificationService exposes /ws/notifications or /notifications/stream)
    # Prefer documented path /notifications/stream (HTTP placeholder), but most WS servers reuse same path.
    # We'll try /ws/notifications first; fallback to /notifications/stream.
    downstream_candidates = [
        f"{NOTIFICATION_SERVICE_URL.rstrip('/')}/ws/notifications",
        f"{NOTIFICATION_SERVICE_URL.rstrip('/')}/notifications/stream",
    ]

    params = dict(websocket.query_params)
    headers = {}
    if INTERNAL_SERVICE_TOKEN:
        headers["X-Internal-Token"] = INTERNAL_SERVICE_TOKEN

    async with httpx.AsyncClient(timeout=None) as client:
        ws = None
        last_error = None
        for url in downstream_candidates:
            try:
                ws = await client.ws_connect(url, params=params, headers=headers)
                break
            except Exception as ex:
                last_error = ex
                continue
        if ws is None:
            await websocket.close(code=1011)
            return

        async def client_to_service():
            try:
                while True:
                    data = await websocket.receive_text()
                    await ws.send_text(data)
            except WebSocketDisconnect:
                await ws.aclose()
            except Exception:
                await ws.aclose()
                try:
                    await websocket.close()
                except Exception:
                    pass

        async def service_to_client():
            try:
                async for msg in ws.iter_text():
                    await websocket.send_text(msg)
            except Exception:
                try:
                    await websocket.close()
                except Exception:
                    pass

        await asyncio.gather(client_to_service(), service_to_client())


# -----------------------------
# Custom OpenAPI override to align minor doc hints
# -----------------------------
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi  # type: ignore


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=GATEWAY_HOST, port=GATEWAY_PORT, reload=False)
