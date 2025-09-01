# API Gateway & Backend Service

FastAPI-based API Gateway that proxies requests to domain services:
- UserService
- NotificationService
- Hotel&MenuService
- OrderService
- PaymentService
- PromotionService
- LocationService
- Review&RatingService

Features:
- JWT bearer middleware (best-effort introspection against UserService)
- Proxy endpoints based on openapi/gateway.yaml
- WebSocket bridge at /ws/notifications to NotificationService
- Async httpx client with timeout and error handling
- Health endpoint at /health
- OpenAPI docs at /docs and /openapi.json

## Run

1. Set environment variables (see .env.example).
2. Install dependencies:

   pip install -r requirements.txt

3. Start:

   uvicorn app.main:app --host ${GATEWAY_HOST:-0.0.0.0} --port ${GATEWAY_PORT:-8080}

## Environment variables

See .env.example for the full list. At minimum, set the downstream service URLs to match your environment.
