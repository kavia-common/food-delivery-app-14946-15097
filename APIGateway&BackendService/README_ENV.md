# Environment configuration

Copy .env.example to .env and adjust values for your environment.

Required downstream services (defaults assume local dev):
- USER_SERVICE_URL=http://localhost:8101
- NOTIFICATION_SERVICE_URL=http://localhost:8106
- HOTEL_MENU_SERVICE_URL=http://localhost:8103
- ORDER_SERVICE_URL=http://localhost:8102
- PAYMENT_SERVICE_URL=http://localhost:8104
- PROMOTION_SERVICE_URL=http://localhost:8108
- LOCATION_SERVICE_URL=http://localhost:8107
- REVIEW_RATING_SERVICE_URL=http://localhost:8105

Shared settings:
- HTTP_TIMEOUT_SECONDS=20
- INTERNAL_SERVICE_TOKEN= (optional)
