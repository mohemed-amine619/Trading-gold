# 🔐 Security Guide — Trading-Gold System

> [!CAUTION]
> This system handles **live financial trading**, real money, and sensitive broker credentials. Every security control described here is **mandatory**, not optional. A single leaked API key or misconfigured endpoint can result in financial loss or account compromise.

---

## Table of Contents

1. [Never Hardcode Credentials](#1-never-hardcode-credentials)
2. [Environment File (.env) Usage](#2-environment-file-env-usage)
3. [Secret Manager in Production](#3-secret-manager-in-production)
4. [Dashboard Authentication (JWT)](#4-dashboard-authentication-jwt)
5. [HTTPS Requirement](#5-https-requirement)
6. [API Rate Limiting](#6-api-rate-limiting)
7. [CORS Configuration](#7-cors-configuration)
8. [Input Validation with Pydantic](#8-input-validation-with-pydantic)
9. [SQL Injection Prevention](#9-sql-injection-prevention)
10. [Telegram Token Protection](#10-telegram-token-protection)
11. [MT5 Credentials Protection](#11-mt5-credentials-protection)
12. [LLM API Key Protection](#12-llm-api-key-protection)
13. [Audit Logging](#13-audit-logging)
14. [Access Control & Role-Based Permissions](#14-access-control--role-based-permissions)
15. [Security Checklist Before Going Live](#15-security-checklist-before-going-live)

---

## 1. Never Hardcode Credentials

Hardcoding secrets in source code is the **#1 cause of credential leaks**. Even if you delete the value later, Git history retains it permanently.

### ❌ WRONG — Never do this

```python
# config.py  — DO NOT EVER DO THIS
MT5_PASSWORD     = "MyBrokerPass123!"
TELEGRAM_TOKEN   = "7123456789:AAF_xYzAbCdEfGhIjKlMnOpQrStUvWxYz"
OPENAI_API_KEY   = "sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
DB_URL           = "postgresql://admin:secret@localhost:5432/trading"
JWT_SECRET_KEY   = "supersecretjwt"
```

```python
# Even passing them as default values is dangerous
def connect(password="MyBrokerPass123!"):
    ...
```

### ✅ RIGHT — Always use environment variables

```python
# config.py
import os
from dotenv import load_dotenv

load_dotenv()  # Loads from .env file in development

MT5_PASSWORD   = os.environ["MT5_PASSWORD"]      # Raises KeyError if missing (fail-fast)
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
DB_URL         = os.environ["DATABASE_URL"]
JWT_SECRET_KEY = os.environ["JWT_SECRET_KEY"]
```

> [!TIP]
> Use `os.environ["KEY"]` (raises `KeyError`) instead of `os.getenv("KEY")` (returns `None`) for required secrets. Failing loudly on startup is far better than silently running with missing credentials.

### Pre-commit Hook to Catch Secrets

Install `detect-secrets` to automatically block commits that contain secrets:

```bash
pip install detect-secrets pre-commit
```

Create `.pre-commit-config.yaml` in the project root:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args: ["--baseline", ".secrets.baseline"]
        exclude: "tests/fixtures/"

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: check-added-large-files
      - id: check-merge-conflict
      - id: detect-private-key          # Catches RSA/DSA private keys
```

Initialize and install:

```bash
# Create baseline of known (acceptable) secrets
detect-secrets scan > .secrets.baseline

# Install hooks into .git/hooks/
pre-commit install

# Test manually
pre-commit run --all-files
```

> [!IMPORTANT]
> Commit `.secrets.baseline` and `.pre-commit-config.yaml` to the repository. Every developer **must** run `pre-commit install` after cloning.

---

## 2. Environment File (.env) Usage

### Installing python-dotenv

```bash
pip install python-dotenv
```

### Loading the .env File

```python
# At the very top of your entry point (main.py, app.py, etc.)
from dotenv import load_dotenv
import os

# load_dotenv() reads .env in the current directory (or searches parents)
load_dotenv(override=False)   # override=False: real env vars take priority over .env
```

### .env File Template

Save this as `.env.example` and commit it. **Never commit `.env` itself.**

```dotenv
# ============================================================
# Trading-Gold — Environment Variables Template
# Copy this file to .env and fill in real values.
# NEVER commit the actual .env file.
# ============================================================

# --- MetaTrader 5 ---
MT5_LOGIN=123456789
MT5_PASSWORD=your_mt5_password_here
MT5_SERVER=YourBroker-Live
MT5_TIMEOUT=10000

# --- Telegram Bot ---
TELEGRAM_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
TELEGRAM_ADMIN_CHAT_IDS=111111111,222222222   # Comma-separated admin IDs

# --- OpenAI / LLM ---
OPENAI_API_KEY=sk-proj-your_key_here
OPENAI_MODEL=gpt-4o

# --- Database ---
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/trading_gold
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20

# --- JWT Authentication ---
JWT_SECRET_KEY=generate_a_long_random_string_here_min_64_chars
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7

# --- Dashboard ---
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8000
ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com

# --- Security ---
BCRYPT_ROUNDS=12
SECRET_SALT=another_random_secret_for_salting

# --- Environment ---
ENVIRONMENT=production    # development | staging | production
DEBUG=false
LOG_LEVEL=INFO
```

Generate a strong `JWT_SECRET_KEY`:

```bash
python -c "import secrets; print(secrets.token_hex(64))"
```

### .gitignore Setup

Add these lines to your `.gitignore`:

```gitignore
# ============================================================
# Secrets & Credentials — NEVER commit these
# ============================================================
.env
.env.local
.env.production
.env.staging
*.env
secrets/
credentials/
*.pem
*.key
*.p12
*.pfx
service-account*.json

# Python cache
__pycache__/
*.pyc
*.pyo

# IDE
.vscode/settings.json
.idea/

# Logs (may contain sensitive data)
*.log
logs/
```

> [!WARNING]
> If you accidentally committed `.env`, immediately rotate **all** credentials it contained. Then remove it from history: `git filter-repo --path .env --invert-paths`. Treat the leaked secrets as permanently compromised.

---

## 3. Secret Manager in Production

`.env` files are suitable for local development only. In production, use a dedicated secret manager.

### Option A — AWS Secrets Manager

```python
# utils/secrets.py
import boto3
import json
import os
from functools import lru_cache

@lru_cache(maxsize=None)
def get_secret(secret_name: str, region: str = "eu-west-1") -> dict:
    """Fetch a secret from AWS Secrets Manager (cached per process)."""
    client = boto3.client("secretsmanager", region_name=region)
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response["SecretString"])

# Usage
secrets = get_secret("trading-gold/production")
MT5_PASSWORD   = secrets["MT5_PASSWORD"]
TELEGRAM_TOKEN = secrets["TELEGRAM_TOKEN"]
OPENAI_API_KEY = secrets["OPENAI_API_KEY"]
```

Required IAM policy for the EC2/ECS role:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["secretsmanager:GetSecretValue"],
    "Resource": "arn:aws:secretsmanager:eu-west-1:ACCOUNT_ID:secret:trading-gold/*"
  }]
}
```

### Option B — HashiCorp Vault

```python
# utils/vault.py
import hvac
import os

def get_vault_secrets() -> dict:
    client = hvac.Client(
        url=os.environ["VAULT_ADDR"],
        token=os.environ["VAULT_TOKEN"],
    )
    assert client.is_authenticated(), "Vault authentication failed"
    secret = client.secrets.kv.v2.read_secret_version(
        path="trading-gold/credentials"
    )
    return secret["data"]["data"]
```

### Option C — System Environment Variables (Minimal Setup)

On a Linux server, set variables in `/etc/environment` or via systemd unit:

```ini
# /etc/systemd/system/trading-gold.service
[Unit]
Description=Trading Gold Bot

[Service]
Type=simple
User=trading
EnvironmentFile=/etc/trading-gold/secrets.env   # Owned by root, chmod 600
ExecStart=/opt/trading-gold/venv/bin/python main.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
# Secure the secrets file
sudo chmod 600 /etc/trading-gold/secrets.env
sudo chown root:root /etc/trading-gold/secrets.env
```

---

## 4. Dashboard Authentication (JWT)

### Dependencies

```bash
pip install python-jose[cryptography] passlib[bcrypt] fastapi
```

### Password Hashing with bcrypt

```python
# auth/password.py
from passlib.context import CryptContext

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12,   # Cost factor — increase as hardware improves
)

def hash_password(plain_password: str) -> str:
    """Hash a plain-text password. Store the result, never the plain text."""
    return pwd_context.hash(plain_password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a login attempt against a stored hash."""
    return pwd_context.verify(plain_password, hashed_password)
```

### JWT Token Creation and Validation

```python
# auth/jwt.py
from datetime import datetime, timedelta, timezone
from typing import Optional
from jose import JWTError, jwt
import os

SECRET_KEY = os.environ["JWT_SECRET_KEY"]
ALGORITHM  = os.environ.get("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES  = int(os.environ.get("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", 30))
REFRESH_TOKEN_EXPIRE_DAYS    = int(os.environ.get("JWT_REFRESH_TOKEN_EXPIRE_DAYS", 7))


def create_access_token(subject: str, role: str, extra: dict = None) -> str:
    """Create a short-lived access token (default 30 min)."""
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": subject,
        "role": role,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "access",
    }
    if extra:
        payload.update(extra)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(subject: str) -> str:
    """Create a long-lived refresh token (default 7 days)."""
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    payload = {
        "sub": subject,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "refresh",
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    """
    Decode and validate a JWT. Raises JWTError on invalid/expired tokens.
    Never catch this silently — let it propagate to return 401.
    """
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
```

### FastAPI Authentication Dependency

```python
# auth/dependencies.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError
from .jwt import decode_token

security = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """Dependency that validates the Bearer token and returns the payload."""
    try:
        payload = decode_token(credentials.credentials)
        if payload.get("type") != "access":
            raise HTTPException(status_code=401, detail="Invalid token type")
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_admin(current_user: dict = Depends(get_current_user)) -> dict:
    """Dependency that additionally enforces admin role."""
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user
```

### Login Endpoint

```python
# routers/auth.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from auth.password import verify_password
from auth.jwt import create_access_token, create_refresh_token
from db.users import get_user_by_username   # Your DB lookup

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest):
    user = await get_user_by_username(body.username)
    if not user or not verify_password(body.password, user.hashed_password):
        # Same error for both wrong username and wrong password (prevent enumeration)
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token  = create_access_token(subject=user.username, role=user.role)
    refresh_token = create_refresh_token(subject=user.username)
    return TokenResponse(access_token=access_token, refresh_token=refresh_token)
```

> [!IMPORTANT]
> Store refresh tokens in the database with a revocation flag. When a user logs out or changes their password, mark all their refresh tokens as revoked.

---

## 5. HTTPS Requirement

### Nginx Reverse Proxy Configuration

```nginx
# /etc/nginx/sites-available/trading-gold
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;

    # Force all HTTP traffic to HTTPS
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com www.yourdomain.com;

    # --- TLS/SSL Certificate (Let's Encrypt) ---
    ssl_certificate     /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    # --- Strong TLS Settings ---
    ssl_protocols       TLSv1.2 TLSv1.3;
    ssl_ciphers         ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache   shared:SSL:10m;
    ssl_session_timeout 1d;

    # --- Security Headers ---
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Content-Security-Policy "default-src 'self'" always;

    # --- Proxy to FastAPI ---
    location / {
        proxy_pass         http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header   Upgrade $http_upgrade;
        proxy_set_header   Connection "upgrade";
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
    }
}
```

### Free TLS Certificate with Let's Encrypt

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Obtain and auto-install certificate
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com

# Verify auto-renewal
sudo certbot renew --dry-run
```

### Self-Signed Certificate (Development/Staging Only)

```bash
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem \
    -days 365 -nodes \
    -subj "/C=DZ/ST=Algiers/L=Algiers/O=TradingGold/CN=localhost"
```

> [!WARNING]
> Self-signed certificates must **never** be used in production. Browsers will show security warnings, and automated clients may reject the connection. Use Let's Encrypt (free) or a commercial CA for any public-facing deployment.

---

## 6. API Rate Limiting

### Using slowapi (FastAPI/Starlette)

```bash
pip install slowapi
```

```python
# main.py
from fastapi import FastAPI, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# Use real IP (respects X-Forwarded-For behind nginx)
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200/minute"],   # Global default
)

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
```

```python
# routers/trading.py
from fastapi import APIRouter, Depends, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from auth.dependencies import get_current_user, require_admin

router  = APIRouter(prefix="/trading", tags=["trading"])
limiter = Limiter(key_func=get_remote_address)


@router.post("/start")
@limiter.limit("5/minute")           # Per-IP: max 5 start requests per minute
async def start_trading(request: Request, admin=Depends(require_admin)):
    ...


@router.post("/emergency-stop")
@limiter.limit("10/minute")          # Emergency stop can be triggered more often
async def emergency_stop(request: Request, admin=Depends(require_admin)):
    ...


@router.get("/status")
@limiter.limit("60/minute")          # Status polling: generous limit
async def get_status(request: Request, user=Depends(get_current_user)):
    ...
```

### Per-User Rate Limiting

```python
from auth.dependencies import get_current_user

def get_user_id(request: Request) -> str:
    """Extract username from JWT for per-user rate limiting."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        try:
            from auth.jwt import decode_token
            payload = decode_token(auth_header[7:])
            return payload.get("sub", get_remote_address(request))
        except Exception:
            pass
    return get_remote_address(request)

user_limiter = Limiter(key_func=get_user_id)

@router.post("/order")
@user_limiter.limit("10/minute")     # Per authenticated user
async def place_order(request: Request, user=Depends(get_current_user)):
    ...
```

---

## 7. CORS Configuration

```python
# main.py
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Parse allowed origins from environment variable
raw_origins = os.environ.get("ALLOWED_ORIGINS", "")
allowed_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]

if not allowed_origins:
    raise RuntimeError("ALLOWED_ORIGINS environment variable must be set")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,       # e.g. ["https://yourdomain.com"]
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
    max_age=600,                          # Cache preflight for 10 minutes
)
```

> [!CAUTION]
> **Never use `allow_origins=["*"]` (wildcard).** In a trading system, a wildcard CORS policy means **any website on the internet** can make authenticated requests on behalf of your logged-in users (Cross-Site Request Forgery). Always whitelist only the exact origins of your dashboard.

| Setting | ❌ Dangerous | ✅ Safe |
|---|---|---|
| `allow_origins` | `["*"]` | `["https://yourdomain.com"]` |
| `allow_credentials` | `True` with wildcard | `True` only with explicit origins |
| `allow_methods` | `["*"]` | Explicit list only |
| `allow_headers` | `["*"]` | Explicit list only |

---

## 8. Input Validation with Pydantic

All API endpoints **must** validate input using Pydantic models. Never accept raw dicts or unvalidated data.

```python
# schemas/trading.py
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Literal
from decimal import Decimal
import re


class OrderRequest(BaseModel):
    """Validated order placement request."""
    model_config = {"extra": "forbid"}  # Reject any fields not defined here

    symbol: str = Field(..., min_length=3, max_length=10, pattern=r"^[A-Z0-9]+$")
    action: Literal["BUY", "SELL"]
    lot_size: Decimal = Field(..., gt=Decimal("0.01"), le=Decimal("100.0"))
    stop_loss: Decimal = Field(..., gt=0)
    take_profit: Decimal = Field(..., gt=0)
    comment: str = Field(default="", max_length=100)

    @field_validator("symbol")
    @classmethod
    def symbol_whitelist(cls, v: str) -> str:
        allowed = {"XAUUSD", "EURUSD", "GBPUSD", "USDJPY"}
        if v not in allowed:
            raise ValueError(f"Symbol '{v}' is not in the allowed list: {allowed}")
        return v

    @field_validator("comment")
    @classmethod
    def sanitize_comment(cls, v: str) -> str:
        # Strip potentially dangerous characters
        return re.sub(r"[^\w\s\-.,]", "", v)

    @model_validator(mode="after")
    def validate_sl_tp_logic(self) -> "OrderRequest":
        if self.action == "BUY" and self.stop_loss >= self.take_profit:
            raise ValueError("For BUY: stop_loss must be less than take_profit")
        if self.action == "SELL" and self.stop_loss <= self.take_profit:
            raise ValueError("For SELL: stop_loss must be greater than take_profit")
        return self


class BotConfigUpdate(BaseModel):
    model_config = {"extra": "forbid"}

    risk_percent: float = Field(..., gt=0.0, le=5.0,
                                 description="Risk per trade as % of balance")
    max_daily_trades: int = Field(..., ge=1, le=20)
    trading_enabled: bool


class LoginRequest(BaseModel):
    model_config = {"extra": "forbid"}

    username: str = Field(..., min_length=3, max_length=50,
                          pattern=r"^[a-zA-Z0-9_]+$")
    password: str = Field(..., min_length=8, max_length=128)
```

```python
# routers/trading.py
from fastapi import APIRouter
from schemas.trading import OrderRequest, BotConfigUpdate
from auth.dependencies import require_admin

router = APIRouter()

@router.post("/order")
async def place_order(body: OrderRequest, admin=Depends(require_admin)):
    # body is fully validated — safe to use
    ...

@router.put("/config")
async def update_config(body: BotConfigUpdate, admin=Depends(require_admin)):
    ...
```

> [!NOTE]
> The `model_config = {"extra": "forbid"}` setting is critical. Without it, clients can inject arbitrary extra fields that may confuse your logic or be used for parameter pollution attacks.

---

## 9. SQL Injection Prevention

### ✅ Always Use SQLAlchemy ORM or Parameterized Queries

```python
# db/trades.py — CORRECT
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from models import Trade

async def get_trades_by_symbol(session: AsyncSession, symbol: str) -> list[Trade]:
    # SQLAlchemy automatically parameterizes — no injection possible
    stmt = select(Trade).where(Trade.symbol == symbol).order_by(Trade.opened_at.desc())
    result = await session.execute(stmt)
    return result.scalars().all()

async def get_trade_by_id(session: AsyncSession, trade_id: int) -> Trade | None:
    return await session.get(Trade, trade_id)
```

```python
# If you must use raw SQL, ALWAYS use bound parameters
from sqlalchemy import text

async def raw_query_example(session: AsyncSession, symbol: str):
    # ✅ Safe: parameterized
    stmt = text("SELECT * FROM trades WHERE symbol = :symbol")
    result = await session.execute(stmt, {"symbol": symbol})
    return result.fetchall()
```

### ❌ Never Build SQL with String Formatting

```python
# WRONG — SQL injection vulnerability
symbol = request.query_params.get("symbol")
query = f"SELECT * FROM trades WHERE symbol = '{symbol}'"   # ← NEVER DO THIS

# Also WRONG
query = "SELECT * FROM trades WHERE symbol = '" + symbol + "'"
```

> [!CAUTION]
> An attacker can pass `symbol = "'; DROP TABLE trades; --"` and destroy your entire trading history database. Always use the ORM or bound parameters.

---

## 10. Telegram Token Protection

```python
# utils/telegram.py
import os
import logging

logger = logging.getLogger(__name__)

TELEGRAM_TOKEN       = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID     = int(os.environ["TELEGRAM_CHAT_ID"])
ADMIN_CHAT_IDS       = set(
    int(cid.strip())
    for cid in os.environ.get("TELEGRAM_ADMIN_CHAT_IDS", "").split(",")
    if cid.strip()
)

def _mask_token(token: str) -> str:
    """Show only first 10 chars for diagnostic logging."""
    return token[:10] + "..." if len(token) > 10 else "***"

# Log at startup for diagnostics — MASKED
logger.info("Telegram bot initialized with token: %s", _mask_token(TELEGRAM_TOKEN))

async def handle_message(update):
    """Reject messages from unauthorized chat IDs."""
    chat_id = update.message.chat_id
    if chat_id not in ADMIN_CHAT_IDS and chat_id != TELEGRAM_CHAT_ID:
        logger.warning("Rejected Telegram message from unauthorized chat_id=%d", chat_id)
        return   # Silently ignore

    # Process the message
    ...
```

> [!IMPORTANT]
> Restrict your bot to specific `TELEGRAM_ADMIN_CHAT_IDS`. Without this check, **anyone who discovers your bot username** can send commands to your trading system.

---

## 11. MT5 Credentials Protection

```python
# brokers/mt5_client.py
import MetaTrader5 as mt5
import os
import logging

logger = logging.getLogger(__name__)

MT5_LOGIN   = int(os.environ["MT5_LOGIN"])
MT5_PASSWORD = os.environ["MT5_PASSWORD"]
MT5_SERVER   = os.environ["MT5_SERVER"]

def initialize_mt5() -> bool:
    """Connect to MT5 broker. Never logs the password."""
    logger.info("Connecting to MT5: login=%d, server=%s", MT5_LOGIN, MT5_SERVER)
    # Note: password is passed but never logged
    result = mt5.initialize(
        login=MT5_LOGIN,
        password=MT5_PASSWORD,
        server=MT5_SERVER,
    )
    if not result:
        error = mt5.last_error()
        # Log the error code but NOT the credentials
        logger.error("MT5 connection failed. Error code: %s", error[0])
        return False

    logger.info("MT5 connected successfully. Account: %d", MT5_LOGIN)
    return True

def _safe_error_message(exc: Exception) -> str:
    """Remove any credentials that may appear in exception messages."""
    msg = str(exc)
    msg = msg.replace(MT5_PASSWORD, "***")
    return msg
```

> [!CAUTION]
> Some MT5 libraries may include the password in exception messages or debug output. Always filter credentials from exception strings before logging.

---

## 12. LLM API Key Protection

```python
# llm/openai_client.py
import os
import logging
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL   = os.environ.get("OPENAI_MODEL", "gpt-4o")

# Log only the first 8 characters for diagnostics
logger.info("OpenAI client initialized. Key prefix: %s...", OPENAI_API_KEY[:8])

client = AsyncOpenAI(api_key=OPENAI_API_KEY)


async def analyze_market(prompt: str) -> str:
    try:
        response = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as e:
        # Sanitize error: OpenAI errors may echo the request (which contains the key in headers)
        error_msg = str(e)
        if OPENAI_API_KEY in error_msg:
            error_msg = error_msg.replace(OPENAI_API_KEY, "sk-***")
        logger.error("LLM analysis failed: %s", error_msg)
        raise
```

### Key Rotation Schedule

| Frequency | Trigger |
|---|---|
| Every 90 days | Scheduled rotation (calendar reminder) |
| Immediately | On any suspicion of exposure |
| Immediately | When a developer leaves the team |
| Immediately | After a git history scrub |

---

## 13. Audit Logging

Every administrative action (start/stop bot, change config, place manual order, emergency stop) must be recorded to an **append-only audit log** with full context.

```python
# utils/audit.py
import logging
import json
from datetime import datetime, timezone
from typing import Any

# Dedicated audit logger — write to a separate file
audit_logger = logging.getLogger("audit")

def _setup_audit_logger():
    handler = logging.FileHandler("logs/audit.log", mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))  # Raw JSON lines
    audit_logger.addHandler(handler)
    audit_logger.setLevel(logging.INFO)
    audit_logger.propagate = False  # Do not send to root logger

_setup_audit_logger()


def log_action(
    actor: str,
    action: str,
    resource: str,
    outcome: str,        # "success" | "failure" | "denied"
    details: dict[str, Any] = None,
    request_ip: str = None,
):
    """
    Write an immutable audit log entry.

    Args:
        actor:      Username or system identity performing the action.
        action:     What was done (e.g., "BOT_START", "EMERGENCY_STOP", "CONFIG_UPDATE").
        resource:   What was acted upon (e.g., "trading_bot", "order:12345").
        outcome:    Result of the action.
        details:    Additional context (sanitized — no secrets).
        request_ip: Remote IP address of the request.
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "actor":      actor,
        "action":     action,
        "resource":   resource,
        "outcome":    outcome,
        "ip":         request_ip,
        "details":    details or {},
    }
    audit_logger.info(json.dumps(entry))
```

```python
# Usage in a route handler
from utils.audit import log_action

@router.post("/emergency-stop")
async def emergency_stop(request: Request, admin=Depends(require_admin)):
    try:
        await bot.emergency_stop()
        log_action(
            actor=admin["sub"],
            action="EMERGENCY_STOP",
            resource="trading_bot",
            outcome="success",
            request_ip=request.client.host,
        )
        return {"status": "stopped"}
    except Exception as e:
        log_action(
            actor=admin["sub"],
            action="EMERGENCY_STOP",
            resource="trading_bot",
            outcome="failure",
            details={"error": str(e)},
            request_ip=request.client.host,
        )
        raise
```

Sample audit log output (one JSON per line, append-only):

```json
{"timestamp":"2025-10-08T09:30:00Z","actor":"admin","action":"BOT_START","resource":"trading_bot","outcome":"success","ip":"192.168.1.10","details":{}}
{"timestamp":"2025-10-08T11:45:12Z","actor":"admin","action":"CONFIG_UPDATE","resource":"bot_config","outcome":"success","ip":"192.168.1.10","details":{"risk_percent":1.5,"max_daily_trades":5}}
{"timestamp":"2025-10-08T14:02:55Z","actor":"viewer_user","action":"BOT_STOP","resource":"trading_bot","outcome":"denied","ip":"10.0.0.5","details":{"reason":"insufficient_role"}}
```

> [!NOTE]
> Ship audit logs to a centralized, write-protected location (S3 with Object Lock, CloudWatch Logs, or an immutable SIEM). Local files can be deleted or tampered with.

---

## 14. Access Control & Role-Based Permissions

### Roles

| Role | Description |
|---|---|
| `admin` | Full access: start/stop bot, change config, place manual orders, emergency stop |
| `viewer` | Read-only: view dashboard, read logs, see trade history |

### Permission Matrix

| Action | `admin` | `viewer` |
|---|---|---|
| View dashboard / status | ✅ | ✅ |
| View trade history | ✅ | ✅ |
| View audit logs | ✅ | ❌ |
| Start bot | ✅ | ❌ |
| Stop bot | ✅ | ❌ |
| Emergency stop | ✅ | ❌ |
| Place manual order | ✅ | ❌ |
| Update bot config | ✅ | ❌ |
| Create / manage users | ✅ | ❌ |

### Implementation

```python
# auth/roles.py
from enum import StrEnum

class Role(StrEnum):
    ADMIN  = "admin"
    VIEWER = "viewer"

WRITE_ACTIONS = {
    "BOT_START", "BOT_STOP", "EMERGENCY_STOP",
    "MANUAL_ORDER", "CONFIG_UPDATE", "USER_MANAGEMENT",
}

def can_perform(role: str, action: str) -> bool:
    if role == Role.ADMIN:
        return True   # Admin can do everything
    if role == Role.VIEWER:
        return action not in WRITE_ACTIONS
    return False      # Unknown roles denied by default
```

```python
# FastAPI route protection
from auth.roles import Role

def require_role(*roles: Role):
    """Factory for role-checking dependencies."""
    def _check(current_user: dict = Depends(get_current_user)) -> dict:
        if current_user.get("role") not in roles:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return current_user
    return _check

# Usage
@router.post("/start")
async def start_bot(
    request: Request,
    user=Depends(require_role(Role.ADMIN))
):
    ...

@router.get("/status")
async def get_status(
    user=Depends(require_role(Role.ADMIN, Role.VIEWER))   # Both roles allowed
):
    ...
```

---

## 15. Security Checklist Before Going Live

Complete **every item** before deploying to production with real money.

### Credentials & Secrets

- [ ] **1.** All secrets are stored in environment variables or a secret manager — no hardcoded values anywhere in the codebase.
- [ ] **2.** `.env` file is listed in `.gitignore` and confirmed absent from Git history (`git log --all --full-history -- .env`).
- [ ] **3.** `.env.example` with dummy placeholder values is committed to the repo.
- [ ] **4.** `detect-secrets` pre-commit hook is installed and passing on all developers' machines.
- [ ] **5.** `JWT_SECRET_KEY` is at least 64 random bytes (`secrets.token_hex(64)`), unique per environment.
- [ ] **6.** Production uses AWS Secrets Manager, Vault, or equivalent — not a `.env` file.

### Authentication & Authorization

- [ ] **7.** All dashboard endpoints require a valid JWT (no unauthenticated routes except `/auth/login`).
- [ ] **8.** JWT access tokens expire in ≤ 30 minutes; refresh tokens in ≤ 7 days.
- [ ] **9.** Passwords are hashed with bcrypt (minimum 12 rounds) — no MD5, SHA1, or plain text.
- [ ] **10.** Role-based access control is enforced; viewers cannot trigger START/STOP/EMERGENCY actions.
- [ ] **11.** Login endpoint returns the same error message for wrong username and wrong password (prevents user enumeration).

### Network & Transport

- [ ] **12.** HTTPS is enforced; all HTTP traffic is redirected to HTTPS (301).
- [ ] **13.** TLS certificate is valid, from a trusted CA, and auto-renewing (e.g., certbot cron).
- [ ] **14.** TLS 1.0 and 1.1 are disabled; only TLS 1.2 and 1.3 are allowed.
- [ ] **15.** Security headers are set: `Strict-Transport-Security`, `X-Frame-Options`, `X-Content-Type-Options`.
- [ ] **16.** CORS `allowed_origins` lists only the exact production dashboard URL — no wildcards.
- [ ] **17.** API rate limiting is active: per-IP limits on all endpoints, stricter limits on write actions.

### Data & Input

- [ ] **18.** All API endpoints use Pydantic models with `extra="forbid"` — no unvalidated input reaches business logic.
- [ ] **19.** Database queries use SQLAlchemy ORM or parameterized `text()` — zero raw f-string SQL.
- [ ] **20.** Telegram bot rejects messages from any `chat_id` not in the configured whitelist.

### Logging & Monitoring

- [ ] **21.** No secrets (passwords, tokens, API keys) appear in any log file — verified with a grep scan: `grep -rn "sk-\|password\|token" logs/`.
- [ ] **22.** Audit log is active and writing a JSON entry for every admin action.
- [ ] **23.** Audit logs are shipped off the server to a write-protected location (S3, CloudWatch, etc.).
- [ ] **24.** An alert is configured for failed login attempts (threshold: > 5 failures in 1 minute from the same IP).

### MT5 & Broker

- [ ] **25.** MT5 password is never logged, never appears in error messages, and is never stored in plaintext.
- [ ] **26.** MT5 account has **IP whitelist** configured at the broker level (only your server's IP can connect).
- [ ] **27.** MT5 account uses a **read-only** sub-account for market data and a separate trading account — principle of least privilege.

---

*Last updated: 2025-10-08 — Trading-Gold Security Documentation*
