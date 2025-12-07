# Cloud LLM Chat Interface

A self-hosted LM Studio-like chat interface with scale-to-zero GPU backend.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Your Setup                           │
│                                                             │
│  ┌──────────────────┐         ┌─────────────────────────┐  │
│  │   Open WebUI     │────────▶│   GPU Backend (Modal)   │  │
│  │   (Railway)      │         │                         │  │
│  │                  │   API   │ • vLLM + Mistral-7B     │  │
│  │ • Chat interface │ ◀────── │ • Scales to zero        │  │
│  │ • Authentication │         │ • 2 min idle timeout    │  │
│  │ • Chat history   │         │                         │  │
│  │ • Model selector │         │ Cold start: ~30-60 sec  │  │
│  │                  │         │ Cost: ~$1/hr when used  │  │
│  │ Cost: ~$5/month  │         │                         │  │
│  └──────────────────┘         └─────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Deploy GPU Backend (Modal)

```bash
# Install Modal CLI
pip install modal
modal setup  # One-time auth

# Create API key secret (optional, for auth)
modal secret create api-auth API_KEY=your-secret-key

# Deploy
modal deploy -m src.app
```

Note your endpoint URL: `https://YOUR_USERNAME--llm-backend-api.modal.run`

### 2. Deploy Chat Interface (Railway)

#### Option A: One-Click Deploy
[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/open-webui)

#### Option B: From This Repo

1. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
2. Select this repository
3. Add a **Volume** mounted to `/app/backend/data` (for chat history)
4. Set environment variables:

```env
OPENAI_API_BASE_URL=https://YOUR_USERNAME--llm-backend-api.modal.run/v1
OPENAI_API_KEY=your-api-key
WEBUI_AUTH=true
WEBUI_NAME=LLM Studio
ENABLE_SIGNUP=false
PORT=8080
```

5. Deploy! First user to sign up becomes admin.

### 3. Use It

1. Open your Railway URL
2. Create an account (first user = admin)
3. Start chatting — the GPU backend wakes up on first message (~30-60s cold start)

## Authentication & User Management

### How Open WebUI Handles Auth

Open WebUI supports multiple authentication methods:

1. **Local Auth (Default)**: Email/password signup
   - First user automatically becomes **super admin**
   - Subsequent users are regular users
   - Admin can promote users to admin role

2. **OAuth/SSO**: Integrate with Google, Microsoft, GitHub, or any OIDC provider
   - Set `ENABLE_OAUTH_ROLE_MANAGEMENT=true` for role sync
   - See [Open WebUI SSO docs](https://docs.openwebui.com/features/sso)

3. **Trusted Header Auth**: For reverse proxy setups (Authelia, Authentik)

### Restrict to Specific Users

**Option 1: Disable Signups (Recommended)**

Set in Railway environment variables:

```env
ENABLE_SIGNUP=false
```

Then manually create users:
1. Log in as admin (first user)
2. Go to **Settings → Users**
3. Click **Add User** → Enter email/password
4. Assign role (user/admin)

**Option 2: OAuth with Domain Restriction**

Use OAuth (Google Workspace, Microsoft 365) and restrict by email domain in your identity provider.

**Option 3: Admin Approval**

Keep `ENABLE_SIGNUP=true` but:
1. Admin must approve new users
2. Or disable signups after initial setup

### Role-Based Access Control

- **Super Admin**: First user, cannot be demoted
- **Admin**: Can manage users, settings, models
- **User**: Can chat, manage own conversations

Admins can change user roles in **Settings → Users**.

## Local Development

```bash
# Run Open WebUI locally
docker compose up

# Access at http://localhost:3000
```

## CI/CD Deployment

This repository includes a GitHub Actions workflow that automatically deploys to Modal when changes are pushed to the `main` branch.

### Setting Up GitHub Actions Secrets

To enable automatic deployment, you need to configure Modal authentication secrets in your GitHub repository:

1. **Get your Modal token credentials:**
   - Go to [Modal Tokens](https://modal.com/tokens) (or run `modal token new` locally)
   - Create a new token or use an existing one
   - You'll need both the **Token ID** and **Token Secret**

2. **Add secrets to GitHub:**
   - Go to your repository on GitHub
   - Navigate to **Settings** → **Secrets and variables** → **Actions**
   - Click **New repository secret**
   - Add the following secrets:
     - **Name:** `MODAL_TOKEN_ID`
       - **Value:** Your Modal token ID
     - **Name:** `MODAL_TOKEN_SECRET`
       - **Value:** Your Modal token secret

3. **Verify deployment:**
   - Push a change to the `main` branch
   - Check the **Actions** tab in your GitHub repository
   - The workflow will automatically deploy `src.app` to Modal

**Note:** You can also manually trigger the deployment by going to **Actions** → **Deploy to Modal** → **Run workflow**.

### Manual Deployment

If you prefer to deploy manually or the workflow fails:

```bash
# Install Modal CLI
pip install modal

# Authenticate (one-time setup)
modal setup

# Deploy
modal deploy -m src.app
```

## Files

| File | Purpose |
|------|---------|
| `src/app.py` | GPU inference backend (Modal) |
| `Dockerfile` | Railway deployment config |
| `railway.toml` | Railway service settings |
| `docker-compose.yml` | Local development |

## Cost Breakdown

| Component | Cost | Notes |
|-----------|------|-------|
| Railway (Open WebUI) | ~$5/month | Always on, cheap container |
| Modal (GPU) | ~$0.50-1/hr | Only when chatting, scales to zero |

**Typical usage**: If you chat for 2 hours/day, GPU costs ~$30-60/month.

## Customization

### Change the Model

Edit `src/config.py` to modify the model registry:

```python
MODEL_REGISTRY: ModelRegistry = ModelRegistry(
    models={
        "mistralai/Mistral-7B-Instruct-v0.3": ModelConfig(
            model_id="mistralai/Mistral-7B-Instruct-v0.3",
            gpu_type="A10G",  # T4 is cheaper, A100 is faster
            max_model_len=4096,
        ),
        # Add more models here...
    },
)
```

### Adjust Scale-to-Zero Timeout

Edit `src/config.py`:

```python
CONTAINER_IDLE_TIMEOUT = 120  # Seconds before GPU shuts down
```

Lower = saves money, but more cold starts.
